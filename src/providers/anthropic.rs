//! # Provider: Anthropic
//!
//! ## Responsibility
//! Translate provider-agnostic [`ChatRequest`] / [`ChatResponse`] types to and
//! from the Anthropic Messages API, including SSE streaming.
//!
//! ## Guarantees
//! - Uses the `anthropic-version: 2023-06-01` header on all requests
//! - Cost calculation uses the pricing table defined in `model_price_per_1k`
//! - System messages are extracted and sent via the top-level `system` field
//! - SSE stream is parsed from `data:` prefixed event lines
//!
//! ## NOT Responsible For
//! - Retry logic (see [`crate::retry`])
//! - Circuit breaking (see [`crate::circuit_breaker`])
//! - Budget enforcement (see [`crate::budget`])

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::error::LlmError;
use crate::types::{ChatRequest, ChatResponse, Role, StreamChunk, Usage};

use super::Provider;

/// Base URL for the Anthropic Messages API.
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
/// Required API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

//  Wire types 

#[derive(Debug, Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: Option<String>,
    model: Option<String>,
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicStreamDelta>,
    usage: Option<AnthropicStreamUsage>,
    // Present on message_start events; kept for future input-token tracking
    #[allow(dead_code)]
    message: Option<AnthropicStreamMessage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamUsage {
    output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamMessage {
    // Carries input_tokens on message_start; kept for future budget pre-charging
    #[allow(dead_code)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    message: String,
}

//  Pricing 

/// Returns (input_price_per_1k, output_price_per_1k) in USD for a model.
fn model_price_per_1k(model: &str) -> (f64, f64) {
    match model {
        "claude-3-5-sonnet-20241022" => (0.003, 0.015),
        "claude-3-5-haiku-20241022" => (0.001, 0.005),
        "claude-3-opus-20240229" => (0.015, 0.075),
        "claude-3-sonnet-20240229" => (0.003, 0.015),
        "claude-3-haiku-20240307" => (0.00025, 0.00125),
        _ => (0.003, 0.015), // conservative default matching Sonnet
    }
}

/// Compute the USD cost of a completed Anthropic request.
fn compute_cost(model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
    let (input_price, output_price) = model_price_per_1k(model);
    (input_tokens as f64 / 1000.0) * input_price
        + (output_tokens as f64 / 1000.0) * output_price
}

//  Provider 

/// Anthropic Messages API provider.
///
/// # Example
/// ```rust,no_run
/// # use tokio_llm::providers::anthropic::AnthropicProvider;
/// let provider = AnthropicProvider::new("sk-ant-...");
/// ```
#[derive(Debug)]
pub struct AnthropicProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl AnthropicProvider {
    /// Create a new [`AnthropicProvider`] with the given API key.
    ///
    /// # Arguments
    /// * `api_key`  -  Anthropic secret key (starts with `sk-ant-`)
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            base_url: ANTHROPIC_BASE_URL.to_string(),
        }
    }

    /// Create a provider pointing at a custom base URL (for testing/proxies).
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Extract optional system message and non-system messages from a request.
    fn split_messages<'a>(
        req: &'a ChatRequest,
    ) -> (Option<&'a str>, Vec<AnthropicMessage<'a>>) {
        let system = req
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.as_str());

        let messages = req
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "user", // should be filtered, but safe fallback
                },
                content: &m.content,
            })
            .collect();

        (system, messages)
    }

    async fn handle_error_response(resp: reqwest::Response) -> LlmError {
        let status = resp.status().as_u16();
        if status == 429 {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            return LlmError::RateLimited {
                retry_after_secs: retry_after,
            };
        }
        let message = resp
            .json::<AnthropicErrorBody>()
            .await
            .map(|b| b.error.message)
            .unwrap_or_else(|_| "unknown error".to_string());
        LlmError::ApiError { status, message }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    #[instrument(skip(self, req), fields(model = req.model.as_str()))]
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let (system, messages) = Self::split_messages(req);
        let max_tokens = req.max_tokens.unwrap_or(1024);
        let body = AnthropicRequest {
            model: req.model.as_str(),
            messages,
            max_tokens,
            system,
            temperature: req.temperature,
            stream: None,
        };

        let url = format!("{}/messages", self.base_url);
        debug!(url = %url, "sending Anthropic chat request");

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(Self::handle_error_response(resp).await);
        }

        let anth: AnthropicResponse = resp.json().await?;
        let content = anth
            .content
            .into_iter()
            .filter(|b| b.block_type == "text")
            .filter_map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        let model_id = anth.model.as_deref().unwrap_or(req.model.as_str());
        let (input_tok, output_tok) = anth
            .usage
            .map(|u| (u.input_tokens, u.output_tokens))
            .unwrap_or((0, 0));
        let cost = compute_cost(model_id, input_tok, output_tok);
        let usage = Usage::new(input_tok, output_tok, cost);

        Ok(ChatResponse {
            content,
            model: model_id.to_string(),
            usage,
            request_id: anth.id,
        })
    }

    #[instrument(skip(self, req), fields(model = req.model.as_str()))]
    async fn chat_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError> {
        let (system, messages) = Self::split_messages(req);
        let max_tokens = req.max_tokens.unwrap_or(1024);
        let body = AnthropicRequest {
            model: req.model.as_str(),
            messages,
            max_tokens,
            system,
            temperature: req.temperature,
            stream: Some(true),
        };

        let url = format!("{}/messages", self.base_url);
        debug!(url = %url, "sending Anthropic streaming chat request");

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(Self::handle_error_response(resp).await);
        }

        let model = req.model.as_str().to_string();
        let byte_stream = resp.bytes_stream();

        let stream = byte_stream
            .map(move |chunk_result| {
                let model = model.clone();
                match chunk_result {
                    Err(e) => vec![Err(LlmError::Transport(e))],
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        parse_anthropic_sse(&text, &model)
                    }
                }
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

/// Parse Anthropic SSE events from a raw bytes chunk.
fn parse_anthropic_sse(text: &str, model: &str) -> Vec<Result<StreamChunk, LlmError>> {
    let mut results = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with("data:") {
            continue;
        }
        let payload = line.trim_start_matches("data:").trim();
        match serde_json::from_str::<AnthropicStreamEvent>(payload) {
            Err(e) => results.push(Err(LlmError::StreamError {
                message: format!("failed to parse Anthropic SSE event: {e}"),
            })),
            Ok(event) => match event.event_type.as_str() {
                "content_block_delta" => {
                    if let Some(delta) = &event.delta {
                        if delta.delta_type.as_deref() == Some("text_delta") {
                            if let Some(text) = &delta.text {
                                if !text.is_empty() {
                                    results.push(Ok(StreamChunk::delta(text.clone())));
                                }
                            }
                        }
                    }
                }
                "message_delta" => {
                    if let Some(delta) = &event.delta {
                        if delta.stop_reason.as_deref() == Some("end_turn") {
                            // Try to get final usage from message_start if available
                            let usage = event.usage.and_then(|u| u.output_tokens).map(|out| {
                                let cost = compute_cost(model, 0, out);
                                Usage::new(0, out, cost)
                            });
                            results.push(Ok(StreamChunk::final_chunk(usage)));
                        }
                    }
                }
                "message_stop" => {
                    results.push(Ok(StreamChunk::final_chunk(None)));
                }
                "message_start" => {
                    // Contains initial usage (input tokens); we record but don't emit a chunk
                }
                _ => {
                    // Ignore unknown event types gracefully
                }
            },
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, Model};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    //  Unit tests 

    #[test]
    fn test_compute_cost_claude35_sonnet() {
        // 1000 input @ 0.003/1k = 0.003
        // 500 output @ 0.015/1k = 0.0075
        let cost = compute_cost("claude-3-5-sonnet-20241022", 1000, 500);
        assert!((cost - 0.0105).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_zero_tokens() {
        assert_eq!(compute_cost("claude-3-haiku-20240307", 0, 0), 0.0);
    }

    #[test]
    fn test_compute_cost_unknown_uses_default() {
        let cost = compute_cost("unknown", 1000, 1000);
        // default: 0.003 + 0.015 = 0.018
        assert!((cost - 0.018).abs() < 1e-9);
    }

    #[test]
    fn test_model_price_all_known_models_nonzero() {
        let models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ];
        for m in &models {
            let (i, o) = model_price_per_1k(m);
            assert!(i > 0.0, "input price for {m}");
            assert!(o > 0.0, "output price for {m}");
        }
    }

    #[test]
    fn test_split_messages_separates_system() {
        let req = ChatRequest::new(
            Model::Claude35Sonnet,
            vec![Message::system("be helpful"), Message::user("hello")],
        );
        let (system, messages) = AnthropicProvider::split_messages(&req);
        assert_eq!(system, Some("be helpful"));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    }

    #[test]
    fn test_split_messages_no_system() {
        let req = ChatRequest::new(
            Model::Claude35Sonnet,
            vec![Message::user("hello"), Message::assistant("hi")],
        );
        let (system, messages) = AnthropicProvider::split_messages(&req);
        assert!(system.is_none());
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_parse_anthropic_sse_text_delta() {
        let sse = r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}"#;
        let results = parse_anthropic_sse(sse, "claude-3-5-sonnet-20241022");
        assert_eq!(results.len(), 1);
        match &results[0] {
            Ok(chunk) => {
                assert_eq!(chunk.delta, "Hello");
                assert!(!chunk.is_final);
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn test_parse_anthropic_sse_message_stop() {
        let sse = r#"data: {"type":"message_stop"}"#;
        let results = parse_anthropic_sse(sse, "claude-3-haiku-20240307");
        assert_eq!(results.len(), 1);
        assert!(results[0].as_ref().map(|c| c.is_final).unwrap_or(false));
    }

    #[test]
    fn test_parse_anthropic_sse_message_delta_end_turn() {
        let sse = r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#;
        let results = parse_anthropic_sse(sse, "claude-3-5-sonnet-20241022");
        assert_eq!(results.len(), 1);
        assert!(results[0].as_ref().map(|c| c.is_final).unwrap_or(false));
    }

    #[test]
    fn test_parse_anthropic_sse_invalid_json_returns_stream_error() {
        let sse = "data: {not valid json}";
        let results = parse_anthropic_sse(sse, "claude-3-5-sonnet-20241022");
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], Err(LlmError::StreamError { .. })));
    }

    #[test]
    fn test_parse_anthropic_sse_empty_no_results() {
        assert!(parse_anthropic_sse("", "claude-3-haiku-20240307").is_empty());
    }

    #[test]
    fn test_parse_anthropic_sse_ignores_non_data_lines() {
        let sse = "event: content_block_delta\ndata: {\"type\":\"message_stop\"}";
        let results = parse_anthropic_sse(sse, "claude-3-5-sonnet-20241022");
        assert_eq!(results.len(), 1);
        assert!(results[0].as_ref().map(|c| c.is_final).unwrap_or(false));
    }

    #[test]
    fn test_parse_anthropic_sse_message_start_no_chunk() {
        let sse = r#"data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}"#;
        let results = parse_anthropic_sse(sse, "claude-3-5-sonnet-20241022");
        // message_start should produce no output chunks
        assert!(results.is_empty());
    }

    #[test]
    fn test_anthropic_provider_new_stores_key() {
        let p = AnthropicProvider::new("sk-ant-test");
        assert_eq!(p.api_key, "sk-ant-test");
    }

    #[test]
    fn test_anthropic_provider_with_base_url() {
        let p = AnthropicProvider::with_base_url("sk-ant-test", "http://localhost:9090");
        assert_eq!(p.base_url, "http://localhost:9090");
    }

    //  Integration tests with wiremock 

    fn chat_request() -> ChatRequest {
        ChatRequest::new(
            Model::Claude35Haiku,
            vec![Message::system("be brief"), Message::user("Say hello")],
        )
        .with_max_tokens(50)
    }

    #[tokio::test]
    async fn test_chat_success_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_123",
                "model": "claude-3-5-haiku-20241022",
                "content": [{"type": "text", "text": "Hello!"}],
                "usage": {"input_tokens": 12, "output_tokens": 3}
            })))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("sk-ant-test", server.uri());
        let resp = provider.chat(&chat_request()).await;
        assert!(resp.is_ok(), "expected Ok, got {resp:?}");
        let r = resp.unwrap_or_else(|_| ChatResponse {
            content: String::new(),
            model: String::new(),
            usage: Usage::default(),
            request_id: None,
        });
        assert_eq!(r.content, "Hello!");
        assert_eq!(r.usage.prompt_tokens, 12);
        assert_eq!(r.usage.completion_tokens, 3);
        assert_eq!(r.request_id.as_deref(), Some("msg_123"));
    }

    #[tokio::test]
    async fn test_chat_400_returns_api_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {"type": "invalid_request_error", "message": "bad input"}
            })))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("sk-ant-test", server.uri());
        match provider.chat(&chat_request()).await {
            Err(LlmError::ApiError { status, message }) => {
                assert_eq!(status, 400);
                assert!(message.contains("bad input"));
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_429_returns_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(
                ResponseTemplate::new(429)
                    .append_header("retry-after", "60")
                    .set_body_json(serde_json::json!({
                        "error": {"message": "rate limit exceeded"}
                    })),
            )
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("sk-ant-test", server.uri());
        match provider.chat(&chat_request()).await {
            Err(LlmError::RateLimited { retry_after_secs }) => {
                assert_eq!(retry_after_secs, Some(60));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_multiple_content_blocks_joined() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_456",
                "model": "claude-3-5-haiku-20241022",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2}
            })))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("sk-ant-test", server.uri());
        let resp = provider.chat(&chat_request()).await.unwrap_or_else(|_| ChatResponse {
            content: String::new(),
            model: String::new(),
            usage: Usage::default(),
            request_id: None,
        });
        assert_eq!(resp.content, "Hello world");
    }

    #[tokio::test]
    async fn test_chat_stream_yields_chunks() {
        let server = MockServer::start().await;
        let sse_body = "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\
                        data: {\"type\":\"message_stop\"}\n";
        Mock::given(method("POST"))
            .and(path("/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("sk-ant-test", server.uri());
        let mut req = chat_request();
        req.stream = Some(true);

        let stream_result = provider.chat_stream(&req).await;
        assert!(stream_result.is_ok());
        let mut stream = stream_result.unwrap_or_else(|_| Box::pin(futures::stream::empty()));
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.push(chunk);
        }
        assert!(!collected.is_empty());
        assert!(collected.iter().any(|c| c.as_ref().map(|ch| ch.is_final).unwrap_or(false)));
    }
}
