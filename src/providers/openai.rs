//! # Provider: OpenAI
//!
//! ## Responsibility
//! Translate provider-agnostic [`ChatRequest`] / [`ChatResponse`] types to and
//! from the OpenAI Chat Completions API, including SSE streaming.
//!
//! ## Guarantees
//! - Cost calculation uses the pricing table defined in `model_price_per_1k`
//! - Streaming is parsed line-by-line from the `data:` SSE prefix
//! - All HTTP errors are mapped to typed [`LlmError`] variants
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

/// Base URL for the OpenAI API.
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

//  Wire types

#[derive(Debug, Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoiceMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorBody {
    error: OpenAiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorDetail {
    message: String,
}

//  Pricing

/// Returns (prompt_price_per_1k, completion_price_per_1k) in USD for a model.
fn model_price_per_1k(model: &str) -> (f64, f64) {
    match model {
        "gpt-4o" => (0.005, 0.015),
        "gpt-4o-mini" => (0.000150, 0.000600),
        "gpt-4-turbo" => (0.010, 0.030),
        "gpt-3.5-turbo" => (0.000500, 0.001500),
        "o1" => (0.015, 0.060),
        "o1-mini" => (0.003, 0.012),
        "o3-mini" => (0.0011, 0.0044),
        _ => (0.002, 0.002), // conservative default
    }
}

/// Compute the USD cost of a completed request.
fn compute_cost(model: &str, prompt_tokens: u32, completion_tokens: u32) -> f64 {
    let (prompt_price, completion_price) = model_price_per_1k(model);
    (prompt_tokens as f64 / 1000.0) * prompt_price
        + (completion_tokens as f64 / 1000.0) * completion_price
}

//  Provider

/// OpenAI API provider.
///
/// # Example
/// ```rust,no_run
/// # use tokio_llm::providers::openai::OpenAiProvider;
/// let provider = OpenAiProvider::new("sk-...");
/// ```
#[derive(Debug)]
pub struct OpenAiProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl OpenAiProvider {
    /// Create a new [`OpenAiProvider`] with the given API key.
    ///
    /// # Arguments
    /// * `api_key`  -  OpenAI secret key (starts with `sk-`)
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: Client::new(),
            base_url: OPENAI_BASE_URL.to_string(),
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

    fn build_request<'a>(&self, req: &'a ChatRequest, stream: bool) -> OpenAiRequest<'a> {
        OpenAiRequest {
            model: req.model.as_str(),
            messages: req
                .messages
                .iter()
                .map(|m| OpenAiMessage {
                    role: role_str(&m.role),
                    content: &m.content,
                })
                .collect(),
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: if stream { Some(true) } else { None },
        }
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
            .json::<OpenAiErrorBody>()
            .await
            .map(|b| b.error.message)
            .unwrap_or_else(|_| "unknown error".to_string());
        LlmError::ApiError { status, message }
    }
}

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    #[instrument(skip(self, req), fields(model = req.model.as_str()))]
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError> {
        let body = self.build_request(req, false);
        let url = format!("{}/chat/completions", self.base_url);
        debug!(url = %url, "sending OpenAI chat request");

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(Self::handle_error_response(resp).await);
        }

        let oai: OpenAiResponse = resp.json().await?;
        let content = oai
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        let model_id = oai.model.as_deref().unwrap_or(req.model.as_str());
        let (prompt_tok, completion_tok) = oai
            .usage
            .map(|u| (u.prompt_tokens, u.completion_tokens))
            .unwrap_or((0, 0));
        let cost = compute_cost(model_id, prompt_tok, completion_tok);
        let usage = Usage::new(prompt_tok, completion_tok, cost);

        Ok(ChatResponse {
            content,
            model: model_id.to_string(),
            usage,
            request_id: oai.id,
        })
    }

    #[instrument(skip(self, req), fields(model = req.model.as_str()))]
    async fn chat_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError> {
        let body = self.build_request(req, true);
        let url = format!("{}/chat/completions", self.base_url);
        debug!(url = %url, "sending OpenAI streaming chat request");

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
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
                        parse_sse_chunk(&text, &model)
                    }
                }
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }
}

/// Parse one or more SSE `data:` lines from a raw bytes chunk.
fn parse_sse_chunk(text: &str, model: &str) -> Vec<Result<StreamChunk, LlmError>> {
    let mut results = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || !line.starts_with("data:") {
            continue;
        }
        let payload = line.trim_start_matches("data:").trim();
        if payload == "[DONE]" {
            results.push(Ok(StreamChunk::final_chunk(None)));
            continue;
        }
        match serde_json::from_str::<OpenAiStreamChunk>(payload) {
            Err(e) => results.push(Err(LlmError::StreamError {
                message: format!("failed to parse SSE chunk: {e}"),
            })),
            Ok(chunk) => {
                let finish_reason = chunk
                    .choices
                    .first()
                    .and_then(|c| c.finish_reason.as_deref());
                let delta_text = chunk
                    .choices
                    .first()
                    .and_then(|c| c.delta.content.clone())
                    .unwrap_or_default();

                if finish_reason == Some("stop") {
                    let usage = chunk.usage.map(|u| {
                        let cost = compute_cost(model, u.prompt_tokens, u.completion_tokens);
                        Usage::new(u.prompt_tokens, u.completion_tokens, cost)
                    });
                    results.push(Ok(StreamChunk::final_chunk(usage)));
                } else if !delta_text.is_empty() {
                    results.push(Ok(StreamChunk::delta(delta_text)));
                }
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Model;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    //  Unit tests

    #[test]
    fn test_role_str_system() {
        assert_eq!(role_str(&Role::System), "system");
    }

    #[test]
    fn test_role_str_user() {
        assert_eq!(role_str(&Role::User), "user");
    }

    #[test]
    fn test_role_str_assistant() {
        assert_eq!(role_str(&Role::Assistant), "assistant");
    }

    #[test]
    fn test_compute_cost_gpt4o_mini() {
        // 1000 prompt tokens @ $0.000150/1k = $0.000150
        // 500 completion tokens @ $0.000600/1k = $0.000300
        let cost = compute_cost("gpt-4o-mini", 1000, 500);
        assert!((cost - 0.000450).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_unknown_model_uses_default() {
        let cost = compute_cost("unknown-model", 1000, 1000);
        // default: 0.002/1k for both
        assert!((cost - 0.004).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_zero_tokens() {
        assert_eq!(compute_cost("gpt-4o", 0, 0), 0.0);
    }

    #[test]
    fn test_model_price_all_known_models_nonzero() {
        let models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3-mini",
        ];
        for m in &models {
            let (p, c) = model_price_per_1k(m);
            assert!(p > 0.0, "prompt price for {m} should be > 0");
            assert!(c > 0.0, "completion price for {m} should be > 0");
        }
    }

    #[test]
    fn test_parse_sse_chunk_delta() {
        let sse = r#"data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        let results = parse_sse_chunk(sse, "gpt-4o");
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
    fn test_parse_sse_chunk_done() {
        let sse = "data: [DONE]";
        let results = parse_sse_chunk(sse, "gpt-4o");
        assert_eq!(results.len(), 1);
        assert!(results[0].as_ref().map(|c| c.is_final).unwrap_or(false));
    }

    #[test]
    fn test_parse_sse_chunk_stop_finish_reason() {
        let sse = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let results = parse_sse_chunk(sse, "gpt-4o");
        assert_eq!(results.len(), 1);
        assert!(results[0].as_ref().map(|c| c.is_final).unwrap_or(false));
    }

    #[test]
    fn test_parse_sse_chunk_invalid_json_returns_stream_error() {
        let sse = "data: {not valid json}";
        let results = parse_sse_chunk(sse, "gpt-4o");
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0], Err(LlmError::StreamError { .. })));
    }

    #[test]
    fn test_parse_sse_chunk_empty_string_no_results() {
        let results = parse_sse_chunk("", "gpt-4o");
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_sse_chunk_multiple_lines() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"A\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"B\"},\"finish_reason\":null}]}\n\
                   data: [DONE]";
        let results = parse_sse_chunk(sse, "gpt-4o");
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].as_ref().map(|c| c.delta.as_str()).unwrap_or(""),
            "A"
        );
        assert_eq!(
            results[1].as_ref().map(|c| c.delta.as_str()).unwrap_or(""),
            "B"
        );
        assert!(results[2].as_ref().map(|c| c.is_final).unwrap_or(false));
    }

    #[test]
    fn test_parse_sse_ignores_non_data_lines() {
        let sse = "event: content_block_delta\ndata: [DONE]";
        let results = parse_sse_chunk(sse, "gpt-4o");
        assert_eq!(results.len(), 1);
        assert!(results[0].as_ref().map(|c| c.is_final).unwrap_or(false));
    }

    #[test]
    fn test_openai_provider_new_stores_key() {
        let p = OpenAiProvider::new("sk-test");
        assert_eq!(p.api_key, "sk-test");
    }

    #[test]
    fn test_openai_provider_with_base_url() {
        let p = OpenAiProvider::with_base_url("sk-test", "http://localhost:8080");
        assert_eq!(p.base_url, "http://localhost:8080");
    }

    //  Integration tests with wiremock

    fn chat_request() -> ChatRequest {
        use crate::types::Message;
        ChatRequest::new(Model::Gpt4oMini, vec![Message::user("Say hello")]).with_max_tokens(50)
    }

    #[tokio::test]
    async fn test_chat_success_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("sk-test", server.uri());
        let req = chat_request();
        let resp = provider.chat(&req).await;
        assert!(resp.is_ok(), "expected Ok, got {resp:?}");
        let r = resp.unwrap_or_else(|_| ChatResponse {
            content: String::new(),
            model: String::new(),
            usage: Usage::default(),
            request_id: None,
        });
        assert_eq!(r.content, "Hello!");
        assert_eq!(r.model, "gpt-4o-mini");
        assert_eq!(r.usage.prompt_tokens, 10);
        assert_eq!(r.usage.completion_tokens, 5);
        assert_eq!(r.request_id.as_deref(), Some("chatcmpl-123"));
    }

    #[tokio::test]
    async fn test_chat_400_returns_api_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {"message": "invalid request", "type": "invalid_request_error"}
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("sk-test", server.uri());
        match provider.chat(&chat_request()).await {
            Err(LlmError::ApiError { status, message }) => {
                assert_eq!(status, 400);
                assert!(message.contains("invalid request"));
            }
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_429_returns_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429)
                    .append_header("retry-after", "30")
                    .set_body_json(serde_json::json!({
                        "error": {"message": "rate limit exceeded"}
                    })),
            )
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("sk-test", server.uri());
        match provider.chat(&chat_request()).await {
            Err(LlmError::RateLimited { retry_after_secs }) => {
                assert_eq!(retry_after_secs, Some(30));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_500_returns_api_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_json(serde_json::json!({
                "error": {"message": "internal server error"}
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("sk-test", server.uri());
        match provider.chat(&chat_request()).await {
            Err(LlmError::ApiError { status, .. }) => assert_eq!(status, 500),
            other => panic!("expected ApiError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_cost_calculated_correctly() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-abc",
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500}
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("sk-test", server.uri());
        let resp = provider
            .chat(&chat_request())
            .await
            .unwrap_or_else(|_| ChatResponse {
                content: String::new(),
                model: String::new(),
                usage: Usage::default(),
                request_id: None,
            });
        // gpt-4o-mini: 0.000150/1k prompt, 0.000600/1k completion
        // 1000 prompt = 0.000150; 500 completion = 0.000300 → total 0.000450
        assert!((resp.usage.cost_usd - 0.000450).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_chat_stream_yields_deltas() {
        let server = MockServer::start().await;
        let sse_body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
                        data: [DONE]\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url("sk-test", server.uri());
        let mut req = chat_request();
        req.stream = Some(true);

        let stream_result = provider.chat_stream(&req).await;
        assert!(stream_result.is_ok(), "expected Ok stream");
        let mut stream = stream_result.unwrap_or_else(|_| Box::pin(futures::stream::empty()));
        let mut collected = Vec::new();
        while let Some(chunk) = stream.next().await {
            collected.push(chunk);
        }
        // Should have: "Hello", " world", [DONE]
        assert!(collected.len() >= 2);
        assert!(collected
            .iter()
            .any(|c| c.as_ref().map(|ch| ch.is_final).unwrap_or(false)));
    }
}
