# tokio-llm

[![CI](https://github.com/Mattbusel/tokio-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/tokio-llm/actions/workflows/ci.yml)

An async Rust client for OpenAI and Anthropic chat APIs with retry, a circuit breaker, a USD budget cap and SSE streaming built in.

Calling an LLM in production is more than one HTTP request: you need backoff on 429s and 5xx, you need to stop hammering a provider that is down, and you need to know what you are spending. `tokio-llm` wraps both providers behind one `LlmClient` with those pieces already wired together, typed errors for every failure, and `tracing` spans on each request.

## Features

| Feature | Details |
|---|---|
| **Two providers, one API** | `LlmClient::openai(key)` (Chat Completions) and `LlmClient::anthropic(key)` (Messages API); system messages are mapped correctly for each |
| **Retry** | `RetryPolicy::exponential(attempts, base_delay)` with jitter and a delay cap; only rate limits, timeouts and HTTP 500/502/503/504 are retried |
| **Circuit breaker** | Closed, Open, HalfOpen state machine: opens after N consecutive failures, probes again after a timeout |
| **Budget cap** | Lock-free USD spend tracker; once the cap is reached `chat` returns `LlmError::BudgetExceeded` |
| **Streaming** | `chat_stream` returns a `Stream<Item = Result<StreamChunk, LlmError>>` for both providers |
| **Cost per call** | `ChatResponse.usage` carries prompt tokens, completion tokens and `cost_usd` from a built-in price table |
| **Custom endpoints** | `OpenAiProvider::with_base_url` / `AnthropicProvider::with_base_url` plus `LlmClient::with_provider` for proxies, gateways or OpenAI-compatible servers |
| **No panics** | `unwrap`, `expect` and `panic` are denied by Clippy lints |

## Install

Not published on crates.io yet; use the git dependency:

```toml
[dependencies]
tokio-llm = { git = "https://github.com/Mattbusel/tokio-llm" }
tokio = { version = "1", features = ["full"] }
futures = "0.3" # for streaming
```

## Quick start

```rust
use std::time::Duration;
use tokio_llm::{ChatRequest, LlmClient, LlmError, Message, Model, RetryPolicy};

#[tokio::main]
async fn main() -> Result<(), LlmError> {
    let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
    let client = LlmClient::openai(key)
        .with_retry(RetryPolicy::exponential(3, Duration::from_millis(200)))
        .with_budget(5.0)                                    // hard cap in USD
        .with_circuit_breaker(5, Duration::from_secs(30))    // open after 5 failures
        .build()?;

    let req = ChatRequest::new(
        Model::Gpt4oMini,
        vec![
            Message::system("You are a concise assistant."),
            Message::user("What is the capital of France?"),
        ],
    )
    .with_max_tokens(50);

    match client.chat(req).await {
        Ok(resp) => {
            println!("{}", resp.content);
            println!("cost ${:.6}, remaining ${:.4}", resp.usage.cost_usd, client.remaining_budget().unwrap_or(0.0));
        }
        Err(LlmError::RateLimited { retry_after_secs }) => eprintln!("rate limited, retry after {retry_after_secs:?}s"),
        Err(LlmError::BudgetExceeded { spent, limit }) => eprintln!("spent ${spent:.4} of ${limit:.4}"),
        Err(LlmError::CircuitOpen { reset_after_secs }) => eprintln!("provider down, retry in {reset_after_secs:.1}s"),
        Err(e) => eprintln!("error: {e}"),
    }
    Ok(())
}
```

Anthropic is the same code with a different constructor and model:

```rust
use tokio_llm::{ChatRequest, LlmClient, LlmError, Message, Model};

async fn ask_claude() -> Result<String, LlmError> {
    let client = LlmClient::anthropic(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default()).build()?;
    let req = ChatRequest::new(Model::Custom("claude-sonnet-4-5".into()), vec![Message::user("Hello!")]);
    Ok(client.chat(req).await?.content)
}
```

### Streaming

```rust
use futures::StreamExt;
use tokio_llm::{ChatRequest, LlmClient, LlmError, Message, Model};

async fn haiku(client: &LlmClient) -> Result<(), LlmError> {
    let req = ChatRequest::new(Model::Gpt4oMini, vec![Message::user("Write a haiku about Rust.")]);
    let mut stream = client.chat_stream(req).await?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.is_final {
            break;
        }
        print!("{}", chunk.delta);
    }
    Ok(())
}
```

## Models

`Model` has named variants for `Gpt4o`, `Gpt4oMini`, `Gpt4Turbo`, `Gpt35Turbo`, `O1`, `O1Mini`, `O3Mini`, `Claude35Sonnet`, `Claude35Haiku`, `Claude3Opus`, `Claude3Sonnet` and `Claude3Haiku`. Use `Model::Custom("model-id".into())` for anything else, including newer models; the provider is chosen by the client you built, not by the model name.

## How it works

```
LlmClient::chat(req)
  -> RetryPolicy        loop with exponential backoff + jitter
     -> CircuitBreaker  rejects immediately while Open
        -> Provider     OpenAiProvider | AnthropicProvider (reqwest, JSON or SSE)
  -> BudgetEnforcer     records usage.cost_usd, errors once the cap is crossed
```

| File | What it holds |
|---|---|
| `src/client.rs` | `LlmClient`, `ClientBuilder` |
| `src/providers/openai.rs`, `anthropic.rs` | request/response mapping, SSE parsing, price tables |
| `src/retry.rs` | `RetryPolicy` |
| `src/circuit_breaker.rs` | `CircuitBreaker`, `CircuitState` |
| `src/budget.rs` | `BudgetEnforcer` (atomic compare-and-swap on an `f64` bit pattern) |
| `src/types.rs`, `src/error.rs` | `ChatRequest`, `ChatResponse`, `Message`, `Model`, `Usage`, `StreamChunk`, `LlmError` |

Provider tests run against a local `wiremock` server, so `cargo test` needs no API keys.

## Status and limitations

Version 0.1.

- The budget is charged after a response arrives, so the call that crosses the cap has already been paid for; its response is returned as `BudgetExceeded`.
- Streaming calls are not retried and are not charged to the budget automatically.
- Price tables cover the named `Model` variants; `Custom` models are priced with a fallback default, so treat their `cost_usd` as an estimate.
- Text chat only: no tool calling, images or embeddings.

```bash
cargo test
```

---

Part of a set of Rust crates for LLM agents, see [rust-crates](https://github.com/Mattbusel/rust-crates).
