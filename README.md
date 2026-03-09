# tokio-llm

[![Crates.io](https://img.shields.io/crates/v/tokio-llm.svg)](https://crates.io/crates/tokio-llm)
[![Docs.rs](https://docs.rs/tokio-llm/badge.svg)](https://docs.rs/tokio-llm)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/tokio-prompt/ci.yml)](https://github.com/your-org/tokio-prompt/actions)

A Tokio-native async LLM client for **OpenAI** and **Anthropic** with built-in
retry, circuit breaking, streaming, budget enforcement, and structured tracing.

---

## Quick Start

### OpenAI

```rust
use tokio_llm::client::LlmClient;
use tokio_llm::types::{ChatRequest, Message, Model};

#[tokio::main]
async fn main() -> Result<(), tokio_llm::error::LlmError> {
    let client = LlmClient::openai("sk-...").build()?;

    let req = ChatRequest::new(
        Model::Gpt4oMini,
        vec![Message::user("What is the capital of France?")],
    );
    let resp = client.chat(req).await?;
    println!("{}", resp.content);   // "Paris"
    println!("Cost: ${:.6}", resp.usage.cost_usd);
    Ok(())
}
```

### Anthropic

```rust
use tokio_llm::client::LlmClient;
use tokio_llm::types::{ChatRequest, Message, Model};

#[tokio::main]
async fn main() -> Result<(), tokio_llm::error::LlmError> {
    let client = LlmClient::anthropic("sk-ant-...").build()?;

    let req = ChatRequest::new(
        Model::Claude35Haiku,
        vec![
            Message::system("You are a concise assistant."),
            Message::user("Hello!"),
        ],
    );
    let resp = client.chat(req).await?;
    println!("{}", resp.content);
    Ok(())
}
```

---

## Features

| Feature | Description |
|---|---|
| **Dual provider** | OpenAI (GPT-4o, o1, o3) and Anthropic (Claude 3.5) via a unified API |
| **Retry** | Exponential backoff with 25% jitter; configurable attempt count |
| **Circuit breaker** | Closed → Open → HalfOpen state machine; prevents cascade failures |
| **Budget enforcement** | Lock-free atomic USD spend tracker; hard spending limits |
| **Streaming** | SSE-native `Stream<Item = StreamChunk>` for both providers |
| **Tracing** | `tracing` instrumentation on every request path |
| **Zero panics** | No `unwrap`/`expect`/`panic!` in production code paths |
| **Typed errors** | Every failure mode is a named `LlmError` variant, matchable exhaustively |

---

## Builder Pattern: Full Configuration

```rust
use tokio_llm::client::LlmClient;
use tokio_llm::retry::RetryPolicy;
use std::time::Duration;

let client = LlmClient::openai("sk-...")
    // Retry up to 3 times with 200ms base exponential backoff + jitter
    .with_retry(RetryPolicy::exponential(3, Duration::from_millis(200)))
    // Hard USD spending cap  -  BudgetExceeded returned if exceeded
    .with_budget(5.0)
    // Open circuit after 5 consecutive failures; probe again after 30s
    .with_circuit_breaker(5, Duration::from_secs(30))
    .build()?;

println!("Remaining budget: ${:.2}", client.remaining_budget().unwrap_or(0.0));
```

---

## Streaming

```rust
use tokio_llm::client::LlmClient;
use tokio_llm::types::{ChatRequest, Message, Model};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), tokio_llm::error::LlmError> {
    let client = LlmClient::openai("sk-...").build()?;

    let req = ChatRequest::new(
        Model::Gpt4oMini,
        vec![Message::user("Write me a haiku.")],
    )
    .with_max_tokens(100);

    let mut stream = client.chat_stream(req).await?;
    while let Some(chunk) = stream.next().await {
        match chunk? {
            c if c.is_final => break,
            c => print!("{}", c.delta),
        }
    }
    Ok(())
}
```

---

## Error Handling

```rust
use tokio_llm::error::LlmError;

match client.chat(req).await {
    Ok(resp) => println!("{}", resp.content),
    Err(LlmError::RateLimited { retry_after_secs }) => {
        eprintln!("Rate limited; retry after {retry_after_secs:?}s");
    }
    Err(LlmError::BudgetExceeded { spent, limit }) => {
        eprintln!("Spent ${spent:.4} of ${limit:.4} budget");
    }
    Err(LlmError::CircuitOpen { reset_after_secs }) => {
        eprintln!("Circuit open; will reset in {reset_after_secs:.1}s");
    }
    Err(e) => eprintln!("Error: {e}"),
}
```

---

## Supported Models

### OpenAI
- `Model::Gpt4o`  -  GPT-4o (flagship multimodal)
- `Model::Gpt4oMini`  -  GPT-4o mini (fast and cheap)
- `Model::Gpt4Turbo`  -  GPT-4 Turbo
- `Model::Gpt35Turbo`  -  GPT-3.5 Turbo
- `Model::O1`, `Model::O1Mini`, `Model::O3Mini`  -  reasoning models

### Anthropic
- `Model::Claude35Sonnet`  -  Claude 3.5 Sonnet (best balance)
- `Model::Claude35Haiku`  -  Claude 3.5 Haiku (fastest)
- `Model::Claude3Opus`  -  Claude 3 Opus (most capable)
- `Model::Claude3Sonnet`, `Model::Claude3Haiku`

### Custom
```rust
Model::Custom("my-fine-tuned-model".into())
```

---

## See Also

- [tokio-prompt-orchestrator](../README.md)  -  the full multi-stage LLM pipeline that
  uses `tokio-llm` as its provider layer, adding RAG, deduplication, and
  multi-agent coordination on top.

---

## License

MIT
