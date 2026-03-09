//! # tokio-llm
//!
//! A Tokio-native async LLM client for OpenAI and Anthropic with built-in
//! retry, circuit breaking, streaming, budget enforcement, and tracing.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use tokio_llm::client::LlmClient;
//! use tokio_llm::types::{ChatRequest, Message, Model};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), tokio_llm::error::LlmError> {
//!     // OpenAI
//!     let client = LlmClient::openai("sk-...")
//!         .with_budget(5.0)
//!         .build()?;
//!
//!     let req = ChatRequest::new(
//!         Model::Gpt4oMini,
//!         vec![Message::user("What is the capital of France?")],
//!     );
//!     let resp = client.chat(req).await?;
//!     println!("{}", resp.content);
//!
//!     // Anthropic
//!     let client = LlmClient::anthropic("sk-ant-...")
//!         .with_circuit_breaker(5, Duration::from_secs(30))
//!         .build()?;
//!
//!     let req = ChatRequest::new(
//!         Model::Claude35Haiku,
//!         vec![
//!             Message::system("You are a concise assistant."),
//!             Message::user("Hello!"),
//!         ],
//!     );
//!     let resp = client.chat(req).await?;
//!     println!("{}", resp.content);
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | **Dual provider** | OpenAI and Anthropic via a unified `LlmClient` |
//! | **Retry** | Exponential backoff with jitter; configurable attempt count |
//! | **Circuit breaker** | Open/HalfOpen/Closed state machine; prevents cascade failures |
//! | **Budget enforcement** | Lock-free atomic spend tracker; hard USD limits |
//! | **Streaming** | SSE-native `Stream<Item = StreamChunk>` for both providers |
//! | **Tracing** | `tracing` instrumentation on every request |
//! | **Zero panics** | No `unwrap`/`expect`/`panic!` in production code |
//! | **Typed errors** | Every failure mode is a named `LlmError` variant |
//!
//! ## Architecture
//!
//! ```text
//! LlmClient
//!    BudgetEnforcer   (pre-call spend guard)
//!    RetryPolicy      (exponential backoff scheduler)
//!    CircuitBreaker   (Open/HalfOpen/Closed state machine)
//!    Provider trait
//!          OpenAiProvider    (Chat Completions API + SSE)
//!          AnthropicProvider (Messages API + SSE)
//! ```

pub mod budget;
pub mod circuit_breaker;
pub mod client;
pub mod error;
pub mod providers;
pub mod retry;
pub mod types;

//  Re-exports 

pub use budget::BudgetEnforcer;
pub use circuit_breaker::{CircuitBreaker, CircuitState};
pub use client::{ClientBuilder, LlmClient};
pub use error::LlmError;
pub use providers::anthropic::AnthropicProvider;
pub use providers::openai::OpenAiProvider;
pub use retry::RetryPolicy;
pub use types::{
    ChatRequest, ChatResponse, Message, Model, Role, StreamChunk, Usage,
};
