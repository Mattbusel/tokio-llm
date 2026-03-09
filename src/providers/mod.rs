//! LLM provider implementations.
//!
//! Each provider module exposes a struct that implements the HTTP calls for
//! a specific API. The [`Provider`] trait is the common interface consumed by
//! [`crate::client::LlmClient`].

pub mod anthropic;
pub mod openai;

use std::pin::Pin;

use futures::Stream;

use crate::error::LlmError;
use crate::types::{ChatRequest, ChatResponse, StreamChunk};

/// The common interface that all provider backends must implement.
///
/// Implementors are responsible for:
/// - Translating [`ChatRequest`] into provider-specific JSON
/// - Executing the HTTP call
/// - Translating the response (or stream) back into provider-agnostic types
/// - Computing `cost_usd` from the model's pricing table
///
/// # Panics
/// No implementation of this trait should ever panic.
#[async_trait::async_trait]
pub trait Provider: std::fmt::Debug {
    /// Execute a blocking (non-streaming) chat completion.
    async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse, LlmError>;

    /// Execute a streaming chat completion.
    ///
    /// Returns a `Stream` of incremental [`StreamChunk`] values.
    async fn chat_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>, LlmError>;
}
