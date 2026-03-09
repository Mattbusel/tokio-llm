//! Typed error hierarchy for tokio-llm.
//!
//! Every failure mode is represented as a named variant so callers can
//! match exhaustively and handle each case explicitly.

use thiserror::Error;

/// The unified error type returned by all tokio-llm operations.
///
/// # Variants
/// - [`LlmError::ApiError`]  -  the upstream API returned a non-2xx status
/// - [`LlmError::RateLimited`]  -  HTTP 429 with optional retry-after seconds
/// - [`LlmError::BudgetExceeded`]  -  spend limit has been reached
/// - [`LlmError::CircuitOpen`]  -  circuit breaker is open; request fast-failed
/// - [`LlmError::StreamError`]  -  error parsing or reading the SSE stream
/// - [`LlmError::InvalidConfig`]  -  builder was configured incorrectly
/// - [`LlmError::RequestTimeout`]  -  request exceeded the configured deadline
/// - [`LlmError::Transport`]  -  underlying HTTP transport error
/// - [`LlmError::Serialization`]  -  JSON encode/decode failure
#[derive(Debug, Error)]
pub enum LlmError {
    /// The upstream API returned a non-2xx HTTP status code.
    #[error("API error {status}: {message}")]
    ApiError {
        /// HTTP status code returned by the provider.
        status: u16,
        /// Error message extracted from the response body.
        message: String,
    },

    /// The provider returned HTTP 429 Too Many Requests.
    #[error("rate limited by provider (retry after {retry_after_secs:?}s)")]
    RateLimited {
        /// Number of seconds to wait before retrying, if provided by the API.
        retry_after_secs: Option<u64>,
    },

    /// Total spend has reached or exceeded the configured budget limit.
    #[error("budget exceeded: spent ${spent:.4} of ${limit:.4} limit")]
    BudgetExceeded {
        /// Total USD spent so far.
        spent: f64,
        /// Configured spending limit in USD.
        limit: f64,
    },

    /// The circuit breaker is in the Open state; the request was not sent.
    #[error("circuit breaker open  -  service is unavailable, reset in {reset_after_secs:.1}s")]
    CircuitOpen {
        /// Approximate seconds until the circuit transitions to HalfOpen.
        reset_after_secs: f64,
    },

    /// An error occurred while parsing or reading the SSE stream.
    #[error("stream error: {message}")]
    StreamError {
        /// Human-readable description of what went wrong.
        message: String,
    },

    /// The client or builder was configured with an invalid value.
    #[error("invalid configuration: {message}")]
    InvalidConfig {
        /// Description of the configuration problem.
        message: String,
    },

    /// The request exceeded the configured timeout deadline.
    #[error("request timed out after {timeout_ms}ms")]
    RequestTimeout {
        /// The timeout that was exceeded, in milliseconds.
        timeout_ms: u64,
    },

    /// An HTTP transport-level error (connection refused, DNS failure, etc.).
    #[error("transport error: {0}")]
    Transport(#[from] reqwest::Error),

    /// A JSON serialization or deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl LlmError {
    /// Returns `true` if this error is considered transient and safe to retry.
    ///
    /// Transient errors: [`LlmError::RateLimited`], [`LlmError::RequestTimeout`],
    /// and [`LlmError::ApiError`] with status 500/502/503/504.
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::error::LlmError;
    /// let err = LlmError::RateLimited { retry_after_secs: Some(5) };
    /// assert!(err.is_retryable());
    /// ```
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::RateLimited { .. } => true,
            LlmError::RequestTimeout { .. } => true,
            LlmError::ApiError { status, .. } => {
                matches!(status, 500 | 502 | 503 | 504)
            }
            _ => false,
        }
    }

    /// Returns `true` if this error represents a budget constraint violation.
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::error::LlmError;
    /// let err = LlmError::BudgetExceeded { spent: 1.0, limit: 0.5 };
    /// assert!(err.is_budget_error());
    /// ```
    pub fn is_budget_error(&self) -> bool {
        matches!(self, LlmError::BudgetExceeded { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_error_display_includes_status_and_message() {
        let err = LlmError::ApiError {
            status: 400,
            message: "bad request".into(),
        };
        let s = err.to_string();
        assert!(s.contains("400"));
        assert!(s.contains("bad request"));
    }

    #[test]
    fn test_rate_limited_with_retry_after_display() {
        let err = LlmError::RateLimited {
            retry_after_secs: Some(30),
        };
        assert!(err.to_string().contains("30"));
    }

    #[test]
    fn test_rate_limited_without_retry_after_display() {
        let err = LlmError::RateLimited {
            retry_after_secs: None,
        };
        assert!(err.to_string().contains("rate limited"));
    }

    #[test]
    fn test_budget_exceeded_display_includes_amounts() {
        let err = LlmError::BudgetExceeded {
            spent: 1.2345,
            limit: 1.0,
        };
        let s = err.to_string();
        assert!(s.contains("1.2345"));
        assert!(s.contains("1.0000"));
    }

    #[test]
    fn test_circuit_open_display_includes_seconds() {
        let err = LlmError::CircuitOpen {
            reset_after_secs: 42.5,
        };
        assert!(err.to_string().contains("42.5"));
    }

    #[test]
    fn test_stream_error_display() {
        let err = LlmError::StreamError {
            message: "unexpected EOF".into(),
        };
        assert!(err.to_string().contains("unexpected EOF"));
    }

    #[test]
    fn test_invalid_config_display() {
        let err = LlmError::InvalidConfig {
            message: "max_attempts must be > 0".into(),
        };
        assert!(err.to_string().contains("max_attempts must be > 0"));
    }

    #[test]
    fn test_request_timeout_display_includes_ms() {
        let err = LlmError::RequestTimeout { timeout_ms: 5000 };
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_is_retryable_rate_limited_true() {
        let err = LlmError::RateLimited {
            retry_after_secs: None,
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_timeout_true() {
        let err = LlmError::RequestTimeout { timeout_ms: 1000 };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_500_true() {
        let err = LlmError::ApiError {
            status: 500,
            message: "internal server error".into(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_502_true() {
        let err = LlmError::ApiError {
            status: 502,
            message: "bad gateway".into(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_503_true() {
        let err = LlmError::ApiError {
            status: 503,
            message: "service unavailable".into(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_504_true() {
        let err = LlmError::ApiError {
            status: 504,
            message: "gateway timeout".into(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_is_retryable_400_false() {
        let err = LlmError::ApiError {
            status: 400,
            message: "bad request".into(),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_is_retryable_circuit_open_false() {
        let err = LlmError::CircuitOpen {
            reset_after_secs: 10.0,
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_is_retryable_budget_exceeded_false() {
        let err = LlmError::BudgetExceeded {
            spent: 2.0,
            limit: 1.0,
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_is_budget_error_true() {
        let err = LlmError::BudgetExceeded {
            spent: 1.0,
            limit: 0.5,
        };
        assert!(err.is_budget_error());
    }

    #[test]
    fn test_is_budget_error_false_for_api_error() {
        let err = LlmError::ApiError {
            status: 400,
            message: "bad".into(),
        };
        assert!(!err.is_budget_error());
    }

    #[test]
    fn test_serialization_error_from_serde() {
        let raw = "not valid json {{{";
        let serde_err = serde_json::from_str::<serde_json::Value>(raw).unwrap_err();
        let err = LlmError::from(serde_err);
        assert!(matches!(err, LlmError::Serialization(_)));
    }

    #[test]
    fn test_all_variants_implement_debug() {
        let variants: Vec<Box<dyn std::fmt::Debug>> = vec![
            Box::new(LlmError::ApiError {
                status: 400,
                message: "x".into(),
            }),
            Box::new(LlmError::RateLimited {
                retry_after_secs: None,
            }),
            Box::new(LlmError::BudgetExceeded {
                spent: 0.0,
                limit: 1.0,
            }),
            Box::new(LlmError::CircuitOpen {
                reset_after_secs: 0.0,
            }),
            Box::new(LlmError::StreamError {
                message: "x".into(),
            }),
            Box::new(LlmError::InvalidConfig {
                message: "x".into(),
            }),
            Box::new(LlmError::RequestTimeout { timeout_ms: 0 }),
        ];
        for v in &variants {
            assert!(!format!("{v:?}").is_empty());
        }
    }
}
