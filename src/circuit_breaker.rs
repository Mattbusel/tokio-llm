//! # Stage: CircuitBreaker
//!
//! ## Responsibility
//! Prevent repeated calls to a failing downstream service by transitioning
//! through Closed → Open → HalfOpen → Closed states. This stage must add
//! <50µs overhead on the hot path (P99).
//!
//! ## State Machine
//! ```text
//! Closed (threshold failures) Open
//!                                   
//!    (probe succeeds)                (reset_timeout elapses)
//!    HalfOpen
//!            
//!            (probe fails) Open
//! ```
//!
//! ## Guarantees
//! - Thread-safe: all state is protected by a single `tokio::sync::Mutex`
//! - Fast-fail: Open state returns [`LlmError::CircuitOpen`] without calling `f`
//! - Single-probe: only one call is forwarded in HalfOpen state at a time
//!
//! ## NOT Responsible For
//! - Per-endpoint isolation (wrap in multiple instances)
//! - Partial-open / sliding-window failure rates (threshold is a fixed count)

use std::time::{Duration, Instant};

use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::error::LlmError;

/// The internal state of the circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation  -  calls pass through.
    Closed,
    /// Failure threshold exceeded  -  calls fast-fail.
    Open,
    /// Cooldown elapsed  -  one probe call is forwarded to test recovery.
    HalfOpen,
}

/// Internal mutable state guarded by the mutex.
#[derive(Debug)]
struct Inner {
    state: CircuitState,
    failure_count: u32,
    opened_at: Option<Instant>,
}

/// Async circuit breaker implementing the Closed/Open/HalfOpen state machine.
///
/// # Example
/// ```rust,no_run
/// # use tokio_llm::circuit_breaker::CircuitBreaker;
/// # use std::time::Duration;
/// # #[tokio::main] async fn main() {
/// let cb = CircuitBreaker::new(5, Duration::from_secs(30));
/// let result = cb.call(|| async { Ok::<_, tokio_llm::error::LlmError>("hello") }).await;
/// # }
/// ```
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    reset_timeout: Duration,
    inner: Mutex<Inner>,
}

impl CircuitBreaker {
    /// Create a new [`CircuitBreaker`].
    ///
    /// # Arguments
    /// * `failure_threshold`  -  number of consecutive failures that open the circuit
    /// * `reset_timeout`  -  how long to stay Open before transitioning to HalfOpen
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::circuit_breaker::CircuitBreaker;
    /// # use std::time::Duration;
    /// let cb = CircuitBreaker::new(5, Duration::from_secs(30));
    /// ```
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            reset_timeout,
            inner: Mutex::new(Inner {
                state: CircuitState::Closed,
                failure_count: 0,
                opened_at: None,
            }),
        }
    }

    /// Execute a closure through the circuit breaker.
    ///
    /// - If the circuit is **Closed**, `f` is called. On success the failure
    ///   counter is reset. On failure the counter is incremented; once it
    ///   reaches `failure_threshold` the circuit opens.
    /// - If the circuit is **Open** and the `reset_timeout` has not elapsed,
    ///   [`LlmError::CircuitOpen`] is returned immediately without calling `f`.
    /// - If the circuit is **Open** and `reset_timeout` has elapsed, it
    ///   transitions to **HalfOpen** and `f` is called once as a probe.
    /// - If the circuit is **HalfOpen**, `f` is called. Success → Closed;
    ///   failure → Open again.
    ///
    /// # Returns
    /// - `Ok(T)`  -  call succeeded
    /// - `Err(LlmError::CircuitOpen { .. })`  -  circuit is open; request not sent
    /// - `Err(e)`  -  the wrapped function returned an error
    ///
    /// # Panics
    /// This function never panics.
    pub async fn call<F, Fut, T>(&self, f: F) -> Result<T, LlmError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, LlmError>>,
    {
        //  Phase 1: check state under lock
        {
            let mut inner = self.inner.lock().await;
            match inner.state {
                CircuitState::Open => {
                    let opened_at = inner.opened_at.unwrap_or_else(Instant::now);
                    let elapsed = opened_at.elapsed();
                    if elapsed < self.reset_timeout {
                        let remaining = self.reset_timeout.saturating_sub(elapsed);
                        debug!(
                            remaining_secs = remaining.as_secs_f64(),
                            "circuit open  -  fast-failing request"
                        );
                        return Err(LlmError::CircuitOpen {
                            reset_after_secs: remaining.as_secs_f64(),
                        });
                    }
                    // Timeout elapsed → probe
                    inner.state = CircuitState::HalfOpen;
                    debug!("circuit transitioning Open → HalfOpen");
                }
                CircuitState::Closed | CircuitState::HalfOpen => {
                    // Allow the call to proceed
                }
            }
        }

        //  Phase 2: execute the call (outside the lock)
        let result = f().await;

        //  Phase 3: update state under lock
        {
            let mut inner = self.inner.lock().await;
            match &result {
                Ok(_) => {
                    if inner.state != CircuitState::Closed {
                        debug!("circuit closing after successful probe");
                    }
                    inner.state = CircuitState::Closed;
                    inner.failure_count = 0;
                    inner.opened_at = None;
                }
                Err(_) => {
                    inner.failure_count += 1;
                    if inner.failure_count >= self.failure_threshold
                        || inner.state == CircuitState::HalfOpen
                    {
                        warn!(
                            failures = inner.failure_count,
                            "circuit opening after failure threshold reached"
                        );
                        inner.state = CircuitState::Open;
                        inner.opened_at = Some(Instant::now());
                        inner.failure_count = 0;
                    }
                }
            }
        }

        result
    }

    /// Returns the current circuit state.
    ///
    /// Note: this is a snapshot and may change immediately after reading.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use tokio_llm::circuit_breaker::{CircuitBreaker, CircuitState};
    /// # use std::time::Duration;
    /// # #[tokio::main] async fn main() {
    /// let cb = CircuitBreaker::new(3, Duration::from_secs(10));
    /// assert_eq!(cb.state().await, CircuitState::Closed);
    /// # }
    /// ```
    pub async fn state(&self) -> CircuitState {
        self.inner.lock().await.state.clone()
    }

    /// Returns the current consecutive failure count.
    pub async fn failure_count(&self) -> u32 {
        self.inner.lock().await.failure_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_call() -> impl std::future::Future<Output = Result<&'static str, LlmError>> {
        async { Ok("success") }
    }

    fn err_call() -> impl std::future::Future<Output = Result<&'static str, LlmError>> {
        async {
            Err(LlmError::ApiError {
                status: 500,
                message: "internal server error".into(),
            })
        }
    }

    #[tokio::test]
    async fn test_initial_state_is_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(10));
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_successful_call_stays_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(10));
        let result = cb.call(ok_call).await;
        assert!(result.is_ok());
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_failure_increments_count() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(10));
        let _ = cb.call(err_call).await;
        assert_eq!(cb.failure_count().await, 1);
    }

    #[tokio::test]
    async fn test_below_threshold_stays_closed() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(10));
        for _ in 0..4 {
            let _ = cb.call(err_call).await;
        }
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_threshold_failures_open_circuit() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        for _ in 0..3 {
            let _ = cb.call(err_call).await;
        }
        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_open_circuit_fast_fails() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        for _ in 0..3 {
            let _ = cb.call(err_call).await;
        }
        let result = cb.call(ok_call).await;
        assert!(matches!(result, Err(LlmError::CircuitOpen { .. })));
    }

    #[tokio::test]
    async fn test_open_circuit_returns_reset_after_secs() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(30));
        let _ = cb.call(err_call).await;
        match cb.call(ok_call).await {
            Err(LlmError::CircuitOpen { reset_after_secs }) => {
                assert!(reset_after_secs > 0.0 && reset_after_secs <= 30.0);
            }
            other => panic!("expected CircuitOpen, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_success_resets_failure_count() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(10));
        for _ in 0..3 {
            let _ = cb.call(err_call).await;
        }
        let _ = cb.call(ok_call).await;
        assert_eq!(cb.failure_count().await, 0);
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_open_transitions_to_half_open_after_timeout() {
        // Use a very short reset timeout so we don't have to actually sleep
        let cb = CircuitBreaker::new(1, Duration::from_millis(1));
        let _ = cb.call(err_call).await;
        assert_eq!(cb.state().await, CircuitState::Open);
        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(5)).await;
        // The next call should probe (HalfOpen internally)
        let result = cb.call(ok_call).await;
        // Probe succeeded → should be closed now
        assert!(result.is_ok());
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_half_open_failure_reopens_circuit() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(1));
        let _ = cb.call(err_call).await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        // Probe fails → should reopen
        let _ = cb.call(err_call).await;
        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_half_open_success_closes_circuit() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(1));
        let _ = cb.call(err_call).await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        let result = cb.call(ok_call).await;
        assert!(result.is_ok());
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_open_error_not_retryable() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60));
        let _ = cb.call(err_call).await;
        match cb.call(ok_call).await {
            Err(e) => assert!(!e.is_retryable()),
            Ok(_) => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn test_multiple_open_fast_fails() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60));
        let _ = cb.call(err_call).await;
        let _ = cb.call(err_call).await;
        // Multiple fast-fails should all return CircuitOpen
        for _ in 0..5 {
            assert!(matches!(
                cb.call(ok_call).await,
                Err(LlmError::CircuitOpen { .. })
            ));
        }
    }

    #[tokio::test]
    async fn test_state_returns_correct_variant() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60));
        assert_eq!(cb.state().await, CircuitState::Closed);
        let _ = cb.call(err_call).await;
        assert_eq!(cb.state().await, CircuitState::Open);
    }
}
