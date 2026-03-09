//! # Stage: BudgetEnforcer
//!
//! ## Responsibility
//! Track cumulative USD spend across all API calls and reject requests that
//! would exceed the configured limit. This stage must add <1µs overhead on
//! the hot path.
//!
//! ## Guarantees
//! - Thread-safe: uses `AtomicU64` (bit-cast f64) for lock-free tracking
//! - Monotonic: spend only ever increases; no negative adjustments
//! - Bounded: rejects new charges once `limit_usd` is reached
//! - Non-blocking: `record_usage` never waits on any lock
//!
//! ## NOT Responsible For
//! - Rollback on failed requests (callers should only record confirmed usage)
//! - Cross-process budget state (in-process only)
//! - Alerting or logging (use the `tracing` crate at call sites)

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::LlmError;
use crate::types::Usage;

/// Lock-free USD spend tracker with a hard upper limit.
///
/// Internally represents the running total as an `AtomicU64` containing the
/// bit-pattern of an `f64`, allowing compare-and-swap updates without a mutex.
#[derive(Debug)]
pub struct BudgetEnforcer {
    /// Running total spend in USD, stored as bits of an `f64`.
    spent_bits: AtomicU64,
    /// Maximum allowed spend in USD.
    limit_usd: f64,
}

impl BudgetEnforcer {
    /// Create a new [`BudgetEnforcer`] with the given USD limit.
    ///
    /// # Arguments
    /// * `limit_usd` — maximum total spend in USD before requests are rejected
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::budget::BudgetEnforcer;
    /// let enforcer = BudgetEnforcer::new(5.0);
    /// assert_eq!(enforcer.remaining(), 5.0);
    /// ```
    pub fn new(limit_usd: f64) -> Self {
        Self {
            spent_bits: AtomicU64::new(0),
            limit_usd,
        }
    }

    /// Record the cost of a completed API call.
    ///
    /// Uses a compare-and-swap loop to atomically add `usage.cost_usd` to the
    /// running total. Returns [`LlmError::BudgetExceeded`] if the new total
    /// would exceed the configured limit.
    ///
    /// # Arguments
    /// * `usage` — the usage record from a completed API response
    ///
    /// # Returns
    /// - `Ok(())` — the charge was accepted and recorded
    /// - `Err(LlmError::BudgetExceeded)` — the limit would be exceeded
    ///
    /// # Panics
    /// This function never panics.
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::budget::BudgetEnforcer;
    /// # use tokio_llm::types::Usage;
    /// let enforcer = BudgetEnforcer::new(1.0);
    /// let usage = Usage::new(100, 50, 0.50);
    /// assert!(enforcer.record_usage(&usage).is_ok());
    /// assert!((enforcer.remaining() - 0.50).abs() < 1e-9);
    /// ```
    pub fn record_usage(&self, usage: &Usage) -> Result<(), LlmError> {
        let cost = usage.cost_usd;
        loop {
            let current_bits = self.spent_bits.load(Ordering::Acquire);
            let current = f64::from_bits(current_bits);
            let new_total = current + cost;
            if new_total > self.limit_usd {
                return Err(LlmError::BudgetExceeded {
                    spent: new_total,
                    limit: self.limit_usd,
                });
            }
            let new_bits = new_total.to_bits();
            if self
                .spent_bits
                .compare_exchange(current_bits, new_bits, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
            // CAS failed — another thread updated concurrently; retry
        }
    }

    /// Returns the remaining spend allowance in USD.
    ///
    /// This is a snapshot; the value may decrease concurrently if other tasks
    /// are calling [`record_usage`](Self::record_usage).
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::budget::BudgetEnforcer;
    /// let enforcer = BudgetEnforcer::new(2.0);
    /// assert_eq!(enforcer.remaining(), 2.0);
    /// ```
    pub fn remaining(&self) -> f64 {
        let spent = f64::from_bits(self.spent_bits.load(Ordering::Acquire));
        (self.limit_usd - spent).max(0.0)
    }

    /// Returns the total amount spent so far in USD.
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::budget::BudgetEnforcer;
    /// # use tokio_llm::types::Usage;
    /// let enforcer = BudgetEnforcer::new(10.0);
    /// let _ = enforcer.record_usage(&Usage::new(0, 0, 1.5));
    /// assert!((enforcer.spent() - 1.5).abs() < 1e-9);
    /// ```
    pub fn spent(&self) -> f64 {
        f64::from_bits(self.spent_bits.load(Ordering::Acquire))
    }

    /// Returns the configured spending limit in USD.
    pub fn limit(&self) -> f64 {
        self.limit_usd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_usage(cost: f64) -> Usage {
        Usage::new(100, 50, cost)
    }

    #[test]
    fn test_new_has_zero_spent_and_full_remaining() {
        let e = BudgetEnforcer::new(10.0);
        assert_eq!(e.spent(), 0.0);
        assert_eq!(e.remaining(), 10.0);
        assert_eq!(e.limit(), 10.0);
    }

    #[test]
    fn test_record_usage_within_limit_succeeds() {
        let e = BudgetEnforcer::new(5.0);
        let result = e.record_usage(&make_usage(1.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_usage_accumulates_correctly() {
        let e = BudgetEnforcer::new(10.0);
        e.record_usage(&make_usage(1.0)).unwrap_or(());
        e.record_usage(&make_usage(2.0)).unwrap_or(());
        assert!((e.spent() - 3.0).abs() < 1e-9);
        assert!((e.remaining() - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_usage_exactly_at_limit_succeeds() {
        let e = BudgetEnforcer::new(1.0);
        // Spending exactly the limit should succeed
        let result = e.record_usage(&make_usage(1.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_record_usage_exceeds_limit_returns_budget_exceeded() {
        let e = BudgetEnforcer::new(1.0);
        let result = e.record_usage(&make_usage(1.5));
        match result {
            Err(LlmError::BudgetExceeded { spent, limit }) => {
                assert!((spent - 1.5).abs() < 1e-9);
                assert!((limit - 1.0).abs() < 1e-9);
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
    }

    #[test]
    fn test_record_usage_after_partial_spend_exceeds_limit() {
        let e = BudgetEnforcer::new(1.0);
        e.record_usage(&make_usage(0.6)).unwrap_or(());
        let result = e.record_usage(&make_usage(0.5));
        assert!(matches!(result, Err(LlmError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_remaining_never_goes_negative() {
        // If somehow spent > limit (shouldn't happen, but defensive)
        let e = BudgetEnforcer::new(1.0);
        // Direct write to simulate edge case
        e.spent_bits
            .store(2.0f64.to_bits(), std::sync::atomic::Ordering::Release);
        assert_eq!(e.remaining(), 0.0);
    }

    #[test]
    fn test_spent_reflects_recorded_usage() {
        let e = BudgetEnforcer::new(100.0);
        e.record_usage(&make_usage(3.14)).unwrap_or(());
        assert!((e.spent() - 3.14).abs() < 1e-9);
    }

    #[test]
    fn test_zero_cost_usage_does_not_change_balance() {
        let e = BudgetEnforcer::new(5.0);
        e.record_usage(&make_usage(0.0)).unwrap_or(());
        assert_eq!(e.spent(), 0.0);
        assert_eq!(e.remaining(), 5.0);
    }

    #[test]
    fn test_budget_exceeded_is_budget_error() {
        let e = BudgetEnforcer::new(0.01);
        let err = e.record_usage(&make_usage(1.0)).unwrap_err();
        assert!(err.is_budget_error());
    }

    #[test]
    fn test_multiple_charges_sum_correctly() {
        let e = BudgetEnforcer::new(100.0);
        for _ in 0..10 {
            e.record_usage(&make_usage(1.0)).unwrap_or(());
        }
        assert!((e.spent() - 10.0).abs() < 1e-9);
        assert!((e.remaining() - 90.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_concurrent_charges_total_correctly() {
        use std::sync::Arc;
        let e = Arc::new(BudgetEnforcer::new(1000.0));
        let mut handles = Vec::new();
        for _ in 0..50 {
            let ec = Arc::clone(&e);
            handles.push(tokio::spawn(async move {
                ec.record_usage(&make_usage(1.0))
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        assert!((e.spent() - 50.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_concurrent_charges_stop_at_limit() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering as AO};
        let e = Arc::new(BudgetEnforcer::new(10.0));
        let rejected = Arc::new(AtomicU32::new(0));
        let mut handles = Vec::new();
        for _ in 0..20 {
            let ec = Arc::clone(&e);
            let rc = Arc::clone(&rejected);
            handles.push(tokio::spawn(async move {
                if ec.record_usage(&make_usage(1.0)).is_err() {
                    rc.fetch_add(1, AO::Relaxed);
                }
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        // Exactly 10 should succeed; 10 should be rejected
        assert!((e.spent() - 10.0).abs() < 1e-9);
        assert_eq!(rejected.load(AO::Relaxed), 10);
    }

    #[test]
    fn test_limit_accessor_returns_configured_value() {
        let e = BudgetEnforcer::new(42.0);
        assert!((e.limit() - 42.0).abs() < f64::EPSILON);
    }
}
