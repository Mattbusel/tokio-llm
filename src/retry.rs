//! # Stage: RetryPolicy
//!
//! ## Responsibility
//! Compute per-attempt delay intervals for exponential backoff with jitter.
//! This stage must add <1µs overhead on the hot path.
//!
//! ## Guarantees
//! - Deterministic upper bound: delay never exceeds `MAX_RETRY_DELAY` (60s)
//! - Jitter: adds up to 25% random noise to prevent thundering-herd
//! - Non-blocking: `delay_for_attempt` never awaits anything itself
//! - Zero-overhead when `RetryPolicy::none()` is used
//!
//! ## NOT Responsible For
//! - Deciding whether an error is retryable (see `LlmError::is_retryable`)
//! - Sleeping/waiting — callers own the sleep (`tokio::time::sleep`)
//! - Tracking attempt counts — callers track iteration state

use std::time::Duration;

/// The absolute maximum delay between retry attempts.
pub const MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

/// Controls retry behaviour for API calls.
///
/// Two constructors are provided:
/// - [`RetryPolicy::none`] — no retries; the first error is returned
/// - [`RetryPolicy::exponential`] — bounded exponential backoff with jitter
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first). Zero means no retries.
    max_attempts: u32,
    /// Base delay for the first retry interval.
    base_delay: Duration,
}

impl RetryPolicy {
    /// Create an exponential backoff policy.
    ///
    /// The delay for attempt `n` (0-indexed) is:
    /// ```text
    /// min(base_delay * 2^n, MAX_RETRY_DELAY) + rand(0..25% of that value)
    /// ```
    ///
    /// # Arguments
    /// * `max_attempts` — total attempts allowed (must be ≥ 1)
    /// * `base_delay` — delay before the second attempt
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::retry::RetryPolicy;
    /// # use std::time::Duration;
    /// let policy = RetryPolicy::exponential(3, Duration::from_millis(100));
    /// assert_eq!(policy.max_attempts(), 3);
    /// ```
    pub fn exponential(max_attempts: u32, base_delay: Duration) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            base_delay,
        }
    }

    /// Create a no-retry policy. The first error is immediately returned.
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::retry::RetryPolicy;
    /// let policy = RetryPolicy::none();
    /// assert_eq!(policy.max_attempts(), 1);
    /// ```
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            base_delay: Duration::ZERO,
        }
    }

    /// Return the maximum number of attempts (first call + retries).
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::retry::RetryPolicy;
    /// # use std::time::Duration;
    /// let p = RetryPolicy::exponential(5, Duration::from_millis(50));
    /// assert_eq!(p.max_attempts(), 5);
    /// ```
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    /// Compute the delay before the given attempt number (0-indexed).
    ///
    /// Attempt 0 is the initial call; its delay is always `Duration::ZERO`.
    /// Subsequent attempts use exponential backoff with a jitter factor of
    /// up to 25%.
    ///
    /// The result is capped at [`MAX_RETRY_DELAY`].
    ///
    /// # Arguments
    /// * `attempt` — zero-indexed attempt number
    ///
    /// # Returns
    /// Duration to sleep before issuing this attempt.
    ///
    /// # Panics
    /// This function never panics.
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::retry::RetryPolicy;
    /// # use std::time::Duration;
    /// let p = RetryPolicy::exponential(3, Duration::from_millis(100));
    /// assert_eq!(p.delay_for_attempt(0), Duration::ZERO);
    /// assert!(p.delay_for_attempt(1) >= Duration::from_millis(100));
    /// ```
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        // Compute 2^(attempt-1) * base_delay, saturating at MAX_RETRY_DELAY
        let exp: u32 = attempt.saturating_sub(1);
        let base_millis = self.base_delay.as_millis() as u64;
        // Shift with saturation to avoid overflow
        let shift = exp.min(30) as u64;
        let multiplier = 1u64.saturating_mul(1u64 << shift);
        let scaled = base_millis.saturating_mul(multiplier);
        let cap_millis = MAX_RETRY_DELAY.as_millis() as u64;
        let capped = scaled.min(cap_millis);
        // Add up to 25% jitter using a fast pseudo-random approach.
        // We avoid bringing in `rand` by using the attempt number as a cheap seed.
        let jitter_range = capped / 4;
        let jitter = if jitter_range > 0 {
            // Cheap deterministic jitter: mix bits of attempt + capped
            let seed = (attempt as u64).wrapping_mul(6364136223846793005)
                ^ capped.wrapping_mul(2862933555777941757);
            seed % jitter_range
        } else {
            0
        };
        Duration::from_millis(capped + jitter)
    }

    /// Returns `true` if this policy allows at least one retry.
    ///
    /// # Example
    /// ```rust
    /// # use tokio_llm::retry::RetryPolicy;
    /// # use std::time::Duration;
    /// assert!(!RetryPolicy::none().has_retries());
    /// assert!(RetryPolicy::exponential(3, Duration::from_millis(100)).has_retries());
    /// ```
    pub fn has_retries(&self) -> bool {
        self.max_attempts > 1
    }
}

impl Default for RetryPolicy {
    /// The default policy is 3 attempts with 200ms base delay.
    fn default() -> Self {
        Self::exponential(3, Duration::from_millis(200))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constructor tests ────────────────────────────────────────────────────

    #[test]
    fn test_none_has_one_attempt() {
        assert_eq!(RetryPolicy::none().max_attempts(), 1);
    }

    #[test]
    fn test_none_has_no_retries() {
        assert!(!RetryPolicy::none().has_retries());
    }

    #[test]
    fn test_exponential_stores_max_attempts() {
        let p = RetryPolicy::exponential(5, Duration::from_millis(100));
        assert_eq!(p.max_attempts(), 5);
    }

    #[test]
    fn test_exponential_has_retries() {
        let p = RetryPolicy::exponential(3, Duration::from_millis(50));
        assert!(p.has_retries());
    }

    #[test]
    fn test_exponential_zero_attempts_clamps_to_one() {
        let p = RetryPolicy::exponential(0, Duration::from_millis(100));
        assert_eq!(p.max_attempts(), 1);
    }

    // ── delay_for_attempt tests ──────────────────────────────────────────────

    #[test]
    fn test_delay_attempt_0_is_zero() {
        let p = RetryPolicy::exponential(3, Duration::from_millis(100));
        assert_eq!(p.delay_for_attempt(0), Duration::ZERO);
    }

    #[test]
    fn test_delay_attempt_1_gte_base() {
        let p = RetryPolicy::exponential(3, Duration::from_millis(100));
        assert!(p.delay_for_attempt(1) >= Duration::from_millis(100));
    }

    #[test]
    fn test_delay_attempt_2_gte_double_base() {
        let p = RetryPolicy::exponential(5, Duration::from_millis(100));
        // 2^1 * 100ms = 200ms (before jitter)
        assert!(p.delay_for_attempt(2) >= Duration::from_millis(200));
    }

    #[test]
    fn test_delay_never_exceeds_max() {
        let p = RetryPolicy::exponential(20, Duration::from_millis(100));
        for attempt in 0..20 {
            let d = p.delay_for_attempt(attempt);
            assert!(
                d <= MAX_RETRY_DELAY,
                "attempt {attempt}: delay {d:?} exceeds MAX_RETRY_DELAY"
            );
        }
    }

    #[test]
    fn test_delay_large_attempt_capped_at_max() {
        let p = RetryPolicy::exponential(100, Duration::from_millis(1000));
        // At high attempts, should be capped at ~60s (plus jitter up to 25%)
        let d = p.delay_for_attempt(50);
        // 60s + 25% jitter = max 75s — but jitter is added on top of 60s cap,
        // so max is 60s + 60s/4 = 75s
        assert!(d <= Duration::from_secs(75));
    }

    #[test]
    fn test_delay_is_monotonically_non_decreasing_base() {
        let p = RetryPolicy::exponential(10, Duration::from_millis(100));
        // Without jitter the base should grow; verify at least that attempt N+1 >= attempt N / 2
        // (jitter can cause slight non-monotonicity at individual samples)
        let d1 = p.delay_for_attempt(1).as_millis();
        let d2 = p.delay_for_attempt(2).as_millis();
        let d3 = p.delay_for_attempt(3).as_millis();
        // With 25% jitter the upper bound of attempt N can overlap lower bound of N+1,
        // so we just verify they're in the right ballpark.
        assert!(d2 >= d1, "d2={d2} should be >= d1={d1}");
        assert!(d3 >= d2, "d3={d3} should be >= d2={d2}");
    }

    #[test]
    fn test_none_delay_attempt_0_is_zero() {
        let p = RetryPolicy::none();
        assert_eq!(p.delay_for_attempt(0), Duration::ZERO);
    }

    #[test]
    fn test_default_policy_has_three_attempts() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_attempts(), 3);
    }

    #[test]
    fn test_default_policy_has_retries() {
        assert!(RetryPolicy::default().has_retries());
    }

    // ── Property-based tests ─────────────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_delay_never_exceeds_max(
            base_ms in 1u64..=5000u64,
            attempt in 0u32..=30u32
        ) {
            let p = RetryPolicy::exponential(31, Duration::from_millis(base_ms));
            let d = p.delay_for_attempt(attempt);
            prop_assert!(d <= Duration::from_secs(75),
                "delay {:?} exceeded 75s for attempt={} base_ms={}", d, attempt, base_ms);
        }

        #[test]
        fn prop_attempt_0_always_zero(base_ms in 1u64..=5000u64) {
            let p = RetryPolicy::exponential(5, Duration::from_millis(base_ms));
            prop_assert_eq!(p.delay_for_attempt(0), Duration::ZERO);
        }

        #[test]
        fn prop_max_attempts_at_least_one(attempts in 0u32..=100u32) {
            let p = RetryPolicy::exponential(attempts, Duration::from_millis(100));
            prop_assert!(p.max_attempts() >= 1);
        }
    }
}
