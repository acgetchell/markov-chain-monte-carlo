//! Observable measurements and sample buffers.

use std::error::Error;
use std::fmt;
use std::slice;
use std::vec;

/// Measurement computed from the current chain state.
///
/// Observables let sampling code record derived quantities such as energies,
/// order parameters, or correlation functions without storing full state
/// histories.  The observable is mutable so implementations can keep internal
/// scratch space, counters, or online statistics state.
pub trait Observable<S> {
    /// Measurement value produced by this observable.
    type Output;

    /// Compute a measurement from `state`.
    ///
    /// ```
    /// use markov_chain_monte_carlo::Observable;
    ///
    /// let mut squared = |x: &f64| x * x;
    ///
    /// assert_eq!(Observable::observe(&mut squared, &3.0), 9.0);
    /// ```
    fn observe(&mut self, state: &S) -> Self::Output;
}

impl<S, O, F: FnMut(&S) -> O> Observable<S> for F {
    type Output = O;

    fn observe(&mut self, state: &S) -> Self::Output {
        self(state)
    }
}

/// Fallible measurement computed from the current chain state.
///
/// Use this when measurement itself can fail independently of the
/// Metropolis-Hastings step, for example when a domain-specific observable
/// validates invariants or delegates to a fallible analysis routine.
pub trait TryObservable<S> {
    /// Measurement value produced by this observable.
    type Output;
    /// Measurement-specific error type.
    type Error;

    /// Compute a fallible measurement from `state`.
    ///
    /// ```
    /// use markov_chain_monte_carlo::TryObservable;
    ///
    /// let mut reciprocal = |x: &f64| {
    ///     if *x == 0.0 { Err("zero") } else { Ok(1.0 / *x) }
    /// };
    ///
    /// assert_eq!(TryObservable::try_observe(&mut reciprocal, &2.0)?, 0.5);
    /// # Ok::<(), &'static str>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when the observable cannot compute a valid
    /// measurement for `state`.
    fn try_observe(&mut self, state: &S) -> Result<Self::Output, Self::Error>;
}

impl<S, O, E, F: FnMut(&S) -> Result<O, E>> TryObservable<S> for F {
    type Output = O;
    type Error = E;

    fn try_observe(&mut self, state: &S) -> Result<Self::Output, Self::Error> {
        self(state)
    }
}

/// Error from a sampling step paired with a fallible observation.
///
/// The two variants keep transition failures and measurement failures
/// orthogonal.  A [`Step`](Self::Step) error means the observable was not
/// invoked.  An [`Observation`](Self::Observation) error means the sampling
/// step already completed successfully, then the measurement failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObservedStepError<S, O> {
    /// The sampling step failed before observation.
    Step(S),
    /// The sampling step succeeded, but observation failed.
    Observation(O),
}

impl<S: fmt::Display, O: fmt::Display> fmt::Display for ObservedStepError<S, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Step(err) => write!(f, "sampling step failed before observation: {err}"),
            Self::Observation(err) => {
                write!(
                    f,
                    "observable measurement failed after a successful sampling step: {err}"
                )
            }
        }
    }
}

impl<S: Error + 'static, O: Error + 'static> Error for ObservedStepError<S, O> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Step(err) => Some(err),
            Self::Observation(err) => Some(err),
        }
    }
}

/// In-memory collection of observation outputs.
///
/// `SampleBuffer` is intentionally a thin wrapper around `Vec<T>` so callers
/// can start with a simple collection and later replace it with an online
/// accumulator when they do not need to retain every measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct SampleBuffer<T> {
    samples: Vec<T>,
}

impl<T> SampleBuffer<T> {
    /// Create an empty buffer.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let buffer = SampleBuffer::<f64>::new();
    /// assert!(buffer.is_empty());
    /// ```
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Create an empty buffer with space for at least `capacity` samples.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let buffer = SampleBuffer::<usize>::with_capacity(128);
    /// assert_eq!(buffer.len(), 0);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
        }
    }

    /// Append one observation.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let mut buffer = SampleBuffer::new();
    /// buffer.push(1.5);
    /// assert_eq!(buffer.as_slice(), &[1.5]);
    /// ```
    pub fn push(&mut self, sample: T) {
        self.samples.push(sample);
    }

    /// Number of observations in the buffer.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let buffer: SampleBuffer<_> = [1, 2, 3].into_iter().collect();
    /// assert_eq!(buffer.len(), 3);
    /// ```
    #[must_use]
    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer contains no observations.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let mut buffer = SampleBuffer::new();
    /// assert!(buffer.is_empty());
    /// buffer.push("measurement");
    /// assert!(!buffer.is_empty());
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Remove all observations while retaining allocated capacity.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let mut buffer: SampleBuffer<_> = [1, 2].into_iter().collect();
    /// buffer.clear();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Borrow observations as a slice.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let buffer: SampleBuffer<_> = [0.25, 0.5].into_iter().collect();
    /// assert_eq!(buffer.as_slice(), &[0.25, 0.5]);
    /// ```
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        self.samples.as_slice()
    }

    /// Iterate over observations.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let buffer: SampleBuffer<_> = [1, 2, 3].into_iter().collect();
    /// let total: i32 = buffer.iter().sum();
    /// assert_eq!(total, 6);
    /// ```
    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.samples.iter()
    }

    /// Consume the buffer and return the underlying vector.
    ///
    /// ```
    /// use markov_chain_monte_carlo::SampleBuffer;
    ///
    /// let buffer: SampleBuffer<_> = [1, 2, 3].into_iter().collect();
    /// assert_eq!(buffer.into_vec(), vec![1, 2, 3]);
    /// ```
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.samples
    }
}

impl<T> Default for SampleBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Extend<T> for SampleBuffer<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.samples.extend(iter);
    }
}

impl<T> FromIterator<T> for SampleBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            samples: Vec::from_iter(iter),
        }
    }
}

impl<T> IntoIterator for SampleBuffer<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.samples.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a SampleBuffer<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MeasurementError {
        Domain,
    }

    impl fmt::Display for MeasurementError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::Domain => write!(f, "domain-specific measurement failed"),
            }
        }
    }

    impl Error for MeasurementError {}

    #[test]
    fn closure_observable_measures_state() {
        let mut squared = |state: &f64| state * state;

        assert!((squared.observe(&3.0) - 9.0).abs() < f64::EPSILON);
    }

    #[test]
    fn try_closure_reports_error() {
        let mut positive_only = |state: &f64| {
            if *state > 0.0 {
                Ok(*state)
            } else {
                Err(MeasurementError::Domain)
            }
        };

        assert_eq!(positive_only.try_observe(&2.0), Ok(2.0));
        assert_eq!(
            positive_only.try_observe(&0.0),
            Err(MeasurementError::Domain)
        );
    }

    #[test]
    fn observed_error_messages() {
        type Error = ObservedStepError<MeasurementError, MeasurementError>;

        let step = Error::Step(MeasurementError::Domain);
        let observation = Error::Observation(MeasurementError::Domain);

        assert_eq!(
            step.to_string(),
            "sampling step failed before observation: domain-specific measurement failed"
        );
        assert_eq!(
            observation.to_string(),
            "observable measurement failed after a successful sampling step: domain-specific measurement failed"
        );
    }

    #[test]
    fn observed_error_sources() {
        type Error = ObservedStepError<MeasurementError, MeasurementError>;

        let step = Error::Step(MeasurementError::Domain);
        let observation = Error::Observation(MeasurementError::Domain);

        assert_eq!(
            step.source().map(ToString::to_string),
            Some("domain-specific measurement failed".to_owned())
        );
        assert_eq!(
            observation.source().map(ToString::to_string),
            Some("domain-specific measurement failed".to_owned())
        );
    }

    #[test]
    fn sample_buffer_collects_outputs() {
        let mut buffer = SampleBuffer::with_capacity(2);
        buffer.push(1);
        buffer.extend([2, 3]);

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.as_slice(), &[1, 2, 3]);
        assert_eq!(buffer.into_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn sample_buffer_adapters() {
        let mut buffer = SampleBuffer::new();
        assert!(buffer.is_empty());

        buffer.extend([1, 2, 3]);
        assert!(!buffer.is_empty());

        let borrowed: Vec<_> = (&buffer).into_iter().copied().collect();
        assert_eq!(borrowed, vec![1, 2, 3]);

        buffer.clear();
        assert!(buffer.is_empty());

        let defaulted = SampleBuffer::<i32>::default();
        assert!(defaulted.is_empty());

        let collected: SampleBuffer<_> = [4, 5].into_iter().collect();
        let owned: Vec<_> = collected.into_iter().collect();
        assert_eq!(owned, vec![4, 5]);
    }
}
