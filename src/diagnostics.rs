//! Trace recording and export helpers for MCMC diagnostics.
//!
//! This module stores numeric observable traces independently from plotting,
//! notebook rendering, or downstream statistical estimators.  A [`Trace`]
//! contains one row per completed step, a stable [`ChainId`], accept/reject
//! metadata through [`TraceStepOutcome`], the chain's cached target
//! log-probability, and caller-defined numeric observable columns.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::io::{self, Write};

use crate::{Chain, Step, StepOutcome};

/// Stable identifier for one recorded Markov chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct ChainId(usize);

impl ChainId {
    /// Create a chain identifier from a caller-owned index.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::ChainId;
    ///
    /// let id = ChainId::new(2);
    /// assert_eq!(id.get(), 2);
    /// ```
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the raw chain identifier.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::ChainId;
    ///
    /// assert_eq!(ChainId::new(7).get(), 7);
    /// ```
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for ChainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Acceptance/proposal outcome recorded for one trace row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct TraceStepOutcome {
    accepted: bool,
    proposed: bool,
}

impl TraceStepOutcome {
    /// A concrete proposal was accepted.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::TraceStepOutcome;
    ///
    /// let outcome = TraceStepOutcome::accepted();
    /// assert!(outcome.is_accepted());
    /// assert!(outcome.had_proposal());
    /// ```
    pub const fn accepted() -> Self {
        Self {
            accepted: true,
            proposed: true,
        }
    }

    /// A concrete proposal was rejected by the Metropolis-Hastings draw.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::TraceStepOutcome;
    ///
    /// let outcome = TraceStepOutcome::rejected_proposal();
    /// assert!(!outcome.is_accepted());
    /// assert!(outcome.had_proposal());
    /// ```
    pub const fn rejected_proposal() -> Self {
        Self {
            accepted: false,
            proposed: true,
        }
    }

    /// No concrete proposal was available, so the step was a self-loop.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::TraceStepOutcome;
    ///
    /// let outcome = TraceStepOutcome::no_proposal();
    /// assert!(!outcome.is_accepted());
    /// assert!(!outcome.had_proposal());
    /// ```
    pub const fn no_proposal() -> Self {
        Self {
            accepted: false,
            proposed: false,
        }
    }

    /// Convert a concrete proposal's boolean acceptance result.
    ///
    /// This is the natural adapter for [`crate::Chain::step_mut`] and
    /// [`crate::Sampler::step_mut`] when the caller knows the in-place proposal
    /// produced a concrete move.  A `false` value is recorded as
    /// [`Self::rejected_proposal`], not [`Self::no_proposal`].
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::TraceStepOutcome;
    ///
    /// let accepted = TraceStepOutcome::from_proposal_acceptance(true);
    /// let rejected = TraceStepOutcome::from_proposal_acceptance(false);
    ///
    /// assert!(accepted.is_accepted());
    /// assert!(accepted.had_proposal());
    /// assert!(!rejected.is_accepted());
    /// assert!(rejected.had_proposal());
    /// ```
    pub const fn from_proposal_acceptance(accepted: bool) -> Self {
        if accepted {
            Self::accepted()
        } else {
            Self::rejected_proposal()
        }
    }

    /// Whether the step accepted and committed a concrete proposal.
    #[must_use]
    pub const fn is_accepted(self) -> bool {
        self.accepted
    }

    /// Whether the step included a concrete proposal.
    #[must_use]
    pub const fn had_proposal(self) -> bool {
        self.proposed
    }
}

impl From<StepOutcome> for TraceStepOutcome {
    fn from(outcome: StepOutcome) -> Self {
        match outcome {
            StepOutcome::Accepted => Self::accepted(),
            StepOutcome::RejectedProposal => Self::rejected_proposal(),
            StepOutcome::NoProposal => Self::no_proposal(),
        }
    }
}

impl<I> From<&Step<I>> for TraceStepOutcome {
    fn from(step: &Step<I>) -> Self {
        step.outcome.into()
    }
}

/// One recorded post-step trace row.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct TraceRecord {
    chain_id: ChainId,
    step: usize,
    outcome: TraceStepOutcome,
    log_prob: f64,
    observable_values: Vec<f64>,
}

impl TraceRecord {
    /// Create one trace record from already-computed observable values.
    ///
    /// Most callers should use [`TraceRecorder::record`] so the step number and
    /// log-probability come from a live [`Chain`].  Use this constructor when
    /// assembling rows from an external source or merging trace data manually.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{ChainId, TraceRecord, TraceStepOutcome};
    ///
    /// let record = TraceRecord::new(
    ///     ChainId::new(0),
    ///     10,
    ///     TraceStepOutcome::accepted(),
    ///     -1.25,
    ///     vec![3.0, 0.5],
    /// );
    ///
    /// assert_eq!(record.step(), 10);
    /// assert_eq!(record.observable_values(), &[3.0, 0.5]);
    /// ```
    pub const fn new(
        chain_id: ChainId,
        step: usize,
        outcome: TraceStepOutcome,
        log_prob: f64,
        observable_values: Vec<f64>,
    ) -> Self {
        Self {
            chain_id,
            step,
            outcome,
            log_prob,
            observable_values,
        }
    }

    /// Chain identifier for this row.
    pub const fn chain_id(&self) -> ChainId {
        self.chain_id
    }

    /// Completed step number for this row.
    #[must_use]
    pub const fn step(&self) -> usize {
        self.step
    }

    /// Acceptance/proposal outcome for this row.
    pub const fn outcome(&self) -> TraceStepOutcome {
        self.outcome
    }

    /// Cached target log-probability after this step.
    #[must_use]
    pub const fn log_prob(&self) -> f64 {
        self.log_prob
    }

    /// Numeric observable values in the same order as the trace headers.
    #[must_use]
    pub fn observable_values(&self) -> &[f64] {
        &self.observable_values
    }
}

/// Errors returned while constructing trace data.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TraceError {
    /// An observable name was empty.
    EmptyObservableName {
        /// Zero-based position of the empty name.
        index: usize,
    },
    /// An observable name appeared more than once.
    DuplicateObservableName {
        /// Duplicated observable name.
        name: String,
    },
    /// A row had a different number of values than the trace header.
    ObservableCountMismatch {
        /// Number of values required by the header.
        expected: usize,
        /// Number of values provided for the row.
        actual: usize,
    },
    /// Two traces used different observable columns.
    ObservableNamesMismatch {
        /// Observable columns required by the receiving trace.
        expected: Vec<String>,
        /// Observable columns provided by the appended trace.
        actual: Vec<String>,
    },
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservableName { index } => {
                write!(f, "observable name at index {index} is empty")
            }
            Self::DuplicateObservableName { name } => {
                write!(f, "observable name {name:?} appears more than once")
            }
            Self::ObservableCountMismatch { expected, actual } => write!(
                f,
                "trace row has {actual} observable values, expected {expected}"
            ),
            Self::ObservableNamesMismatch { expected, actual } => write!(
                f,
                "trace observable columns differ: expected {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl Error for TraceError {}

/// Multi-chain numeric trace with shared observable columns.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Trace {
    observable_names: Vec<String>,
    records: Vec<TraceRecord>,
}

impl Trace {
    /// Create an empty trace with the given observable columns.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{Trace, TraceError};
    ///
    /// let trace = Trace::new(["energy", "magnetization"])?;
    ///
    /// assert!(trace.is_empty());
    /// assert_eq!(trace.observable_names(), &["energy", "magnetization"]);
    /// # Ok::<(), TraceError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TraceError`] if an observable name is empty or duplicated.
    pub fn new(
        observable_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, TraceError> {
        Ok(Self {
            observable_names: validate_observable_names(observable_names)?,
            records: Vec::new(),
        })
    }

    /// Observable column names.
    #[must_use]
    pub fn observable_names(&self) -> &[String] {
        &self.observable_names
    }

    /// Recorded rows.
    pub fn records(&self) -> &[TraceRecord] {
        &self.records
    }

    /// Whether the trace contains no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Number of recorded rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.records.len()
    }

    /// Append one record.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{
    ///     ChainId, Trace, TraceError, TraceRecord, TraceStepOutcome,
    /// };
    ///
    /// let mut trace = Trace::new(["energy"])?;
    /// trace.push(TraceRecord::new(
    ///     ChainId::new(0),
    ///     1,
    ///     TraceStepOutcome::accepted(),
    ///     -2.0,
    ///     vec![2.0],
    /// ))?;
    ///
    /// assert_eq!(trace.len(), 1);
    /// # Ok::<(), TraceError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::ObservableCountMismatch`] if the row does not
    /// match this trace's observable columns.
    pub fn push(&mut self, record: TraceRecord) -> Result<(), TraceError> {
        let actual = record.observable_values.len();
        let expected = self.observable_names.len();
        if actual != expected {
            return Err(TraceError::ObservableCountMismatch { expected, actual });
        }
        self.records.push(record);
        Ok(())
    }

    /// Append all rows from another trace with identical observable columns.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{Trace, TraceError};
    ///
    /// let mut combined = Trace::new(["energy"])?;
    /// let other = Trace::new(["energy"])?;
    ///
    /// combined.extend(other)?;
    /// assert!(combined.is_empty());
    /// # Ok::<(), TraceError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::ObservableNamesMismatch`] if the traces use
    /// different observable columns.
    pub fn extend(&mut self, other: Self) -> Result<(), TraceError> {
        if self.observable_names != other.observable_names {
            return Err(TraceError::ObservableNamesMismatch {
                expected: self.observable_names.clone(),
                actual: other.observable_names,
            });
        }
        self.records.extend(other.records);
        Ok(())
    }

    /// Iterate over rows for one chain.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{
    ///     ChainId, Trace, TraceError, TraceRecord, TraceStepOutcome,
    /// };
    ///
    /// let mut trace = Trace::new(["energy"])?;
    /// trace.push(TraceRecord::new(
    ///     ChainId::new(1),
    ///     1,
    ///     TraceStepOutcome::accepted(),
    ///     0.0,
    ///     vec![1.0],
    /// ))?;
    ///
    /// assert_eq!(trace.records_for_chain(ChainId::new(1)).count(), 1);
    /// assert_eq!(trace.records_for_chain(ChainId::new(0)).count(), 0);
    /// # Ok::<(), TraceError>(())
    /// ```
    pub fn records_for_chain(&self, chain_id: ChainId) -> impl Iterator<Item = &TraceRecord> + '_ {
        self.records
            .iter()
            .filter(move |record| record.chain_id == chain_id)
    }

    /// Acceptance rate for one chain, counting no-proposal self-loops as
    /// rejected steps.
    ///
    /// Returns `0.0` when no rows exist for `chain_id`.
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use markov_chain_monte_carlo::prelude::{
    ///     ChainId, Trace, TraceError, TraceRecord, TraceStepOutcome,
    /// };
    ///
    /// let mut trace = Trace::new(["energy"])?;
    /// trace.push(TraceRecord::new(
    ///     ChainId::new(0),
    ///     1,
    ///     TraceStepOutcome::accepted(),
    ///     0.0,
    ///     vec![1.0],
    /// ))?;
    /// trace.push(TraceRecord::new(
    ///     ChainId::new(0),
    ///     2,
    ///     TraceStepOutcome::rejected_proposal(),
    ///     0.0,
    ///     vec![1.0],
    /// ))?;
    ///
    /// assert_relative_eq!(trace.acceptance_rate(ChainId::new(0)), 0.5);
    /// # Ok::<(), TraceError>(())
    /// ```
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "trace lengths are diagnostic counters and fit in f64 for practical runs"
    )]
    pub fn acceptance_rate(&self, chain_id: ChainId) -> f64 {
        let mut accepted = 0_usize;
        let mut total = 0_usize;
        for record in self.records_for_chain(chain_id) {
            total = total.saturating_add(1);
            if record.outcome.is_accepted() {
                accepted = accepted.saturating_add(1);
            }
        }
        if total == 0 {
            0.0
        } else {
            accepted as f64 / total as f64
        }
    }

    /// Write the trace in CSV format.
    ///
    /// The fixed columns are `chain_id`, `step`, `accepted`, `proposed`, and
    /// `log_prob`, followed by the observable columns supplied when the trace
    /// was created.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{
    ///     ChainId, Trace, TraceError, TraceRecord, TraceStepOutcome,
    /// };
    ///
    /// let mut trace = Trace::new(["energy"])?;
    /// trace.push(TraceRecord::new(
    ///     ChainId::new(0),
    ///     1,
    ///     TraceStepOutcome::accepted(),
    ///     -1.0,
    ///     vec![1.5],
    /// ))?;
    ///
    /// let mut csv = Vec::new();
    /// trace.write_csv(&mut csv).expect("writing to Vec cannot fail");
    ///
    /// assert_eq!(
    ///     String::from_utf8(csv).expect("CSV output is UTF-8"),
    ///     "chain_id,step,accepted,proposed,log_prob,energy\n0,1,true,true,-1,1.5\n",
    /// );
    /// # Ok::<(), TraceError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns any I/O error reported by `writer`.
    pub fn write_csv(&self, mut writer: impl Write) -> io::Result<()> {
        writer.write_all(b"chain_id,step,accepted,proposed,log_prob")?;
        for name in &self.observable_names {
            writer.write_all(b",")?;
            write_csv_field(&mut writer, name)?;
        }
        writer.write_all(b"\n")?;

        for record in &self.records {
            write!(
                writer,
                "{},{},{},{},{}",
                record.chain_id.get(),
                record.step,
                record.outcome.is_accepted(),
                record.outcome.had_proposal(),
                record.log_prob
            )?;
            for value in &record.observable_values {
                write!(writer, ",{value}")?;
            }
            writer.write_all(b"\n")?;
        }
        Ok(())
    }
}

/// Recorder for one chain within a multi-chain trace.
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct TraceRecorder {
    chain_id: ChainId,
    trace: Trace,
}

impl TraceRecorder {
    /// Create a recorder for one chain.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{ChainId, TraceError, TraceRecorder};
    ///
    /// let recorder = TraceRecorder::new(ChainId::new(3), ["energy"])?;
    ///
    /// assert_eq!(recorder.chain_id(), ChainId::new(3));
    /// assert_eq!(recorder.trace().observable_names(), &["energy"]);
    /// # Ok::<(), TraceError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TraceError`] if an observable name is empty or duplicated.
    pub fn new(
        chain_id: ChainId,
        observable_names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, TraceError> {
        Ok(Self {
            chain_id,
            trace: Trace::new(observable_names)?,
        })
    }

    /// Chain identifier used by this recorder.
    pub const fn chain_id(&self) -> ChainId {
        self.chain_id
    }

    /// Borrow the accumulated trace.
    pub const fn trace(&self) -> &Trace {
        &self.trace
    }

    /// Consume the recorder and return the accumulated trace.
    pub fn into_trace(self) -> Trace {
        self.trace
    }

    /// Record the current state of `chain` after a completed step.
    ///
    /// Observable values must be supplied in the same order as the recorder's
    /// observable names.
    ///
    /// ```
    /// use markov_chain_monte_carlo::prelude::{
    ///     Chain, ChainId, Target, TraceError, TraceRecorder, TraceStepOutcome,
    /// };
    ///
    /// struct Flat;
    /// impl Target<i32> for Flat {
    ///     fn log_prob(&self, _: &i32) -> f64 { 0.0 }
    /// }
    ///
    /// let chain = Chain::new(4, &Flat)
    ///     .expect("flat target returns a finite log-probability");
    /// let mut recorder = TraceRecorder::new(ChainId::new(0), ["value"])?;
    ///
    /// recorder.record(&chain, TraceStepOutcome::accepted(), [f64::from(*chain.state())])?;
    ///
    /// assert_eq!(recorder.trace().records()[0].observable_values(), &[4.0]);
    /// # Ok::<(), TraceError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TraceError::ObservableCountMismatch`] if `observable_values`
    /// does not match the recorder header.
    pub fn record<S>(
        &mut self,
        chain: &Chain<S>,
        outcome: TraceStepOutcome,
        observable_values: impl IntoIterator<Item = f64>,
    ) -> Result<(), TraceError> {
        let values = observable_values.into_iter().collect();
        let record = TraceRecord::new(
            self.chain_id,
            chain.total_steps(),
            outcome,
            chain.log_prob(),
            values,
        );
        self.trace.push(record)
    }
}

/// Validate trace observable names before storing them as CSV/header metadata.
///
/// This centralizes the non-empty and uniqueness checks so every trace row can
/// rely on stable, unambiguous observable columns.
fn validate_observable_names(
    observable_names: impl IntoIterator<Item = impl Into<String>>,
) -> Result<Vec<String>, TraceError> {
    let names: Vec<_> = observable_names.into_iter().map(Into::into).collect();
    let mut seen = BTreeSet::new();
    for (index, name) in names.iter().enumerate() {
        if name.is_empty() {
            return Err(TraceError::EmptyObservableName { index });
        }
        if !seen.insert(name.as_str()) {
            return Err(TraceError::DuplicateObservableName { name: name.clone() });
        }
    }
    Ok(names)
}

/// Write one CSV field, quoting it only when CSV syntax requires escaping.
fn write_csv_field(writer: &mut impl Write, value: &str) -> io::Result<()> {
    if value.contains([',', '"', '\n', '\r']) {
        writer.write_all(b"\"")?;
        for byte in value.bytes() {
            if byte == b'"' {
                writer.write_all(b"\"\"")?;
            } else {
                writer.write_all(&[byte])?;
            }
        }
        writer.write_all(b"\"")?;
    } else {
        writer.write_all(value.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::{McmcError, Target};

    struct Flat;

    impl Target<i32> for Flat {
        fn log_prob(&self, _: &i32) -> f64 {
            0.0
        }
    }

    #[test]
    fn recorder_collects_chain_rows() -> Result<(), TraceError> {
        let chain = Chain::new(3, &Flat).expect("flat target has valid log-probability");
        let mut recorder = TraceRecorder::new(ChainId::new(2), ["energy", "magnetization"])?;

        recorder.record(&chain, TraceStepOutcome::accepted(), [1.25_f64, -0.5_f64])?;

        let trace = recorder.trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.records()[0].chain_id(), ChainId::new(2));
        assert_eq!(trace.records()[0].step(), 0);
        assert_eq!(trace.records()[0].outcome(), TraceStepOutcome::accepted());
        assert_relative_eq!(trace.records()[0].log_prob(), 0.0);
        assert_eq!(trace.records()[0].observable_values(), &[1.25, -0.5]);
        assert_relative_eq!(trace.acceptance_rate(ChainId::new(2)), 1.0);
        Ok(())
    }

    #[test]
    fn trace_rejects_bad_headers_and_row_widths() {
        assert_eq!(
            Trace::new(["energy", ""]).unwrap_err(),
            TraceError::EmptyObservableName { index: 1 }
        );
        assert_eq!(
            Trace::new(["energy", "energy"]).unwrap_err(),
            TraceError::DuplicateObservableName {
                name: "energy".to_owned()
            }
        );

        let mut trace = Trace::new(["energy"]).unwrap();
        let record = TraceRecord::new(
            ChainId::new(0),
            1,
            TraceStepOutcome::accepted(),
            0.0,
            vec![1.0, 2.0],
        );
        assert_eq!(
            trace.push(record).unwrap_err(),
            TraceError::ObservableCountMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn trace_extend_rejects_mismatched_headers() {
        let mut trace = Trace::new(["energy"]).expect("valid observable name");
        let other = Trace::new(["energy", "magnetization"]).expect("valid observable names");

        assert_eq!(
            trace.extend(other).unwrap_err(),
            TraceError::ObservableNamesMismatch {
                expected: vec!["energy".to_owned()],
                actual: vec!["energy".to_owned(), "magnetization".to_owned()]
            }
        );
    }

    #[test]
    fn trace_extend_reports_equal_width_name_mismatch() {
        let mut trace = Trace::new(["energy"]).expect("valid observable name");
        let other = Trace::new(["magnetization"]).expect("valid observable name");

        assert_eq!(
            trace.extend(other).unwrap_err(),
            TraceError::ObservableNamesMismatch {
                expected: vec!["energy".to_owned()],
                actual: vec!["magnetization".to_owned()]
            }
        );
    }

    #[test]
    fn trace_writes_csv() -> Result<(), io::Error> {
        let mut trace = Trace::new(["energy", "quoted,name"]).expect("valid observable names");
        trace
            .push(TraceRecord::new(
                ChainId::new(0),
                1,
                TraceStepOutcome::rejected_proposal(),
                -2.5,
                vec![3.0, 4.0],
            ))
            .expect("row width matches");

        let mut csv = Vec::new();
        trace.write_csv(&mut csv)?;

        let csv = String::from_utf8(csv).expect("CSV is valid UTF-8");
        assert_eq!(
            csv,
            "chain_id,step,accepted,proposed,log_prob,energy,\"quoted,name\"\n\
             0,1,false,true,-2.5,3,4\n"
        );
        Ok(())
    }

    #[test]
    fn trace_csv_escapes_quoted_headers() -> Result<(), io::Error> {
        let mut trace = Trace::new([r#"quote"field"#]).expect("valid observable name");
        trace
            .push(TraceRecord::new(
                ChainId::new(0),
                1,
                TraceStepOutcome::accepted(),
                0.0,
                vec![1.0],
            ))
            .expect("row width matches");

        let mut csv = Vec::new();
        trace.write_csv(&mut csv)?;

        let csv = String::from_utf8(csv).expect("CSV is valid UTF-8");
        assert_eq!(
            csv,
            "chain_id,step,accepted,proposed,log_prob,\"quote\"\"field\"\n\
             0,1,true,true,0,1\n"
        );
        Ok(())
    }

    #[test]
    fn delayed_step_outcome_converts_to_trace_outcome() {
        assert_eq!(
            TraceStepOutcome::from(StepOutcome::Accepted),
            TraceStepOutcome::accepted()
        );
        assert_eq!(
            TraceStepOutcome::from(StepOutcome::RejectedProposal),
            TraceStepOutcome::rejected_proposal()
        );
        assert_eq!(
            TraceStepOutcome::from(StepOutcome::NoProposal),
            TraceStepOutcome::no_proposal()
        );
    }

    #[test]
    fn delayed_step_reference_converts_to_trace_outcome() {
        let step = Step {
            outcome: StepOutcome::NoProposal,
            info: None::<()>,
            log_prob_before: 0.0,
            log_prob_after: None,
            log_alpha: None,
        };

        assert_eq!(
            TraceStepOutcome::from(&step),
            TraceStepOutcome::no_proposal()
        );
    }

    #[test]
    fn proposal_acceptance_adapter_preserves_proposal_presence() {
        assert_eq!(
            TraceStepOutcome::from_proposal_acceptance(true),
            TraceStepOutcome::accepted()
        );
        assert_eq!(
            TraceStepOutcome::from_proposal_acceptance(false),
            TraceStepOutcome::rejected_proposal()
        );
    }

    #[test]
    fn empty_chain_acceptance_rate_is_zero() -> Result<(), TraceError> {
        let trace = Trace::new(["energy"])?;

        assert_relative_eq!(trace.acceptance_rate(ChainId::new(0)), 0.0);
        Ok(())
    }

    #[test]
    fn chain_construction_still_reports_mcmc_errors() {
        struct NanTarget;

        impl Target<i32> for NanTarget {
            fn log_prob(&self, _: &i32) -> f64 {
                f64::NAN
            }
        }

        assert_eq!(
            Chain::new(0, &NanTarget).unwrap_err(),
            McmcError::NanInitialLogProb
        );
    }
}
