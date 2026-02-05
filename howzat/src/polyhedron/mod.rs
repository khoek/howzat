//! Polyhedron representation conversion and analysis.
//!
//! This module provides the core [`PolyhedronOutput`] type for H-representation to V-representation
//! conversion (and vice versa), along with incidence, adjacency, and repair functionality.

mod build;
mod int;
pub(crate) mod repair;
mod resolve;

use crate::HowzatError as Error;
use crate::dd::DefaultNormalizer;
use crate::dd::{ConeOptions, RayId};
use crate::lp::{LpResult, LpSolver};
use crate::matrix::{LpMatrix, LpMatrixBuilder};
use calculo::num::{CoerceFrom, Epsilon, Normalizer, Num, Rat};
use hullabaloo::adjacency::{
    self as hulla_adjacency, AdjacencyStore, Graph, RowsByNodeAdjacencyOptions,
};
use hullabaloo::matrix::BasisMatrix;
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{
    AdjacencyOutput, ComputationStatus, DualRepresentation, IncidenceOutput, InequalityKind,
    Representation, RepresentationKind, Row, RowIndex, RowSet,
};
use smallvec::SmallVec;
use std::time::{Duration, Instant};

pub use int::IntRowMatrix;

// Re-export integer utilities for repair.rs
pub(crate) use int::{
    bareiss_solve_det_times_matrix_in_place, BareissSolveScratch, scaled_integer_rows,
    scaled_integer_vec, select_row_basis_rows_int, solve_nullspace_1d_rows_with_unit_cols_bareiss_int,
};

type DualOf<R> = <R as DualRepresentation>::Dual;

/// Build adjacency from pre-computed incidence SetFamily.
///
/// Pass `candidate_edges` to use edge hints from a prior adjacency graph.
/// Set `assume_nondegenerate` for the fast-path when degeneracy is impossible.
pub(crate) fn build_adjacency(
    incidence: &SetFamily,
    output_linearity: &RowSet,
    active_rows: &RowSet,
    adj_dim: usize,
    candidate_edges: Option<&[(usize, usize)]>,
    assume_nondegenerate: bool,
    profile: Option<&mut AdjacencyBuildProfile>,
) -> Option<SetFamily> {
    build_adjacency_with(
        incidence,
        output_linearity,
        active_rows,
        adj_dim,
        candidate_edges,
        assume_nondegenerate,
        profile,
    )
}

pub(crate) fn build_adjacency_with<Adj: AdjacencyStore>(
    incidence: &SetFamily,
    output_linearity: &RowSet,
    active_rows: &RowSet,
    adj_dim: usize,
    candidate_edges: Option<&[(usize, usize)]>,
    assume_nondegenerate: bool,
    mut profile: Option<&mut AdjacencyBuildProfile>,
) -> Option<Adj> {
    let family_size = incidence.family_size();
    if family_size < 2 {
        return None;
    }

    let t_start = profile.is_some().then(Instant::now);

    let non_linearity_count = family_size - output_linearity.cardinality();
    if non_linearity_count < 2 {
        return None;
    }

    let row_capacity = incidence.set_capacity();
    debug_assert_eq!(
        output_linearity.len(),
        family_size,
        "output_linearity capacity must match family size"
    );
    debug_assert_eq!(
        active_rows.len(),
        row_capacity,
        "active_rows capacity must match incidence row capacity"
    );

    // Record profiling info
    if let Some(p) = profile.as_mut() {
        let p = &mut **p;
        p.facets_total = family_size;
        p.facets_non_lineality = non_linearity_count;
        p.active_rows = active_rows.cardinality();
        p.candidate_edges = candidate_edges.map(|e| e.len());
        p.adjacency_dense_bytes = dense_set_family_bytes(family_size, family_size);
    }

    // Convert incidence into per-node sorted row lists, filtered by `active_rows`.
    let active_all = active_rows.cardinality() == row_capacity;
    let mut rows_by_node: Vec<Vec<usize>> = Vec::with_capacity(family_size);
    for idx in 0..family_size {
        if output_linearity.contains(idx) {
            rows_by_node.push(Vec::new());
            continue;
        }
        let set = incidence
            .set(idx)
            .unwrap_or_else(|| panic!("SetFamily must contain set for index {idx}"));
        let mut rows: Vec<usize> = Vec::with_capacity(set.cardinality());
        if active_all {
            rows.extend(set.iter().map(|id| id.as_index()));
        } else {
            rows.extend(
                set.iter()
                    .map(|id| id.as_index())
                    .filter(|&r| active_rows.contains(r)),
            );
        }
        rows_by_node.push(rows);
    }

    let excluded_mask = (output_linearity.cardinality() > 0).then(|| {
        let mut mask = vec![false; family_size];
        for idx in output_linearity.iter() {
            mask[idx.as_index()] = true;
        }
        mask
    });
    let excluded_nodes = excluded_mask.as_deref();

    let options = RowsByNodeAdjacencyOptions {
        excluded_nodes,
        candidate_edges,
        assume_nondegenerate,
    };

    let adj = hulla_adjacency::adjacency_from_rows_by_node_with::<Adj::Builder>(
        &rows_by_node,
        row_capacity,
        adj_dim,
        options,
    );

    // Record timing and strategy
    if let Some(p) = profile.as_mut() {
        let p = &mut **p;
        p.time_adjacency = t_start.map(|t| t.elapsed());
        p.strategy = if assume_nondegenerate {
            AdjacencyBuildStrategy::AssumeNondegenerate
        } else if candidate_edges.is_some() {
            AdjacencyBuildStrategy::CandidateEdgesSparseMembers
        } else {
            AdjacencyBuildStrategy::AllPairsSparseMembers
        };
        p.edges_output = count_undirected_edges(&adj);
    }

    Some(adj)
}

fn dense_set_family_bytes(family_size: usize, set_capacity: usize) -> usize {
    if family_size == 0 || set_capacity == 0 {
        return 0;
    }
    let word_bits = usize::BITS as usize;
    let words = set_capacity.div_ceil(word_bits);
    family_size
        .saturating_mul(words)
        .saturating_mul(std::mem::size_of::<usize>())
}

fn count_undirected_edges(adj: &impl Graph) -> usize {
    let mut count = 0usize;
    for i in 0..adj.node_count() {
        for j in adj.neighbors(i) {
            if i < j {
                count += 1;
            }
        }
    }
    count
}

#[derive(Clone, Debug)]
pub struct PolyhedronOptions {
    pub output_incidence: IncidenceOutput,
    pub input_incidence: IncidenceOutput,
    pub output_adjacency: AdjacencyOutput,
    pub input_adjacency: AdjacencyOutput,
    pub save_basis_and_tableau: bool,
    pub save_repair_hints: bool,
    /// Collect counters/timers for adjacency construction (off by default).
    pub profile_adjacency: bool,
}

impl Default for PolyhedronOptions {
    fn default() -> Self {
        Self {
            output_incidence: IncidenceOutput::Off,
            input_incidence: IncidenceOutput::Off,
            output_adjacency: AdjacencyOutput::Off,
            input_adjacency: AdjacencyOutput::Off,
            save_basis_and_tableau: false,
            save_repair_hints: false,
            profile_adjacency: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AdjacencyBuildStrategy {
    /// Adjacency was not requested / not built.
    #[default]
    None,
    /// Degenerate-safe adjacency by enumerating all pairs and using sparse membership intersections.
    AllPairsSparseMembers,
    /// Degenerate-safe adjacency by validating only candidate edges using sparse membership intersections.
    CandidateEdgesSparseMembers,
    /// Fast path: ridge hashing for simplicial keys (share-all-but-one).
    SimplicialRidgeHash,
    /// DD fast path when nondegeneracy is assumed (no containment checks).
    AssumeNondegenerate,
}

#[derive(Clone, Debug, Default)]
pub struct AdjacencyBuildProfile {
    pub strategy: AdjacencyBuildStrategy,
    /// Total output rows (including lineality).
    pub facets_total: usize,
    /// Non-lineality output rows (the FR graph vertex count).
    pub facets_non_lineality: usize,
    /// Active input-row universe size (after slack/lineality filtering).
    pub active_rows: usize,
    /// Candidate edges supplied (when using candidate-edge construction).
    pub candidate_edges: Option<usize>,

    pub pairs_considered: usize,
    pub pairs_passing_threshold: usize,
    pub containment_checks: usize,
    pub containment_early_exit: usize,
    pub edges_output: usize,

    /// Proxy for sparse-membership storage: total (vertex,facet) memberships stored.
    pub membership_entries: usize,
    /// Proxy for adjacency storage: bytes for a dense `F x F` bitset family.
    pub adjacency_dense_bytes: usize,

    pub time_incidence: Option<Duration>,
    pub time_adjacency: Option<Duration>,
}

#[derive(Clone, Debug, Default)]
pub struct ResolveOptions {
    pub relaxed: bool,

    /// Partial resolution normally scans all input rows to orient the resolved normal, detect
    /// violations, and compute the exact incidence (zero-set). When set, partial resolution will
    /// instead trust the incidence certificate (plus any DD-propagated near-zero suspects) to build
    /// the zero-set, and orient the normal using a small number of non-incident witnesses.
    ///
    /// This is intended for fast repair pipelines and may accept incorrect facets if the
    /// certificate incidence is wrong.
    pub partial_use_certificate_only: bool,
}

#[derive(Clone, Debug)]
pub enum PartialResolveIssue<N: Num> {
    WitnessNotOneDim {
        output_row: Row,
    },
    InfeasibleResolvedRow {
        output_row: Row,
        constraint: Row,
        kind: InequalityKind,
        value: N,
    },
}

#[derive(Clone, Debug)]
pub struct PartialResolveResult<N: Num, R: DualRepresentation> {
    poly: PolyhedronOutput<N, R>,
    kept_output_rows: Vec<Row>,
    issues: Vec<PartialResolveIssue<N>>,
}

impl<N: Num, R: DualRepresentation> PartialResolveResult<N, R> {
    pub fn polyhedron(&self) -> &PolyhedronOutput<N, R> {
        &self.poly
    }

    pub fn into_polyhedron(self) -> PolyhedronOutput<N, R> {
        self.poly
    }

    pub fn kept_output_rows(&self) -> &[Row] {
        &self.kept_output_rows
    }

    pub fn issues(&self) -> &[PartialResolveIssue<N>] {
        &self.issues
    }
}

#[derive(Clone, Debug)]
pub struct PreparedPartialResolveResult<N: Rat, R: DualRepresentation> {
    partial: PartialResolveResult<N, R>,
    int_input_rows: IntRowMatrix<<N as Rat>::Int>,
    int_output_rows: IntRowMatrix<<N as Rat>::Int>,
    redund_cols: Vec<usize>,
    redund_mask: Vec<bool>,
}

impl<N: Rat, R: DualRepresentation> PreparedPartialResolveResult<N, R> {
    pub fn partial(&self) -> &PartialResolveResult<N, R> {
        &self.partial
    }

    pub fn into_partial(self) -> PartialResolveResult<N, R> {
        self.partial
    }

    pub fn polyhedron(&self) -> &PolyhedronOutput<N, R> {
        self.partial.polyhedron()
    }

    pub fn kept_output_rows(&self) -> &[Row] {
        self.partial.kept_output_rows()
    }

    pub fn issues(&self) -> &[PartialResolveIssue<N>] {
        self.partial.issues()
    }

    pub fn int_input_rows(&self) -> &IntRowMatrix<<N as Rat>::Int> {
        &self.int_input_rows
    }

    pub fn int_output_rows(&self) -> &IntRowMatrix<<N as Rat>::Int> {
        &self.int_output_rows
    }

    pub fn redund_cols(&self) -> &[usize] {
        &self.redund_cols
    }

    pub fn redund_mask(&self) -> &[bool] {
        &self.redund_mask
    }
}

/// Partial certificate resolution prepared for high-performance facet-graph repair.
///
/// Unlike [`PreparedPartialResolveResult`], this avoids constructing the resolved output matrix in
/// the exact numeric type. It retains the exact input matrix (for rebuilding) along with integer
/// forms of the input/output rows and the facet incidence vertex lists for the kept rows.
#[derive(Clone, Debug)]
pub struct PreparedPartialResolveMinimal<N: Rat, R: DualRepresentation> {
    template: PolyhedronOutput<N, R>,
    kept_output_rows: Vec<Row>,
    issues: Vec<PartialResolveIssue<N>>,
    facet_vertices: Vec<Vec<Row>>,
    int_input_rows: IntRowMatrix<<N as Rat>::Int>,
    int_output_rows: IntRowMatrix<<N as Rat>::Int>,
    redund_cols: Vec<usize>,
    redund_mask: Vec<bool>,
}

impl<N: Rat, R: DualRepresentation> PreparedPartialResolveMinimal<N, R> {
    pub fn template(&self) -> &PolyhedronOutput<N, R> {
        &self.template
    }

    pub fn kept_output_rows(&self) -> &[Row] {
        &self.kept_output_rows
    }

    pub fn issues(&self) -> &[PartialResolveIssue<N>] {
        &self.issues
    }

    pub fn facet_vertices(&self) -> &[Vec<Row>] {
        &self.facet_vertices
    }

    pub fn int_input_rows(&self) -> &IntRowMatrix<<N as Rat>::Int> {
        &self.int_input_rows
    }

    pub fn int_output_rows(&self) -> &IntRowMatrix<<N as Rat>::Int> {
        &self.int_output_rows
    }

    pub fn redund_cols(&self) -> &[usize] {
        &self.redund_cols
    }

    pub fn redund_mask(&self) -> &[bool] {
        &self.redund_mask
    }
}

#[derive(Clone, Debug)]
pub enum VerificationIssue<N: Num> {
    StatusInconsistent {
        status: ComputationStatus,
        output_rows: usize,
    },
    ConstraintViolation {
        output_row: Row,
        constraint: Row,
        kind: InequalityKind,
        value: N,
    },
    OutputIncidenceMismatch {
        output_row: Row,
        expected: RowSet,
        observed: RowSet,
    },
    InputIncidenceMismatch {
        input_row: Row,
        expected: RowSet,
        observed: RowSet,
    },
    RedundantRowsMismatch {
        expected: RowSet,
        observed: RowSet,
    },
    DominantRowsMismatch {
        expected: RowSet,
        observed: RowSet,
    },
    ComputationError(Error),
}

impl<N: Num> std::error::Error for VerificationIssue<N> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ComputationError(err) => Some(err),
            _ => None,
        }
    }
}

impl<N: Num> std::fmt::Display for VerificationIssue<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StatusInconsistent { status, output_rows } => {
                write!(f, "status inconsistent (status={status:?}, output_rows={output_rows})")
            }
            Self::ConstraintViolation {
                output_row,
                constraint,
                kind,
                value,
            } => write!(
                f,
                "constraint violation (output_row={output_row}, constraint={constraint}, kind={kind:?}, value={value:?})"
            ),
            Self::OutputIncidenceMismatch {
                output_row,
                expected,
                observed,
            } => write!(
                f,
                "output incidence mismatch (output_row={output_row}, expected={} observed={})",
                expected.cardinality(),
                observed.cardinality()
            ),
            Self::InputIncidenceMismatch {
                input_row,
                expected,
                observed,
            } => write!(
                f,
                "input incidence mismatch (input_row={input_row}, expected={} observed={})",
                expected.cardinality(),
                observed.cardinality()
            ),
            Self::RedundantRowsMismatch { expected, observed } => write!(
                f,
                "redundant rows mismatch (expected={} observed={})",
                expected.cardinality(),
                observed.cardinality()
            ),
            Self::DominantRowsMismatch { expected, observed } => write!(
                f,
                "dominant rows mismatch (expected={} observed={})",
                expected.cardinality(),
                observed.cardinality()
            ),
            Self::ComputationError(err) => write!(f, "computation error: {err}"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ResolveError<N: Num> {
    MissingCertificate,
    StatusNotAllFound {
        status: ComputationStatus,
    },
    ConversionFailure,
    CertificateShapeMismatch,
    WitnessNotOneDim {
        output_row: Row,
    },
    InfeasibleResolvedRow {
        output_row: Row,
        constraint: Row,
        kind: InequalityKind,
        value: N,
    },
    VerificationFailed {
        issues: Vec<VerificationIssue<N>>,
    },
    ComputationError(Error),
}

impl<N: Num> std::error::Error for ResolveError<N> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ComputationError(err) => Some(err),
            _ => None,
        }
    }
}

impl<N: Num> std::fmt::Display for ResolveError<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCertificate => write!(f, "missing certificate"),
            Self::StatusNotAllFound { status } => {
                write!(f, "status not all found (status={status:?})")
            }
            Self::ConversionFailure => write!(f, "numeric conversion failed"),
            Self::CertificateShapeMismatch => write!(f, "certificate shape mismatch"),
            Self::WitnessNotOneDim { output_row } => {
                write!(f, "witness not 1-dimensional (output_row={output_row})")
            }
            Self::InfeasibleResolvedRow {
                output_row,
                constraint,
                kind,
                value,
            } => write!(
                f,
                "infeasible resolved row (output_row={output_row}, constraint={constraint}, kind={kind:?}, value={value:?})"
            ),
            Self::VerificationFailed { issues } => {
                write!(f, "verification failed (issues={})", issues.len())
            }
            Self::ComputationError(err) => write!(f, "computation error: {err}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PolyhedronOutput<N: Num, R: DualRepresentation, Adj = SetFamily> {
    representation: RepresentationKind,
    homogeneous: bool,
    dimension: Row,
    input: LpMatrix<N, R>,
    output: LpMatrix<N, DualOf<R>>,
    equality_kinds: Vec<InequalityKind>,
    linearity_dimension: Row,
    output_size: Row,
    incidence: Option<SetFamily>,
    adjacency: Option<Adj>,
    input_incidence: Option<SetFamily>,
    input_adjacency: Option<SetFamily>,
    redundant_rows: Option<RowSet>,
    dominant_rows: Option<RowSet>,
    status: ComputationStatus,
    is_empty: bool,
    cost_vector: Option<Vec<N>>,
    row_positions: RowIndex,
    column_mapping: Vec<Option<usize>>,
    trace: Option<Box<DdTrace<N>>>,
    repair_hints: Option<Box<DdRepairHints>>,
    adjacency_profile: Option<Box<AdjacencyBuildProfile>>,
}

pub type Polyhedron<N> = PolyhedronOutput<N, hullabaloo::types::Inequality>;

/// Optional DD-only trace artifacts (basis + tableau snapshots).
///
/// These are captured when `PolyhedronOptions.save_basis_and_tableau` is enabled.
#[derive(Clone, Debug)]
pub struct DdTrace<N: Num> {
    saved_basis: Option<BasisMatrix<N>>,
    tableau_snapshot: Option<Vec<Vec<N>>>,
    tableau_nonbasic: Option<Vec<isize>>,
    tableau_basic_col_for_row: Option<Vec<isize>>,
    tableau_rows: usize,
    tableau_cols: usize,
}

impl<N: Num> DdTrace<N> {
    pub fn saved_basis(&self) -> Option<&BasisMatrix<N>> {
        self.saved_basis.as_ref()
    }

    pub fn tableau_snapshot(&self) -> Option<&[Vec<N>]> {
        self.tableau_snapshot.as_deref()
    }

    pub fn tableau_nonbasic(&self) -> Option<&[isize]> {
        self.tableau_nonbasic.as_deref()
    }

    pub fn tableau_basic_col_for_row(&self) -> Option<&[isize]> {
        self.tableau_basic_col_for_row.as_deref()
    }

    pub fn tableau_rows(&self) -> usize {
        self.tableau_rows
    }

    pub fn tableau_cols(&self) -> usize {
        self.tableau_cols
    }
}

#[derive(Clone, Debug)]
pub struct DdRepairHints {
    facet_witness_basis: Vec<SmallVec<[Row; 16]>>,
    incoming_edge_hints: Option<Vec<Vec<DdEdgeHint>>>,
    facet_near_zero_offsets: Option<Vec<usize>>,
    facet_near_zero_rows: Vec<Row>,
    dedup_drops: u64,
}

impl DdRepairHints {
    pub fn dedup_drops(&self) -> u64 {
        self.dedup_drops
    }

    pub(crate) fn facet_near_zero_rows(&self, facet: usize) -> &[Row] {
        let Some(offsets) = self.facet_near_zero_offsets.as_deref() else {
            return &[];
        };
        let Some(&start) = offsets.get(facet) else {
            return &[];
        };
        let Some(&end) = offsets.get(facet + 1) else {
            return &[];
        };
        self.facet_near_zero_rows.get(start..end).unwrap_or(&[])
    }
}

#[derive(Clone, Debug)]
pub struct DdEdgeHint {
    neighbor: Row,
    ridge_basis: SmallVec<[Row; 16]>,
    drop_candidates: SmallVec<[Row; 8]>,
    from_witness: Option<Row>,
    minimizers: SmallVec<[Row; 8]>,
    minimizers_complete: bool,
    entered_row: Option<Row>,
}

#[derive(Clone, Debug)]
pub struct DdConfig {
    pub cone: ConeOptions,
    pub poly: PolyhedronOptions,
}

impl Default for DdConfig {
    fn default() -> Self {
        Self {
            cone: ConeOptions::default(),
            poly: PolyhedronOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LrsConfig {
    pub poly: PolyhedronOptions,
    pub lrs: crate::lrs::Options,
}

impl Default for LrsConfig {
    fn default() -> Self {
        Self {
            poly: PolyhedronOptions::default(),
            lrs: crate::lrs::Options::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PolyhedronBuilder<N: Num, R: DualRepresentation> {
    matrix: LpMatrix<N, R>,
    dd: DdConfig,
}

impl<N: Num, R: DualRepresentation> PolyhedronBuilder<N, R> {
    pub fn new(matrix: LpMatrix<N, R>) -> Self {
        Self {
            matrix,
            dd: DdConfig::default(),
        }
    }

    pub fn dd_config(&mut self, dd: DdConfig) -> &mut Self {
        self.dd = dd;
        self
    }

    pub fn cone_options(&mut self, options: ConeOptions) -> &mut Self {
        self.dd.cone = options;
        self
    }

    pub fn polyhedron_options(&mut self, options: PolyhedronOptions) -> &mut Self {
        self.dd.poly = options;
        self
    }

    pub fn run_dd<U: crate::dd::Umpire<N>>(
        &mut self,
        umpire: U,
    ) -> Result<PolyhedronOutput<N, R>, Error>
    {
        let matrix = std::mem::replace(&mut self.matrix, LpMatrix::<N, R>::new(0, 0));
        let dd = self.dd.clone();
        PolyhedronOutput::from_matrix_dd(matrix, dd, umpire)
    }

    pub fn run_dd_int(&mut self) -> Result<PolyhedronOutput<N, R>, Error>
    where
        N: Rat,
    {
        let matrix = std::mem::replace(&mut self.matrix, LpMatrix::<N, R>::new(0, 0));
        let dd = self.dd.clone();
        PolyhedronOutput::from_matrix_dd_int(matrix, dd)
    }

    pub fn run_dd_with_eps<E: Epsilon<N>>(
        &mut self,
        eps: E,
    ) -> Result<PolyhedronOutput<N, R>, Error>
    where
        N: DefaultNormalizer,
    {
        self.run_dd_with_eps_and_normalizer(eps, <N as DefaultNormalizer>::Norm::default())
    }

    pub fn run_dd_with_eps_and_normalizer<E: Epsilon<N>, NM: Normalizer<N>>(
        &mut self,
        eps: E,
        normalizer: NM,
    ) -> Result<PolyhedronOutput<N, R>, Error>
    {
        self.run_dd(crate::dd::SinglePrecisionUmpire::with_normalizer(
            eps, normalizer,
        ))
    }

    pub fn run_lrs_as_exact<Q, M>(
        &mut self,
        lrs_options: crate::lrs::Options,
        eps: &impl Epsilon<M>,
    ) -> Result<PolyhedronOutput<M, R>, Error>
    where
        Q: Rat + CoerceFrom<N>,
        M: Num + CoerceFrom<Q>,
    {
        let matrix = std::mem::replace(&mut self.matrix, LpMatrix::<N, R>::new(0, 0));
        let config = LrsConfig {
            poly: self.dd.poly.clone(),
            lrs: lrs_options,
        };
        PolyhedronOutput::<N, R>::from_matrix_lrs_as_exact::<Q, M>(matrix, config, eps)
    }
}

#[derive(Clone, Debug)]
pub struct RedundancyCertificate<N: Num> {
    pub coefficients: Vec<N>,
}

#[derive(Clone, Debug)]
pub struct RelativeInterior<N: Num> {
    pub implicit_linearity: RowSet,
    pub linearity_basis: RowSet,
    pub exists: bool,
    pub lp_solution: Option<LpResult<N>>,
}

#[derive(Clone, Debug)]
pub struct RestrictedFaceWitness<N: Num> {
    pub exists: bool,
    pub lp_solution: Option<LpResult<N>>,
}

#[derive(Clone, Debug)]
pub(crate) struct OutputRayData<N: Num> {
    pub ray_id: RayId,
    pub vector: Vec<N>,
    pub zero_set: RowSet,
    pub is_linearity: bool,
    pub near_zero_rows: Vec<Row>,
}

#[derive(Clone, Copy)]
pub(crate) struct IncidenceRequests {
    pub build_output_incidence: bool,
    pub build_input_incidence: bool,
    pub build_input_adjacency: bool,
    pub build_output_adjacency: bool,
}

impl IncidenceRequests {
    pub fn from_options(options: &PolyhedronOptions) -> Self {
        let build_input_incidence = options.input_incidence != IncidenceOutput::Off
            || options.input_adjacency != AdjacencyOutput::Off;
        let build_output_incidence =
            options.output_incidence != IncidenceOutput::Off || build_input_incidence;
        let build_input_adjacency = options.input_adjacency != AdjacencyOutput::Off;
        let build_output_adjacency = options.output_adjacency != AdjacencyOutput::Off;
        Self {
            build_output_incidence,
            build_input_incidence,
            build_input_adjacency,
            build_output_adjacency,
        }
    }

    pub fn wants_any_incidence(&self) -> bool {
        self.build_output_incidence || self.build_input_incidence || self.build_output_adjacency
    }
}

#[derive(Default)]
pub(crate) struct IncidenceArtifacts {
    pub incidence: Option<SetFamily>,
    pub input_incidence: Option<SetFamily>,
    pub redundant_rows: Option<RowSet>,
    pub dominant_rows: Option<RowSet>,
    pub input_adjacency: Option<SetFamily>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResolveMode {
    Strict,
    Partial,
}

impl<N: Num, R: DualRepresentation, Adj> PolyhedronOutput<N, R, Adj> {
    pub fn dd_trace(&self) -> Option<&DdTrace<N>> {
        self.trace.as_deref()
    }

    pub fn dd_repair_hints(&self) -> Option<&DdRepairHints> {
        self.repair_hints.as_deref()
    }

    pub fn representation(&self) -> RepresentationKind {
        self.representation
    }

    pub fn homogeneous(&self) -> bool {
        self.homogeneous
    }

    pub fn dimension(&self) -> Row {
        self.dimension
    }

    pub fn input(&self) -> &LpMatrix<N, R> {
        &self.input
    }

    pub fn output(&self) -> &LpMatrix<N, DualOf<R>> {
        &self.output
    }

    /// Return a formatted copy of the output matrix using `eps`.
    ///
    /// For generator outputs, this dehomogenizes point rows (so the leading coordinate becomes
    /// exactly `1`) and zeros near-zero entries under `eps`. For inequality outputs, this currently
    /// returns an unmodified clone.
    pub fn output_formatted(&self, eps: &impl Epsilon<N>) -> LpMatrix<N, DualOf<R>> {
        let mut out = self.output.clone();
        if DualOf::<R>::KIND == RepresentationKind::Generator {
            for row in out.rows_mut() {
                Self::normalize_generator(row, eps);
            }
        }
        out
    }

    pub fn into_output(self) -> LpMatrix<N, DualOf<R>> {
        self.output
    }

    pub fn builder(matrix: LpMatrix<N, R>) -> PolyhedronBuilder<N, R> {
        PolyhedronBuilder::new(matrix)
    }

    pub fn equality_kinds(&self) -> &[InequalityKind] {
        &self.equality_kinds
    }

    pub fn linearity_dimension(&self) -> Row {
        self.linearity_dimension
    }

    pub fn output_size(&self) -> Row {
        self.output_size
    }

    pub fn incidence(&self) -> Option<&SetFamily> {
        self.incidence.as_ref()
    }

    pub fn adjacency(&self) -> Option<&Adj> {
        self.adjacency.as_ref()
    }

    pub fn adjacency_profile(&self) -> Option<&AdjacencyBuildProfile> {
        self.adjacency_profile.as_deref()
    }

    pub fn input_incidence(&self) -> Option<&SetFamily> {
        self.input_incidence.as_ref()
    }

    pub fn input_adjacency(&self) -> Option<&SetFamily> {
        self.input_adjacency.as_ref()
    }

    pub fn redundant_rows(&self) -> Option<&RowSet> {
        self.redundant_rows.as_ref()
    }

    pub fn dominant_rows(&self) -> Option<&RowSet> {
        self.dominant_rows.as_ref()
    }

    pub fn status(&self) -> ComputationStatus {
        self.status
    }

    pub fn is_empty(&self) -> bool {
        self.is_empty
    }

    pub fn cost_vector(&self) -> Option<&[N]> {
        self.cost_vector.as_deref()
    }

    pub fn row_positions(&self) -> &RowIndex {
        &self.row_positions
    }

    pub fn column_mapping(&self) -> &[Option<usize>] {
        &self.column_mapping
    }

    pub fn output_required(&self) -> &LpMatrix<N, DualOf<R>> {
        &self.output
    }

    pub fn into_output_required(self) -> LpMatrix<N, DualOf<R>> {
        self.output
    }

    pub fn incidence_required(&self) -> &SetFamily {
        self.incidence
            .as_ref()
            .expect("requested incidence but it was not computed")
    }

    pub fn adjacency_required(&self) -> &Adj {
        self.adjacency
            .as_ref()
            .expect("requested adjacency but it was not computed")
    }

    pub fn input_incidence_required(&self) -> &SetFamily {
        self.input_incidence
            .as_ref()
            .expect("requested input incidence but it was not computed")
    }

    pub fn input_adjacency_required(&self) -> &SetFamily {
        self.input_adjacency
            .as_ref()
            .expect("requested input adjacency but it was not computed")
    }

    pub fn redundant_rows_required(&self) -> &RowSet {
        self.redundant_rows
            .as_ref()
            .expect("requested redundant rows but they were not computed")
    }

    pub fn dominant_rows_required(&self) -> &RowSet {
        self.dominant_rows
            .as_ref()
            .expect("requested dominant rows but they were not computed")
    }

    pub fn num_vertices(&self, eps: &impl Epsilon<N>) -> usize {
        self.output
            .rows()
            .filter(|row| row.first().is_some_and(|v| !eps.is_zero(v)))
            .count()
    }

    pub fn num_rays(&self, eps: &impl Epsilon<N>) -> usize {
        self.output
            .rows()
            .filter(|row| row.first().is_none_or(|v| eps.is_zero(v)))
            .count()
    }

    pub fn all_faces(&self, eps: &impl Epsilon<N>) -> Option<Vec<RowSet>> {
        let output = &self.output;
        let m = self.input.row_count();
        let mut faces: Vec<RowSet> = vec![RowSet::new(m)];
        let mut scratch = RowSet::new(m);
        for row in output.rows() {
            self.input.zero_set_into(row, &mut scratch, eps);
            if !faces.contains(&scratch) {
                faces.push(scratch.clone());
            }
        }
        let mut idx = 0;
        while idx < faces.len() {
            let current = faces[idx].clone();
            for row in output.rows() {
                self.input.zero_set_into(row, &mut scratch, eps);
                let inter = current.intersection(&scratch);
                if !faces.contains(&inter) {
                    faces.push(inter);
                }
            }
            idx += 1;
        }
        faces.sort_by_key(|f| f.cardinality());
        Some(faces)
    }

    pub fn faces_with_relative_interior(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Option<Vec<(RowSet, Option<RestrictedFaceWitness<N>>)>> {
        let faces = self.all_faces(eps)?;
        Some(
            faces
                .into_iter()
                .map(|face| {
                    let strict = RowSet::new(self.input.row_count());
                    let witness = self
                        .input
                        .restricted_face_witness(&face, &strict, LpSolver::DualSimplex, eps)
                        .ok();
                    (face, witness)
                })
                .collect(),
        )
    }
}

impl<N: Num, R: DualRepresentation, Adj> PolyhedronOutput<N, R, Adj> {
    pub(crate) fn redundant_cols_from_column_mapping(
        column_mapping: &[Option<usize>],
    ) -> Vec<usize> {
        column_mapping
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(col, map)| map.is_none().then_some(col))
            .collect()
    }

    pub(crate) fn normalize_generator(vector: &mut [N], eps: &impl Epsilon<N>) {
        // Match cddlib's dd_CopyRay: normalize only points (b != 0) so that the
        // homogeneous coordinate becomes exactly 1, leaving rays unchanged.
        if vector.is_empty() {
            return;
        }

        let b = vector[0].clone();
        if eps.is_zero(&b) {
            return;
        }

        let inv = N::one().ref_div(&b);
        vector[0] = N::one();
        for v in &mut vector[1..] {
            *v = v.ref_mul(&inv);
            if eps.is_zero(v) {
                *v = N::zero();
            }
        }
    }

    pub(crate) fn derive_equality_kinds<NN: Num, RR: Representation>(
        matrix: &LpMatrix<NN, RR>,
    ) -> Vec<InequalityKind> {
        (0..matrix.row_count())
            .map(|row| {
                if matrix.linearity().contains(row) {
                    InequalityKind::Equality
                } else {
                    InequalityKind::Inequality
                }
            })
            .collect()
    }

    pub(crate) fn expanded_row_positions(
        equality_kinds: &[InequalityKind],
        original_rows: usize,
    ) -> RowIndex {
        let mut positions = vec![-1; original_rows];
        let mut next_row = 0usize;
        for (idx, kind) in equality_kinds.iter().enumerate() {
            if idx < original_rows {
                positions[idx] = next_row as isize;
            }
            next_row += match kind {
                InequalityKind::Equality => 2,
                _ => 1,
            };
        }
        positions
    }

    pub(crate) fn empty_output_matrix(col_count: usize) -> LpMatrix<N, DualOf<R>> {
        LpMatrixBuilder::<N, DualOf<R>>::with_columns(col_count).build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dd::SinglePrecisionUmpire;
    use crate::dd::{Cone, ConeEngine, ConeOptions};
    use crate::matrix::LpMatrix;
    use hullabaloo::types::{Inequality, InequalityKind, RowId, RowSet};

    fn cone_for_shape(
        rows: usize,
        dimension: usize,
    ) -> ConeEngine<f64, Inequality, SinglePrecisionUmpire<f64, calculo::num::F64Em12Epsilon>> {
        let eps = calculo::num::F64Em12Epsilon;
        let matrix = LpMatrix::<f64, Inequality>::new(rows, dimension);
        let kinds = vec![InequalityKind::Inequality; rows];
        Cone::new(matrix, kinds, ConeOptions::default())
            .expect("build cone")
            .into_basis_prep(eps)
            .into_state()
    }

    #[test]
    fn row_positions_account_for_equality_expansion() {
        let equality_kinds = vec![InequalityKind::Equality, InequalityKind::Inequality];
        let row_positions =
            PolyhedronOutput::<f64, Inequality>::expanded_row_positions(&equality_kinds, 2);
        assert_eq!(row_positions, vec![0, 2]);

        let canonical_rows = equality_kinds.iter().fold(0usize, |acc, kind| {
            acc + if *kind == InequalityKind::Equality {
                2
            } else {
                1
            }
        });
        let lifting = row_lifting(&row_positions, canonical_rows);

        let mut zero_set = RowSet::new(canonical_rows);
        zero_set.insert(RowId::new(2));
        let rays = vec![OutputRayData {
            ray_id: RayId(0),
            vector: vec![1.0, 0.0],
            zero_set,
            is_linearity: false,
            near_zero_rows: Vec::new(),
        }];

        let cone = Box::new(cone_for_shape(canonical_rows, 2));
        let incidence = build::build_incidence(&cone, &rays, 2, &lifting).expect("build incidence");
        assert_eq!(incidence.family_size(), 1);

        let expected_zero_set = incidence.set(0).unwrap();
        assert!(expected_zero_set.contains(1usize));
        assert!(!expected_zero_set.contains(0usize));
    }

    #[cfg(test)]
    fn row_lifting(row_positions: &RowIndex, canonical_rows: usize) -> Vec<Vec<usize>> {
        let mut lifting = vec![Vec::new(); canonical_rows];
        for (orig, mapped) in row_positions.iter().enumerate() {
            if *mapped < 0 {
                let rep = (-*mapped - 1) as usize;
                lifting[rep].push(orig);
                continue;
            }
            let mapped_idx = *mapped as usize;
            lifting[mapped_idx].push(orig);
        }
        lifting
    }
}
