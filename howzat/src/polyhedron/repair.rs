use super::{
    PartialResolveResult, PolyhedronOutput, PreparedPartialResolveMinimal,
    PreparedPartialResolveResult, build_adjacency,
};
use crate::matrix::LpMatrix;
use ahash::AHashMap;
use calculo::linalg;
use calculo::num::{Epsilon, Int, Num, Rat, Sign};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{ComputationStatus, Generator, Inequality, Row, RowSet};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::VecDeque;

#[derive(Clone, Copy)]
enum CandidateVertices<'a> {
    Minimizers(&'a [Row]),
    Slice(&'a [Row]),
    FaceMinusRidge {
        face: &'a RowSet,
        ridge: &'a [Row],
        limit: usize,
    },
    FaceMinusRidgePlus {
        face: &'a RowSet,
        ridge: &'a [Row],
        extra: &'a [Row],
        limit: usize,
    },
}

type Key = SmallVec<[Row; 16]>;
type RepairEdges = Vec<(usize, usize)>;
type SimplicialRepairOutput = (SimplicialFrontierRepairReport, Option<RepairEdges>);
type GeneralRepairOutput = (GeneralFrontierRepairReport, Option<RepairEdges>);

#[derive(Clone, Debug)]
struct RidgeData {
    count: u32,
    facet0: usize,
    facet1: usize,
    witness: Row,
    basis: Key,
}

#[derive(Clone, Debug)]
pub struct SimplicialFacet<N: Num> {
    vertices: Key,
    normal: Vec<N>,
}

impl<N: Num> SimplicialFacet<N> {
    pub fn vertices(&self) -> &[Row] {
        &self.vertices
    }

    pub fn normal(&self) -> &[N] {
        &self.normal
    }
}

#[derive(Clone, Debug)]
pub struct SimplicialFrontierRepairOptions {
    pub max_steps: usize,
    pub rebuild_polyhedron_output: bool,
}

impl Default for SimplicialFrontierRepairOptions {
    fn default() -> Self {
        Self {
            max_steps: 1_000_000,
            rebuild_polyhedron_output: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimplicialFrontierRepairReport {
    pub initial_facets: usize,
    pub final_facets: usize,
    pub ignored_non_simplicial_facets: usize,
    pub new_non_simplicial_facets: usize,
    pub remaining_frontier_ridges: usize,
    pub steps_attempted: usize,
    pub new_facets: usize,
    pub step_limit_reached: bool,
    pub diagnostics: SimplicialFrontierRepairDiagnostics,
}

#[derive(Clone, Debug, Default)]
pub struct SimplicialFrontierRepairDiagnostics {
    pub orientation_no_witness: usize,
    pub ridge_direction_not_1d: usize,
    pub no_blocking_vertex: usize,
    pub infeasible_new_facet: usize,
    pub tie_steps: usize,
    pub dedupe_hits: usize,
}

#[derive(Clone, Debug)]
pub struct SimplicialFrontierRepairResult<N: Num> {
    facets: Vec<SimplicialFacet<N>>,
    report: SimplicialFrontierRepairReport,
    rebuilt_polyhedron: Option<PolyhedronOutput<N, Generator>>,
}

impl<N: Num> SimplicialFrontierRepairResult<N> {
    pub fn facets(&self) -> &[SimplicialFacet<N>] {
        &self.facets
    }

    pub fn report(&self) -> &SimplicialFrontierRepairReport {
        &self.report
    }

    pub fn rebuilt_polyhedron(&self) -> Option<&PolyhedronOutput<N, Generator>> {
        self.rebuilt_polyhedron.as_ref()
    }
}

#[derive(Clone, Debug)]
pub enum SimplicialFrontierRepairError {
    MissingOutputIncidence,
    InputHasRays { count: usize },
    OutputHasLinearity { count: usize },
}

impl std::error::Error for SimplicialFrontierRepairError {}

impl std::fmt::Display for SimplicialFrontierRepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingOutputIncidence => write!(f, "missing output incidence"),
            Self::InputHasRays { count } => write!(f, "input has rays (count={count})"),
            Self::OutputHasLinearity { count } => write!(f, "output has linearity (count={count})"),
        }
    }
}

impl<N: Num> PartialResolveResult<N, Generator> {
    pub fn repair_simplicial_frontier(
        &self,
        options: SimplicialFrontierRepairOptions,
        eps: &impl Epsilon<N>,
    ) -> Result<SimplicialFrontierRepairResult<N>, SimplicialFrontierRepairError> {
        repair_poly_simplicial_frontier(self.polyhedron(), options, eps)
    }
}

fn repair_poly_simplicial_frontier<N: Num>(
    poly: &PolyhedronOutput<N, Generator>,
    options: SimplicialFrontierRepairOptions,
    eps: &impl Epsilon<N>,
) -> Result<SimplicialFrontierRepairResult<N>, SimplicialFrontierRepairError> {
    let Some(incidence) = poly.incidence() else {
        return Err(SimplicialFrontierRepairError::MissingOutputIncidence);
    };

    let output_linearity = poly.output().linearity();
    if !output_linearity.is_empty() {
        return Err(SimplicialFrontierRepairError::OutputHasLinearity {
            count: output_linearity.cardinality(),
        });
    }

    let ray_rows = count_input_rays(poly.input(), eps);
    if ray_rows > 0 {
        return Err(SimplicialFrontierRepairError::InputHasRays { count: ray_rows });
    }

    let dimension = poly.dimension();
    let input_rows = poly.input().row_count();
    let cols = poly.input().col_count();

    let mut redund_cols: Vec<usize> = Vec::new();
    if poly.column_mapping().len() == cols {
        for (col, map) in poly.column_mapping().iter().enumerate().skip(1) {
            if map.is_none() {
                redund_cols.push(col);
            }
        }
    }
    let mut redund_mask = vec![false; cols];
    for &col in &redund_cols {
        redund_mask[col] = true;
    }

    let mut facets: Vec<Key> = Vec::new();
    let mut normals: Vec<Vec<N>> = Vec::new();
    let mut key_to_idx: AHashMap<Key, usize> = AHashMap::new();

    let mut ignored_non_simplicial_facets = 0usize;

    for out_row in 0..poly.output().row_count() {
        if poly.output().linearity().contains(out_row) {
            continue;
        }

        let Some(face) = incidence.set(out_row) else {
            continue;
        };
        let mut key = Key::new();
        for v in face.iter().map(|v| v.as_index()) {
            if v < input_rows {
                key.push(v);
            }
        }

        if key.len() != dimension {
            ignored_non_simplicial_facets += 1;
            continue;
        }

        if key_to_idx.contains_key(&key) {
            continue;
        }

        let idx = facets.len();
        key_to_idx.insert(key.clone(), idx);
        facets.push(key);
        normals.push(
            poly.output()
                .row(out_row)
                .expect("output row out of bounds")
                .to_vec(),
        );
    }

    let initial_facets = facets.len();
    let mut diagnostics = SimplicialFrontierRepairDiagnostics::default();
    let mut new_non_simplicial_facets = 0usize;

    let mut ridge_map: AHashMap<Key, RidgeData> = AHashMap::new();

    for (facet_idx, facet) in facets.iter().enumerate() {
        add_simplicial_facet_ridges(&mut ridge_map, None, facet, facet_idx);
    }

    let mut frontier: VecDeque<Key> = VecDeque::new();
    for ridge in ridge_map
        .iter()
        .filter(|(_ridge, data)| data.count == 1)
        .map(|(ridge, _data)| ridge.clone())
    {
        frontier.push_back(ridge);
    }

    let mut steps_attempted = 0usize;
    let mut new_facets = 0usize;
    while steps_attempted < options.max_steps {
        let Some(ridge) = frontier.pop_front() else {
            break;
        };

        let Some(ridge_data) = ridge_map.get(&ridge) else {
            continue;
        };
        if ridge_data.count != 1 {
            continue;
        }
        let facet_idx = ridge_data.facet0;

        steps_attempted += 1;

        let facet_normal = &normals[facet_idx];
        let step = match recover_facet_across_ridge(
            poly.input(),
            facet_normal,
            &facets[facet_idx],
            &ridge,
            &ridge,
            None,
            &redund_cols,
            &redund_mask,
            eps,
        ) {
            Ok(step) => step,
            Err(RepairStepFailure::RidgeDirectionNotOneDim) => {
                diagnostics.ridge_direction_not_1d += 1;
                continue;
            }
            Err(RepairStepFailure::OrientationNoWitness) => {
                diagnostics.orientation_no_witness += 1;
                continue;
            }
            Err(RepairStepFailure::NoBlockingVertex) => {
                diagnostics.no_blocking_vertex += 1;
                continue;
            }
            Err(RepairStepFailure::InfeasibleNewFacet) => {
                diagnostics.infeasible_new_facet += 1;
                continue;
            }
        };

        if step.minimizers > 1 {
            diagnostics.tie_steps += 1;
        }

        if step.vertices.len() != dimension {
            new_non_simplicial_facets += 1;
            continue;
        }

        let mut key = Key::new();
        key.extend(step.vertices.iter().copied());

        if key_to_idx.contains_key(&key) {
            diagnostics.dedupe_hits += 1;
            continue;
        }

        let idx = facets.len();
        key_to_idx.insert(key.clone(), idx);
        facets.push(key.clone());
        normals.push(step.normal);
        new_facets += 1;
        add_simplicial_facet_ridges(&mut ridge_map, Some(&mut frontier), &key, idx);
    }

    let mut repaired_facets: Vec<SimplicialFacet<N>> = Vec::with_capacity(facets.len());
    for (key, normal) in facets.into_iter().zip(normals.into_iter()) {
        repaired_facets.push(SimplicialFacet {
            vertices: key,
            normal,
        });
    }
    let final_facets = repaired_facets.len();

    let remaining_frontier_ridges = ridge_map.values().filter(|ridge| ridge.count == 1).count();
    let step_limit_reached = steps_attempted >= options.max_steps && !frontier.is_empty();

    let rebuilt_status = if remaining_frontier_ridges == 0 && !step_limit_reached {
        ComputationStatus::AllFound
    } else {
        ComputationStatus::InProgress
    };
    let rebuilt_polyhedron = options.rebuild_polyhedron_output.then(|| {
        rebuild_polyhedron_output_from_facets(poly, &repaired_facets, rebuilt_status, None, false)
    });

    Ok(SimplicialFrontierRepairResult {
        facets: repaired_facets,
        report: SimplicialFrontierRepairReport {
            initial_facets,
            final_facets,
            ignored_non_simplicial_facets,
            new_non_simplicial_facets,
            remaining_frontier_ridges,
            steps_attempted,
            new_facets,
            step_limit_reached,
            diagnostics,
        },
        rebuilt_polyhedron,
    })
}

fn add_simplicial_facet_ridges(
    ridge_map: &mut AHashMap<Key, RidgeData>,
    mut frontier: Option<&mut VecDeque<Key>>,
    facet: &[Row],
    facet_idx: usize,
) {
    if facet.is_empty() {
        return;
    }

    for drop_pos in 0..facet.len() {
        let dropped_vertex = facet[drop_pos];
        let mut ridge = Key::with_capacity(facet.len().saturating_sub(1));
        ridge.extend(facet[..drop_pos].iter().copied());
        ridge.extend(facet[(drop_pos + 1)..].iter().copied());

        bump_ridge(
            ridge_map,
            frontier.as_deref_mut(),
            ridge.clone(),
            facet_idx,
            dropped_vertex,
            ridge,
        );
    }
}

fn bump_ridge(
    ridge_map: &mut AHashMap<Key, RidgeData>,
    frontier: Option<&mut VecDeque<Key>>,
    ridge: Key,
    facet_idx: usize,
    witness: Row,
    basis: Key,
) {
    match ridge_map.entry(ridge) {
        std::collections::hash_map::Entry::Occupied(mut entry) => {
            let data = entry.get_mut();
            if data.facet0 == facet_idx || data.facet1 == facet_idx {
                return;
            }
            data.count += 1;
            if data.count == 2 {
                data.facet1 = facet_idx;
            }
        }
        std::collections::hash_map::Entry::Vacant(entry) => {
            if let Some(frontier) = frontier {
                frontier.push_back(entry.key().clone());
            }
            entry.insert(RidgeData {
                count: 1,
                facet0: facet_idx,
                facet1: usize::MAX,
                witness,
                basis,
            });
        }
    }
}

pub(crate) fn simplicial_frontier_ridge_count(
    facets: &[Vec<Row>],
    facet_dimension: usize,
) -> usize {
    let mut ridge_map: AHashMap<Key, RidgeData> = AHashMap::new();

    for (facet_idx, facet) in facets.iter().enumerate() {
        if facet.len() != facet_dimension {
            continue;
        }
        for drop_pos in 0..facet.len() {
            let dropped_vertex = facet[drop_pos];
            let mut ridge = Key::with_capacity(facet.len().saturating_sub(1));
            ridge.extend(facet[..drop_pos].iter().copied());
            ridge.extend(facet[(drop_pos + 1)..].iter().copied());
            bump_ridge(
                &mut ridge_map,
                None,
                ridge.clone(),
                facet_idx,
                dropped_vertex,
                ridge,
            );
        }
    }

    ridge_map.values().filter(|data| data.count == 1).count()
}

pub(crate) fn simplicial_frontier_ridges_by_facet_index(
    facets: &[Vec<Row>],
    facet_dimension: usize,
) -> Vec<(Vec<Row>, (usize, Row))> {
    let mut ridge_map: AHashMap<Key, RidgeData> = AHashMap::new();

    for (facet_idx, facet) in facets.iter().enumerate() {
        if facet.len() != facet_dimension {
            continue;
        }
        for drop_pos in 0..facet.len() {
            let dropped_vertex = facet[drop_pos];
            let mut ridge = Key::with_capacity(facet.len().saturating_sub(1));
            ridge.extend(facet[..drop_pos].iter().copied());
            ridge.extend(facet[(drop_pos + 1)..].iter().copied());
            bump_ridge(
                &mut ridge_map,
                None,
                ridge.clone(),
                facet_idx,
                dropped_vertex,
                ridge,
            );
        }
    }

    ridge_map
        .into_iter()
        .filter_map(|(ridge, data)| {
            if data.count != 1 {
                return None;
            }
            Some((ridge.into_vec(), (data.facet0, data.witness)))
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct RepairedFacet<N: Num> {
    vertices: Vec<Row>,
    normal: Vec<N>,
}

impl<N: Num> RepairedFacet<N> {
    pub fn vertices(&self) -> &[Row] {
        &self.vertices
    }

    pub fn normal(&self) -> &[N] {
        &self.normal
    }
}

#[derive(Clone, Debug)]
pub struct FacetGraphRepairOptions {
    pub max_steps: usize,
    pub max_facets: usize,
    pub rebuild_polyhedron_output: bool,
    pub frontier: FrontierRepairMode,
    pub frontier_max_steps: usize,
}

impl Default for FacetGraphRepairOptions {
    fn default() -> Self {
        Self {
            max_steps: 1_000_000,
            max_facets: usize::MAX,
            rebuild_polyhedron_output: false,
            frontier: FrontierRepairMode::Off,
            frontier_max_steps: 1_000_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierRepairMode {
    Off,
    Simplicial,
    General,
}

#[derive(Clone, Debug)]
pub enum FrontierRepairReport {
    Simplicial(SimplicialFrontierRepairReport),
    General(GeneralFrontierRepairReport),
}

impl FrontierRepairReport {
    pub fn remaining_frontier_ridges(&self) -> usize {
        match self {
            Self::Simplicial(report) => report.remaining_frontier_ridges,
            Self::General(report) => report.remaining_frontier_ridges,
        }
    }

    pub fn step_limit_reached(&self) -> bool {
        match self {
            Self::Simplicial(report) => report.step_limit_reached,
            Self::General(report) => report.step_limit_reached,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GeneralFrontierRepairDiagnostics {
    pub ridge_direction_not_1d: usize,
    pub ridge_rejected_by_facet: usize,
    pub orientation_no_witness: usize,
    pub no_blocking_vertex: usize,
    pub infeasible_new_facet: usize,
    pub tie_steps: usize,
    pub dedupe_hits: usize,
}

#[derive(Clone, Debug)]
pub struct GeneralFrontierRepairReport {
    pub initial_facets: usize,
    pub final_facets: usize,
    pub remaining_frontier_ridges: usize,
    pub steps_attempted: usize,
    pub new_facets: usize,
    pub step_limit_reached: bool,
    pub diagnostics: GeneralFrontierRepairDiagnostics,
}

#[derive(Clone, Debug, Default)]
pub struct FacetGraphRepairDiagnostics {
    pub kept_output_row_oob: usize,
    pub missing_inexact_incidence_set: usize,
    pub missing_inexact_adjacency_set: usize,
    pub dd_edge_hints_missing: usize,
    pub dd_edge_hint_missing: usize,
    pub dd_edge_hint_invalid: usize,
    pub dd_edge_hint_used_minimizers: usize,
    pub dd_edge_hint_used_drop_candidates: usize,
    pub dd_edge_hint_used_full_candidates: usize,
    pub ridge_basis_rank_too_small: usize,
    pub ridge_direction_not_1d: usize,
    pub orientation_no_witness: usize,
    pub ratio_no_blocking_vertex: usize,
    pub infeasible_new_facet: usize,
    pub tie_steps: usize,
    pub dedupe_hits: usize,
    pub new_facets: usize,
}

#[derive(Clone, Debug)]
pub struct FacetGraphRepairReport {
    pub initial_known_nodes: usize,
    pub initial_facets: usize,
    pub final_known_nodes: usize,
    pub final_facets: usize,
    pub unresolved_nodes: usize,
    pub steps_attempted: usize,
    pub diagnostics: FacetGraphRepairDiagnostics,
    pub frontier: Option<FrontierRepairReport>,
}

#[derive(Clone, Debug)]
pub struct FacetGraphRepairResult<N: Num> {
    facets: Vec<RepairedFacet<N>>,
    report: FacetGraphRepairReport,
    rebuilt_polyhedron: Option<PolyhedronOutput<N, Generator>>,
}

impl<N: Num> FacetGraphRepairResult<N> {
    pub fn facets(&self) -> &[RepairedFacet<N>] {
        &self.facets
    }

    pub fn report(&self) -> &FacetGraphRepairReport {
        &self.report
    }

    pub fn rebuilt_polyhedron(&self) -> Option<&PolyhedronOutput<N, Generator>> {
        self.rebuilt_polyhedron.as_ref()
    }
}

#[derive(Clone, Debug)]
pub enum FacetGraphRepairError {
    MissingExactOutputIncidence,
    MissingInexactOutputIncidence,
    MissingInexactOutputAdjacency,
    InputShapeMismatch,
    OutputShapeMismatch,
    InputHasRays { count: usize },
    ExactOutputHasLinearity { count: usize },
    InexactOutputHasLinearity { count: usize },
    ColumnMappingMismatch,
}

impl std::error::Error for FacetGraphRepairError {}

impl std::fmt::Display for FacetGraphRepairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingExactOutputIncidence => write!(f, "missing exact output incidence"),
            Self::MissingInexactOutputIncidence => write!(f, "missing inexact output incidence"),
            Self::MissingInexactOutputAdjacency => write!(f, "missing inexact output adjacency"),
            Self::InputShapeMismatch => write!(f, "input shape mismatch"),
            Self::OutputShapeMismatch => write!(f, "output shape mismatch"),
            Self::InputHasRays { count } => write!(f, "input has rays (count={count})"),
            Self::ExactOutputHasLinearity { count } => {
                write!(f, "exact output has linearity (count={count})")
            }
            Self::InexactOutputHasLinearity { count } => {
                write!(f, "inexact output has linearity (count={count})")
            }
            Self::ColumnMappingMismatch => write!(f, "column mapping mismatch"),
        }
    }
}

impl<M: Rat> PartialResolveResult<M, Generator> {
    pub fn repair_facet_graph<N: Num>(
        &self,
        inexact: &PolyhedronOutput<N, Generator>,
        options: FacetGraphRepairOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<FacetGraphRepairResult<M>, FacetGraphRepairError> {
        repair_facet_graph_from_inexact(self, inexact, options, eps)
    }
}

impl<M: Rat> PreparedPartialResolveResult<M, Generator> {
    pub fn repair_facet_graph<N: Num>(
        &self,
        inexact: &PolyhedronOutput<N, Generator>,
        options: FacetGraphRepairOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<FacetGraphRepairResult<M>, FacetGraphRepairError> {
        repair_facet_graph_from_inexact_prepared(self, inexact, options, eps)
    }
}

impl<M: Rat> PreparedPartialResolveMinimal<M, Generator> {
    pub fn repair_facet_graph<N: Num>(
        &self,
        inexact: &PolyhedronOutput<N, Generator>,
        options: FacetGraphRepairOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<FacetGraphRepairResult<M>, FacetGraphRepairError> {
        repair_facet_graph_from_inexact_prepared_minimal(self, inexact, options, eps)
    }
}

#[derive(Clone, Debug)]
struct FacetRecord<'a, Z: Int> {
    vertices: Vec<Row>,
    normal: Cow<'a, [Z]>,
}

#[derive(Clone, Debug)]
struct RatioTestResult<N: Num> {
    best_num: N,
    best_den: N,
    minimizer_vertices: SmallVec<[Row; 8]>,
}

fn hash_key(vertices: &[Row]) -> u64 {
    let mut h = 0xDEADBEEF_u64;
    for &v in vertices {
        let mut x = v as u64;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^= x >> 33;
        h ^= x;
        h = h.wrapping_mul(0x9e3779b97f4a7c15);
    }
    h
}

#[derive(Default)]
struct FacetGraphRecoverScratch {
    ridge_candidates: Vec<Row>,
    hint_candidates: SmallVec<[Row; 8]>,
}

fn collect_intersection_sorted_vec_rowset_with_extra_sorted(
    vertices: &[Row],
    other: &RowSet,
    extra_sorted: &[Row],
    limit: usize,
    out: &mut Vec<Row>,
) {
    out.clear();
    for &v in vertices {
        if v >= limit {
            continue;
        }
        if other.contains(v) || extra_sorted.binary_search(&v).is_ok() {
            out.push(v);
        }
    }
}

fn find_dd_edge_hint(
    edge_hints: Option<&[Vec<super::DdEdgeHint>]>,
    to: usize,
    neighbor: usize,
) -> Option<&super::DdEdgeHint> {
    let edge_hints = edge_hints?;
    let hints = edge_hints.get(to)?;
    let idx = hints
        .binary_search_by(|hint| hint.neighbor.cmp(&neighbor))
        .ok()?;
    hints.get(idx)
}

fn int_row_to_rat_row<M: Rat>(row: &[<M as Rat>::Int]) -> Vec<M> {
    let denom = <M as Rat>::Int::one();
    row.iter()
        .cloned()
        .map(|numer| M::from_frac(numer, denom.clone()))
        .collect()
}

fn reduce_gcd_int_vec<Z: Int>(values: &mut [Z]) -> Option<()> {
    let z1 = Z::one();
    let mut gcd: Option<Z> = None;
    for value in values.iter() {
        if value.is_zero() {
            continue;
        }
        let abs = value.abs().ok()?;
        match gcd.as_mut() {
            None => gcd = Some(abs),
            Some(g) => g.gcd_assign(&abs).ok()?,
        }
    }
    let Some(gcd) = gcd else {
        return Some(());
    };
    if gcd == z1 {
        return Some(());
    }
    for value in values.iter_mut() {
        if value.is_zero() {
            continue;
        }
        value.div_assign_exact(&gcd).ok()?;
    }
    Some(())
}

#[derive(Clone, Debug)]
struct RatioTestResultInt<Z: Int> {
    best_num: Z,
    best_den: Z,
    minimizer_vertices: SmallVec<[Row; 8]>,
}

#[derive(Clone, Debug)]
struct RecoveredFacetInt<Z: Int> {
    vertices: Vec<Row>,
    normal: Vec<Z>,
    minimizers: usize,
}

struct IntRepairCtx<'a, Z: Int> {
    input_rows: &'a super::IntRowMatrix<Z>,
    dot_tmp: Z,
    cmp_scratch: Z::CmpScratch,
    bareiss_scratch: super::BareissSolveScratch<Z>,
}

impl<'a, Z: Int> IntRepairCtx<'a, Z> {
    fn new(input_rows: &'a super::IntRowMatrix<Z>) -> Option<Self> {
        let cols = input_rows.col_count();
        let k = cols.checked_sub(1).filter(|_| cols > 0)?;
        Some(Self {
            input_rows,
            dot_tmp: Z::zero(),
            cmp_scratch: Z::CmpScratch::default(),
            bareiss_scratch: super::BareissSolveScratch::<Z>::new(k),
        })
    }

    fn row_count(&self) -> usize {
        self.input_rows.row_count()
    }

    fn col_count(&self) -> usize {
        self.input_rows.col_count()
    }

    fn dot_into(&mut self, row: Row, vec: &[Z], out: &mut Z) -> Option<()> {
        let src = self.input_rows.row(row)?;
        if src.len() != vec.len() {
            return None;
        }

        let z0 = Z::zero();
        Z::assign_from(out, &z0);
        for (a, b) in src.iter().zip(vec.iter()) {
            Z::assign_from(&mut self.dot_tmp, a);
            self.dot_tmp.mul_assign(b).ok()?;
            *out += &self.dot_tmp;
        }
        Some(())
    }
}

fn ratio_test_on_vertices_int<Z, I>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facet_normal: &[Z],
    direction: &[Z],
    vertices: I,
) -> Option<RatioTestResultInt<Z>>
where
    Z: Int,
    I: IntoIterator<Item = Row>,
{
    let mut has_best = false;
    let mut best_num = Z::zero();
    let mut best_den = Z::zero();
    let mut minimizer_vertices: SmallVec<[Row; 8]> = SmallVec::new();
    let mut num = Z::zero();
    let mut dir_eval = Z::zero();
    let mut den = Z::zero();

    for v in vertices {
        if v >= ctx.row_count() {
            continue;
        }

        ctx.dot_into(v, direction, &mut dir_eval)?;
        if !dir_eval.is_negative() {
            continue;
        }
        ctx.dot_into(v, facet_normal, &mut num)?;
        if !num.is_positive() {
            continue;
        }

        Z::assign_from(&mut den, &dir_eval);
        den.neg_mut().ok()?;

        if !has_best {
            has_best = true;
            Z::assign_from(&mut best_num, &num);
            Z::assign_from(&mut best_den, &den);
            minimizer_vertices.clear();
            minimizer_vertices.push(v);
            continue;
        }

        match Z::cmp_product(&num, &best_den, &best_num, &den, &mut ctx.cmp_scratch) {
            Ordering::Less => {
                Z::assign_from(&mut best_num, &num);
                Z::assign_from(&mut best_den, &den);
                minimizer_vertices.clear();
                minimizer_vertices.push(v);
            }
            Ordering::Equal => {
                minimizer_vertices.push(v);
            }
            Ordering::Greater => {}
        }
    }

    has_best.then_some(RatioTestResultInt {
        best_num,
        best_den,
        minimizer_vertices,
    })
}

fn ratio_test_on_face_minus_ridge_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facet_normal: &[Z],
    direction: &[Z],
    face: &RowSet,
    ridge_vertices: &[Row],
    limit: usize,
) -> Option<RatioTestResultInt<Z>>
where
    Z: Int,
{
    let mut ridge_pos = 0usize;

    let mut has_best = false;
    let mut best_num = Z::zero();
    let mut best_den = Z::zero();
    let mut minimizer_vertices: SmallVec<[Row; 8]> = SmallVec::new();
    let mut num = Z::zero();
    let mut dir_eval = Z::zero();
    let mut den = Z::zero();

    for v in face.iter().map(|v| v.as_index()) {
        if v >= limit {
            continue;
        }
        while ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] < v {
            ridge_pos += 1;
        }
        if ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] == v {
            continue;
        }

        ctx.dot_into(v, direction, &mut dir_eval)?;
        if !dir_eval.is_negative() {
            continue;
        }
        ctx.dot_into(v, facet_normal, &mut num)?;
        if !num.is_positive() {
            continue;
        }

        Z::assign_from(&mut den, &dir_eval);
        den.neg_mut().ok()?;

        if !has_best {
            has_best = true;
            Z::assign_from(&mut best_num, &num);
            Z::assign_from(&mut best_den, &den);
            minimizer_vertices.clear();
            minimizer_vertices.push(v);
            continue;
        }

        match Z::cmp_product(&num, &best_den, &best_num, &den, &mut ctx.cmp_scratch) {
            Ordering::Less => {
                Z::assign_from(&mut best_num, &num);
                Z::assign_from(&mut best_den, &den);
                minimizer_vertices.clear();
                minimizer_vertices.push(v);
            }
            Ordering::Equal => minimizer_vertices.push(v),
            Ordering::Greater => {}
        }
    }

    has_best.then_some(RatioTestResultInt {
        best_num,
        best_den,
        minimizer_vertices,
    })
}

fn ratio_test_on_face_minus_ridge_plus_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facet_normal: &[Z],
    direction: &[Z],
    face: &RowSet,
    ridge_vertices: &[Row],
    extra_vertices: &[Row],
    limit: usize,
) -> Option<RatioTestResultInt<Z>>
where
    Z: Int,
{
    let mut ridge_pos = 0usize;

    let mut has_best = false;
    let mut best_num = Z::zero();
    let mut best_den = Z::zero();
    let mut minimizer_vertices: SmallVec<[Row; 8]> = SmallVec::new();
    let mut num = Z::zero();
    let mut dir_eval = Z::zero();
    let mut den = Z::zero();

    for v in face.iter().map(|v| v.as_index()) {
        if v >= limit {
            continue;
        }
        while ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] < v {
            ridge_pos += 1;
        }
        if ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] == v {
            continue;
        }

        ctx.dot_into(v, direction, &mut dir_eval)?;
        if !dir_eval.is_negative() {
            continue;
        }
        ctx.dot_into(v, facet_normal, &mut num)?;
        if !num.is_positive() {
            continue;
        }

        Z::assign_from(&mut den, &dir_eval);
        den.neg_mut().ok()?;

        if !has_best {
            has_best = true;
            Z::assign_from(&mut best_num, &num);
            Z::assign_from(&mut best_den, &den);
            minimizer_vertices.clear();
            minimizer_vertices.push(v);
            continue;
        }

        match Z::cmp_product(&num, &best_den, &best_num, &den, &mut ctx.cmp_scratch) {
            Ordering::Less => {
                Z::assign_from(&mut best_num, &num);
                Z::assign_from(&mut best_den, &den);
                minimizer_vertices.clear();
                minimizer_vertices.push(v);
            }
            Ordering::Equal => minimizer_vertices.push(v),
            Ordering::Greater => {}
        }
    }

    for &v in extra_vertices {
        if v >= limit {
            continue;
        }
        if face.contains(v) {
            continue;
        }
        if ridge_vertices.binary_search(&v).is_ok() {
            continue;
        }

        ctx.dot_into(v, direction, &mut dir_eval)?;
        if !dir_eval.is_negative() {
            continue;
        }
        ctx.dot_into(v, facet_normal, &mut num)?;
        if !num.is_positive() {
            continue;
        }

        Z::assign_from(&mut den, &dir_eval);
        den.neg_mut().ok()?;

        if !has_best {
            has_best = true;
            Z::assign_from(&mut best_num, &num);
            Z::assign_from(&mut best_den, &den);
            minimizer_vertices.clear();
            minimizer_vertices.push(v);
            continue;
        }

        match Z::cmp_product(&num, &best_den, &best_num, &den, &mut ctx.cmp_scratch) {
            Ordering::Less => {
                Z::assign_from(&mut best_num, &num);
                Z::assign_from(&mut best_den, &den);
                minimizer_vertices.clear();
                minimizer_vertices.push(v);
            }
            Ordering::Equal => minimizer_vertices.push(v),
            Ordering::Greater => {}
        }
    }

    has_best.then_some(RatioTestResultInt {
        best_num,
        best_den,
        minimizer_vertices,
    })
}

fn supporting_incidence_key_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    normal: &mut [Z],
) -> (Vec<Row>, bool)
where
    Z: Int,
{
    let mut vertices: Vec<Row> = Vec::new();
    let mut saw_neg = false;
    let mut saw_pos = false;

    let mut eval = Z::zero();
    for v in 0..ctx.row_count() {
        ctx.dot_into(v, normal, &mut eval)
            .expect("dot product must succeed");
        if eval.is_negative() {
            saw_neg = true;
        } else if eval.is_positive() {
            saw_pos = true;
        } else {
            vertices.push(v);
        }
        if saw_neg && saw_pos {
            return (vertices, false);
        }
    }

    if saw_neg {
        for coeff in normal.iter_mut() {
            coeff.neg_mut().expect("integer negation must succeed");
        }
    }

    (vertices, true)
}

fn recovered_facet_from_ratio_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facet_normal: &[Z],
    direction: &[Z],
    ratio: RatioTestResultInt<Z>,
) -> Result<RecoveredFacetInt<Z>, RepairStepFailure>
where
    Z: Int,
{
    let cols = facet_normal.len();
    let mut new_normal: Vec<Z> = Vec::with_capacity(cols);
    for (n_i, d_i) in facet_normal.iter().zip(direction.iter()) {
        let mut acc = n_i.clone();
        acc.mul_assign(&ratio.best_den)
            .expect("int mul must succeed");
        let mut term = d_i.clone();
        term.mul_assign(&ratio.best_num)
            .expect("int mul must succeed");
        acc += &term;
        new_normal.push(acc);
    }
    reduce_gcd_int_vec(&mut new_normal).expect("gcd reduction must succeed");

    let (vertices, supporting_ok) = supporting_incidence_key_int(ctx, &mut new_normal);
    if !supporting_ok {
        return Err(RepairStepFailure::InfeasibleNewFacet);
    }

    Ok(RecoveredFacetInt {
        vertices,
        normal: new_normal,
        minimizers: ratio.minimizer_vertices.len(),
    })
}

fn recovered_facet_from_global_ratio_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facet_normal: &[Z],
    direction: &[Z],
    facet_vertices: &[Row],
    ratio: RatioTestResultInt<Z>,
    allow_global_scan: bool,
) -> Result<RecoveredFacetInt<Z>, RepairStepFailure>
where
    Z: Int,
{
    let cols = facet_normal.len();
    let mut new_normal: Vec<Z> = Vec::with_capacity(cols);
    for (n_i, d_i) in facet_normal.iter().zip(direction.iter()) {
        let mut acc = n_i.clone();
        acc.mul_assign(&ratio.best_den)
            .expect("int mul must succeed");
        let mut term = d_i.clone();
        term.mul_assign(&ratio.best_num)
            .expect("int mul must succeed");
        acc += &term;
        new_normal.push(acc);
    }
    reduce_gcd_int_vec(&mut new_normal).expect("gcd reduction must succeed");

    let mut ridge_vertices: Vec<Row> = Vec::new();
    let mut eval = Z::zero();
    for &v in facet_vertices {
        if v >= ctx.row_count() {
            continue;
        }
        ctx.dot_into(v, direction, &mut eval)
            .expect("dot product must succeed");
        if eval.is_zero() {
            ridge_vertices.push(v);
            continue;
        }
        if eval.is_negative() {
            if allow_global_scan {
                return recovered_facet_from_ratio_int(ctx, facet_normal, direction, ratio);
            }
            return Err(RepairStepFailure::InfeasibleNewFacet);
        }
    }

    let mut vertices: Vec<Row> = ridge_vertices;
    vertices.extend(ratio.minimizer_vertices.iter().copied());
    vertices.sort_unstable();
    vertices.dedup();

    Ok(RecoveredFacetInt {
        vertices,
        normal: new_normal,
        minimizers: ratio.minimizer_vertices.len(),
    })
}

fn orient_direction_away_from_facet_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    direction: &mut [Z],
    facet_vertices: &[Row],
    ridge_vertices: &[Row],
) -> Option<Row>
where
    Z: Int,
{
    let mut ridge_pos = 0usize;
    let mut eval = Z::zero();

    for &v in facet_vertices {
        while ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] < v {
            ridge_pos += 1;
        }
        if ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] == v {
            continue;
        }

        ctx.dot_into(v, direction, &mut eval)?;
        if eval.is_zero() {
            continue;
        }
        if eval.is_negative() {
            for coeff in direction.iter_mut() {
                coeff.neg_mut().ok()?;
            }
        }
        return Some(v);
    }
    None
}

fn orient_direction_away_from_facet_int_with_witness<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    direction: &mut [Z],
    witness: Row,
) -> Option<Row>
where
    Z: Int,
{
    let mut eval = Z::zero();
    ctx.dot_into(witness, direction, &mut eval)?;
    if eval.is_zero() {
        return None;
    }
    if eval.is_negative() {
        for coeff in direction.iter_mut() {
            coeff.neg_mut().ok()?;
        }
    }
    Some(witness)
}

#[allow(clippy::too_many_arguments)]
fn recover_facet_across_ridge_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facet_normal: &[Z],
    facet_vertices: &[Row],
    ridge_candidates: &[Row],
    ridge_basis: &[Row],
    candidate_vertices: Option<CandidateVertices<'_>>,
    orientation_witness: Option<Row>,
    redund_cols: &[usize],
    redund_mask: &[bool],
    allow_global_scan: bool,
) -> Result<RecoveredFacetInt<Z>, RepairStepFailure>
where
    Z: Int,
{
    let cols = ctx.col_count();
    if cols == 0 || facet_normal.len() != cols || redund_mask.len() != cols {
        return Err(RepairStepFailure::RidgeDirectionNotOneDim);
    }

    let mut unit_cols: SmallVec<[usize; 8]> = SmallVec::new();
    unit_cols.extend_from_slice(redund_cols);

    let mut direction = None;
    for pivot_col in 0..cols {
        if redund_mask[pivot_col] || facet_normal[pivot_col].is_zero() {
            continue;
        }
        unit_cols.push(pivot_col);
        let vec = super::solve_nullspace_1d_rows_with_unit_cols_bareiss_int(
            &mut ctx.bareiss_scratch,
            ctx.input_rows,
            ridge_basis,
            unit_cols.as_slice(),
            redund_mask,
        );
        unit_cols.pop();
        if vec.is_some() {
            direction = vec;
            break;
        }
    }
    let mut direction = direction.ok_or(RepairStepFailure::RidgeDirectionNotOneDim)?;
    reduce_gcd_int_vec(&mut direction).expect("gcd reduction must succeed");

    let mut oriented = None;
    if let Some(witness) = orientation_witness
        && witness < ctx.row_count()
        && facet_vertices.binary_search(&witness).is_ok()
        && ridge_candidates.binary_search(&witness).is_err()
    {
        oriented = orient_direction_away_from_facet_int_with_witness(ctx, &mut direction, witness);
    }
    if oriented.is_none() {
        oriented = orient_direction_away_from_facet_int(
            ctx,
            &mut direction,
            facet_vertices,
            ridge_candidates,
        );
    }
    oriented.ok_or(RepairStepFailure::OrientationNoWitness)?;

    if let Some(candidates) = candidate_vertices {
        let ratio = match candidates {
            CandidateVertices::Minimizers(vertices) | CandidateVertices::Slice(vertices) => {
                ratio_test_on_vertices_int(ctx, facet_normal, &direction, vertices.iter().copied())
            }
            CandidateVertices::FaceMinusRidge { face, ridge, limit } => {
                ratio_test_on_face_minus_ridge_int(
                    ctx,
                    facet_normal,
                    &direction,
                    face,
                    ridge,
                    limit,
                )
            }
            CandidateVertices::FaceMinusRidgePlus {
                face,
                ridge,
                extra,
                limit,
            } => ratio_test_on_face_minus_ridge_plus_int(
                ctx,
                facet_normal,
                &direction,
                face,
                ridge,
                extra,
                limit,
            ),
        };

        if let Some(ratio) = ratio {
            match candidates {
                CandidateVertices::Minimizers(expected) => {
                    if ratio.minimizer_vertices.len() != expected.len()
                        || ratio
                            .minimizer_vertices
                            .iter()
                            .copied()
                            .zip(expected.iter().copied())
                            .any(|(a, b)| a != b)
                    {
                        // Minimizer hint does not match the observed tie set; fall back to the full scan path.
                    } else if let Ok(step) = recovered_facet_from_global_ratio_int(
                        ctx,
                        facet_normal,
                        &direction,
                        facet_vertices,
                        ratio,
                        allow_global_scan,
                    ) {
                        return Ok(step);
                    }
                }
                CandidateVertices::FaceMinusRidge { .. }
                | CandidateVertices::FaceMinusRidgePlus { .. } => {
                    if let Ok(step) = recovered_facet_from_global_ratio_int(
                        ctx,
                        facet_normal,
                        &direction,
                        facet_vertices,
                        ratio.clone(),
                        allow_global_scan,
                    ) {
                        return Ok(step);
                    }

                    if allow_global_scan
                        && let Ok(step) =
                            recovered_facet_from_ratio_int(ctx, facet_normal, &direction, ratio)
                    {
                        return Ok(step);
                    }
                }
                CandidateVertices::Slice(vertices) => {
                    if !allow_global_scan {
                        if ratio.minimizer_vertices.len() == vertices.len()
                            && ratio
                                .minimizer_vertices
                                .iter()
                                .copied()
                                .zip(vertices.iter().copied())
                                .all(|(a, b)| a == b)
                            && let Ok(step) = recovered_facet_from_global_ratio_int(
                                ctx,
                                facet_normal,
                                &direction,
                                facet_vertices,
                                ratio,
                                false,
                            )
                        {
                            return Ok(step);
                        }
                    } else if let Ok(step) =
                        recovered_facet_from_ratio_int(ctx, facet_normal, &direction, ratio)
                    {
                        return Ok(step);
                    }
                }
            }
        }
    }

    if !allow_global_scan {
        return Err(RepairStepFailure::NoBlockingVertex);
    }

    let ratio = ratio_test_on_vertices_int(ctx, facet_normal, &direction, 0..ctx.row_count())
        .ok_or(RepairStepFailure::NoBlockingVertex)?;
    recovered_facet_from_global_ratio_int(
        ctx,
        facet_normal,
        &direction,
        facet_vertices,
        ratio,
        allow_global_scan,
    )
}

#[allow(clippy::too_many_arguments)]
fn recover_facet_from_inexact_neighbor<M>(
    ctx: &mut IntRepairCtx<'_, <M as Rat>::Int>,
    facet_normal: &[<M as Rat>::Int],
    facet_vertices: &[Row],
    from_node: usize,
    to_node: usize,
    to_incidence: &RowSet,
    input_rows: usize,
    ridge_target_rank: usize,
    redund_cols: &[usize],
    redund_mask: &[bool],
    dd_hints: Option<&super::DdRepairHints>,
    scratch: &mut FacetGraphRecoverScratch,
    diagnostics: &mut FacetGraphRepairDiagnostics,
) -> Option<RecoveredFacetInt<<M as Rat>::Int>>
where
    M: Rat,
{
    let dd_edge_hints = dd_hints.and_then(|hints| hints.incoming_edge_hints.as_deref());
    let to_near_zero = dd_hints
        .map(|hints| hints.facet_near_zero_rows(to_node))
        .unwrap_or(&[]);

    let mut hint_ridge_basis: Option<&[Row]> = None;
    let mut hint_drop_candidates: Option<&[Row]> = None;
    let mut hint_minimizers: Option<&[Row]> = None;
    let mut hint_orientation_witness: Option<Row> = None;
    let mut hint_entered_row: Option<Row> = None;

    if let Some(hint) = find_dd_edge_hint(dd_edge_hints, to_node, from_node) {
        if hint.ridge_basis.len() == ridge_target_rank
            && hint.ridge_basis.iter().all(|&v| {
                v < input_rows
                    && to_incidence.contains(v)
                    && facet_vertices.binary_search(&v).is_ok()
            })
        {
            hint_ridge_basis = Some(hint.ridge_basis.as_slice());
            hint_drop_candidates = Some(hint.drop_candidates.as_slice());
            hint_orientation_witness = hint.from_witness.filter(|&v| {
                v < input_rows
                    && !to_incidence.contains(v)
                    && facet_vertices.binary_search(&v).is_ok()
            });
            hint_entered_row = hint
                .entered_row
                .filter(|&v| v < input_rows && facet_vertices.binary_search(&v).is_err());
            if hint.minimizers_complete
                && !hint.minimizers.is_empty()
                && hint.minimizers.iter().all(|&v| {
                    v < input_rows
                        && to_incidence.contains(v)
                        && facet_vertices.binary_search(&v).is_err()
                })
            {
                hint_minimizers = Some(hint.minimizers.as_slice());
            }
        } else {
            diagnostics.dd_edge_hint_invalid += 1;
        }
    } else if dd_edge_hints.is_some() {
        diagnostics.dd_edge_hint_missing += 1;
    } else {
        diagnostics.dd_edge_hints_missing += 1;
    }

    collect_intersection_sorted_vec_rowset_with_extra_sorted(
        facet_vertices,
        to_incidence,
        to_near_zero,
        input_rows,
        &mut scratch.ridge_candidates,
    );
    if scratch.ridge_candidates.len() < ridge_target_rank {
        diagnostics.ridge_basis_rank_too_small += 1;
        return None;
    }

    if let Some(ridge_basis) = hint_ridge_basis {
        if let Some(minimizers) = hint_minimizers
            && let Ok(step) = recover_facet_across_ridge_int(
                ctx,
                facet_normal,
                facet_vertices,
                &scratch.ridge_candidates,
                ridge_basis,
                Some(CandidateVertices::Minimizers(minimizers)),
                hint_orientation_witness,
                redund_cols,
                redund_mask,
                false,
            )
        {
            diagnostics.dd_edge_hint_used_minimizers += 1;
            return Some(step);
        }
        scratch.hint_candidates.clear();
        if let Some(v) = hint_entered_row
            && scratch.ridge_candidates.binary_search(&v).is_err()
            && scratch.hint_candidates.len() < 8
        {
            scratch.hint_candidates.push(v);
        }
        if let Some(drop_candidates) = hint_drop_candidates {
            for &v in drop_candidates {
                if scratch.hint_candidates.len() >= 8 {
                    break;
                }
                if v >= input_rows {
                    continue;
                }
                if !to_incidence.contains(v) {
                    continue;
                }
                if facet_vertices.binary_search(&v).is_ok() {
                    continue;
                }
                if scratch.ridge_candidates.binary_search(&v).is_ok() {
                    continue;
                }
                scratch.hint_candidates.push(v);
            }
        }
        scratch.hint_candidates.sort_unstable();
        scratch.hint_candidates.dedup();

        if !scratch.hint_candidates.is_empty()
            && let Ok(step) = recover_facet_across_ridge_int(
                ctx,
                facet_normal,
                facet_vertices,
                &scratch.ridge_candidates,
                ridge_basis,
                Some(CandidateVertices::Slice(scratch.hint_candidates.as_slice())),
                hint_orientation_witness,
                redund_cols,
                redund_mask,
                false,
            )
        {
            diagnostics.dd_edge_hint_used_drop_candidates += 1;
            return Some(step);
        }
    }
    if let Some(ridge_basis) = hint_ridge_basis
        && let Ok(step) = recover_facet_across_ridge_int(
            ctx,
            facet_normal,
            facet_vertices,
            &scratch.ridge_candidates,
            ridge_basis,
            Some(if to_near_zero.is_empty() {
                CandidateVertices::FaceMinusRidge {
                    face: to_incidence,
                    ridge: &scratch.ridge_candidates,
                    limit: input_rows,
                }
            } else {
                CandidateVertices::FaceMinusRidgePlus {
                    face: to_incidence,
                    ridge: &scratch.ridge_candidates,
                    extra: to_near_zero,
                    limit: input_rows,
                }
            }),
            hint_orientation_witness,
            redund_cols,
            redund_mask,
            false,
        )
    {
        diagnostics.dd_edge_hint_used_full_candidates += 1;
        return Some(step);
    }

    let Some(ridge_basis) = super::select_row_basis_rows_int(
        ctx.input_rows,
        &scratch.ridge_candidates,
        ridge_target_rank,
        redund_mask,
    ) else {
        diagnostics.ridge_basis_rank_too_small += 1;
        return None;
    };

    match recover_facet_across_ridge_int(
        ctx,
        facet_normal,
        facet_vertices,
        &scratch.ridge_candidates,
        &ridge_basis,
        Some(if to_near_zero.is_empty() {
            CandidateVertices::FaceMinusRidge {
                face: to_incidence,
                ridge: &scratch.ridge_candidates,
                limit: input_rows,
            }
        } else {
            CandidateVertices::FaceMinusRidgePlus {
                face: to_incidence,
                ridge: &scratch.ridge_candidates,
                extra: to_near_zero,
                limit: input_rows,
            }
        }),
        hint_orientation_witness,
        redund_cols,
        redund_mask,
        true,
    ) {
        Ok(step) => Some(step),
        Err(RepairStepFailure::RidgeDirectionNotOneDim) => {
            diagnostics.ridge_direction_not_1d += 1;
            None
        }
        Err(RepairStepFailure::OrientationNoWitness) => {
            diagnostics.orientation_no_witness += 1;
            None
        }
        Err(RepairStepFailure::NoBlockingVertex) => {
            diagnostics.ratio_no_blocking_vertex += 1;
            None
        }
        Err(RepairStepFailure::InfeasibleNewFacet) => {
            diagnostics.infeasible_new_facet += 1;
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn repair_simplicial_frontier_on_facet_records_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facets: &mut Vec<FacetRecord<'_, Z>>,
    buckets: &mut AHashMap<u64, SmallVec<[usize; 4]>>,
    dimension: usize,
    redund_cols: &[usize],
    redund_mask: &[bool],
    max_steps: usize,
    max_facets: usize,
) -> Option<SimplicialRepairOutput>
where
    Z: Int,
{
    let initial_facets = facets.len();
    if initial_facets == 0 || dimension == 0 {
        return None;
    }

    let ignored_non_simplicial_facets = facets
        .iter()
        .filter(|facet| facet.vertices.len() != dimension)
        .count();
    if ignored_non_simplicial_facets != 0 {
        return None;
    }

    let mut ridge_map: AHashMap<Key, RidgeData> =
        AHashMap::with_capacity(initial_facets.saturating_mul(dimension.saturating_add(1)));
    for (facet_idx, facet) in facets.iter().enumerate() {
        add_simplicial_facet_ridges(&mut ridge_map, None, &facet.vertices, facet_idx);
    }

    let mut frontier: VecDeque<Key> = VecDeque::new();
    for ridge in ridge_map
        .iter()
        .filter(|(_ridge, data)| data.count == 1)
        .map(|(ridge, _data)| ridge.clone())
    {
        frontier.push_back(ridge);
    }

    let mut diagnostics = SimplicialFrontierRepairDiagnostics::default();
    let mut new_non_simplicial_facets = 0usize;
    let mut steps_attempted = 0usize;
    let mut new_facets = 0usize;

    while steps_attempted < max_steps && facets.len() < max_facets {
        let Some(ridge) = frontier.pop_front() else {
            break;
        };

        let Some(ridge_data) = ridge_map.get(&ridge) else {
            continue;
        };
        if ridge_data.count != 1 {
            continue;
        }
        let facet_idx = ridge_data.facet0;

        steps_attempted += 1;

        let facet = facets.get(facet_idx)?;
        let step = match recover_facet_across_ridge_int(
            ctx,
            facet.normal.as_ref(),
            facet.vertices.as_slice(),
            ridge.as_slice(),
            ridge.as_slice(),
            None,
            Some(ridge_data.witness),
            redund_cols,
            redund_mask,
            true,
        ) {
            Ok(step) => step,
            Err(RepairStepFailure::RidgeDirectionNotOneDim) => {
                diagnostics.ridge_direction_not_1d += 1;
                continue;
            }
            Err(RepairStepFailure::OrientationNoWitness) => {
                diagnostics.orientation_no_witness += 1;
                continue;
            }
            Err(RepairStepFailure::NoBlockingVertex) => {
                diagnostics.no_blocking_vertex += 1;
                continue;
            }
            Err(RepairStepFailure::InfeasibleNewFacet) => {
                diagnostics.infeasible_new_facet += 1;
                continue;
            }
        };

        if step.minimizers > 1 {
            diagnostics.tie_steps += 1;
        }

        if step.vertices.len() != dimension {
            new_non_simplicial_facets += 1;
            continue;
        }

        let hash = hash_key(&step.vertices);
        let existing = buckets.get(&hash).and_then(|candidates| {
            candidates
                .iter()
                .copied()
                .find(|&id| facets[id].vertices == step.vertices)
        });
        if existing.is_some() {
            diagnostics.dedupe_hits += 1;
            continue;
        }

        let id = facets.len();
        buckets.entry(hash).or_default().push(id);
        facets.push(FacetRecord {
            vertices: step.vertices,
            normal: Cow::Owned(step.normal),
        });
        new_facets += 1;

        let vertices = facets.get(id).expect("new facet index must be valid");
        add_simplicial_facet_ridges(&mut ridge_map, Some(&mut frontier), &vertices.vertices, id);
    }

    let final_facets = facets.len();
    let remaining_frontier_ridges = ridge_map.values().filter(|ridge| ridge.count == 1).count();
    let step_limit_reached = (steps_attempted >= max_steps && !frontier.is_empty())
        || (facets.len() >= max_facets && remaining_frontier_ridges > 0);

    let edges = (remaining_frontier_ridges == 0 && !step_limit_reached).then(|| {
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(ridge_map.len());
        for ridge in ridge_map.values() {
            if ridge.count != 2 || ridge.facet1 == usize::MAX {
                continue;
            }
            let (a, b) = if ridge.facet0 < ridge.facet1 {
                (ridge.facet0, ridge.facet1)
            } else {
                (ridge.facet1, ridge.facet0)
            };
            if a != b {
                edges.push((a, b));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    });

    Some((
        SimplicialFrontierRepairReport {
            initial_facets,
            final_facets,
            ignored_non_simplicial_facets,
            new_non_simplicial_facets,
            remaining_frontier_ridges,
            steps_attempted,
            new_facets,
            step_limit_reached,
            diagnostics,
        },
        edges,
    ))
}

struct GeneralFrontierScratch<Z: Int> {
    indices: SmallVec<[usize; 16]>,
    basis: Key,
    ridge: Key,
    eval: Z,
}

fn next_combination(indices: &mut [usize], n: usize) -> bool {
    let k = indices.len();
    if k == 0 || n < k {
        return false;
    }

    for i in (0..k).rev() {
        let max = i + n - k;
        if indices[i] == max {
            continue;
        }
        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
        return true;
    }

    false
}

fn ridge_direction_from_basis_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facet_normal: &[Z],
    ridge_basis: &[Row],
    redund_cols: &[usize],
    redund_mask: &[bool],
) -> Option<Vec<Z>>
where
    Z: Int,
{
    let cols = ctx.col_count();
    if cols == 0 || facet_normal.len() != cols || redund_mask.len() != cols {
        return None;
    }

    let mut unit_cols: SmallVec<[usize; 8]> = SmallVec::new();
    unit_cols.extend_from_slice(redund_cols);

    for pivot_col in 0..cols {
        if redund_mask[pivot_col] || facet_normal[pivot_col].is_zero() {
            continue;
        }
        unit_cols.push(pivot_col);
        let direction = super::solve_nullspace_1d_rows_with_unit_cols_bareiss_int(
            &mut ctx.bareiss_scratch,
            ctx.input_rows,
            ridge_basis,
            unit_cols.as_slice(),
            redund_mask,
        );
        unit_cols.pop();

        if let Some(mut direction) = direction {
            reduce_gcd_int_vec(&mut direction).expect("gcd reduction must succeed");
            return Some(direction);
        }
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn add_general_facet_ridges_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    ridge_map: &mut AHashMap<Key, RidgeData>,
    mut frontier: Option<&mut VecDeque<Key>>,
    facet_normal: &[Z],
    facet_vertices: &[Row],
    facet_idx: usize,
    ridge_target_rank: usize,
    redund_cols: &[usize],
    redund_mask: &[bool],
    scratch: &mut GeneralFrontierScratch<Z>,
    diagnostics: &mut GeneralFrontierRepairDiagnostics,
) -> Option<()>
where
    Z: Int,
{
    let basis_size = ridge_target_rank;
    if basis_size == 0 || facet_vertices.len() <= basis_size {
        return Some(());
    }

    scratch.indices.clear();
    scratch.indices.extend(0..basis_size);

    loop {
        scratch.basis.clear();
        for &idx in scratch.indices.iter() {
            let v = *facet_vertices.get(idx)?;
            scratch.basis.push(v);
        }

        let Some(direction) = ridge_direction_from_basis_int(
            ctx,
            facet_normal,
            scratch.basis.as_slice(),
            redund_cols,
            redund_mask,
        ) else {
            diagnostics.ridge_direction_not_1d += 1;
            if !next_combination(scratch.indices.as_mut_slice(), facet_vertices.len()) {
                break;
            }
            continue;
        };

        scratch.ridge.clear();
        let mut saw_nonzero = false;
        let mut flip = false;
        let mut witness: Option<Row> = None;
        let mut rejected = false;

        for &v in facet_vertices {
            ctx.dot_into(v, &direction, &mut scratch.eval)?;
            if scratch.eval.is_zero() {
                scratch.ridge.push(v);
                continue;
            }
            if !saw_nonzero {
                saw_nonzero = true;
                flip = scratch.eval.is_negative();
            }

            let is_neg = if flip {
                scratch.eval.is_positive()
            } else {
                scratch.eval.is_negative()
            };
            if is_neg {
                diagnostics.ridge_rejected_by_facet += 1;
                rejected = true;
                break;
            }
            if witness.is_none() {
                witness = Some(v);
            }
        }

        if !rejected {
            let Some(witness) = witness else {
                diagnostics.orientation_no_witness += 1;
                if !next_combination(scratch.indices.as_mut_slice(), facet_vertices.len()) {
                    break;
                }
                continue;
            };

            if scratch.ridge.len() >= basis_size {
                bump_ridge(
                    ridge_map,
                    frontier.as_deref_mut(),
                    scratch.ridge.clone(),
                    facet_idx,
                    witness,
                    scratch.basis.clone(),
                );
            }
        }

        if !next_combination(scratch.indices.as_mut_slice(), facet_vertices.len()) {
            break;
        }
    }

    Some(())
}

#[allow(clippy::too_many_arguments)]
fn repair_general_frontier_on_facet_records_int<Z>(
    ctx: &mut IntRepairCtx<'_, Z>,
    facets: &mut Vec<FacetRecord<'_, Z>>,
    buckets: &mut AHashMap<u64, SmallVec<[usize; 4]>>,
    dimension: usize,
    ridge_target_rank: usize,
    redund_cols: &[usize],
    redund_mask: &[bool],
    max_steps: usize,
    max_facets: usize,
) -> Option<GeneralRepairOutput>
where
    Z: Int,
{
    let initial_facets = facets.len();
    if initial_facets == 0 || dimension == 0 {
        return None;
    }

    let mut ridge_map: AHashMap<Key, RidgeData> =
        AHashMap::with_capacity(initial_facets.saturating_mul(dimension.saturating_add(1)));
    let mut scratch = GeneralFrontierScratch {
        indices: SmallVec::new(),
        basis: Key::new(),
        ridge: Key::new(),
        eval: Z::zero(),
    };

    let mut diagnostics = GeneralFrontierRepairDiagnostics::default();
    for (facet_idx, facet) in facets.iter().enumerate() {
        if facet.vertices.len() == dimension {
            add_simplicial_facet_ridges(&mut ridge_map, None, &facet.vertices, facet_idx);
            continue;
        }
        add_general_facet_ridges_int(
            ctx,
            &mut ridge_map,
            None,
            facet.normal.as_ref(),
            &facet.vertices,
            facet_idx,
            ridge_target_rank,
            redund_cols,
            redund_mask,
            &mut scratch,
            &mut diagnostics,
        )?;
    }

    let mut frontier: VecDeque<Key> = VecDeque::new();
    for ridge in ridge_map
        .iter()
        .filter(|(_ridge, data)| data.count == 1)
        .map(|(ridge, _data)| ridge.clone())
    {
        frontier.push_back(ridge);
    }

    let mut steps_attempted = 0usize;
    let mut new_facets = 0usize;

    while steps_attempted < max_steps && facets.len() < max_facets {
        let Some(ridge) = frontier.pop_front() else {
            break;
        };

        let Some(ridge_data) = ridge_map.get(&ridge) else {
            continue;
        };
        if ridge_data.count != 1 {
            continue;
        }

        let facet_idx = ridge_data.facet0;
        let facet = facets.get(facet_idx)?;

        steps_attempted += 1;

        let step = match recover_facet_across_ridge_int(
            ctx,
            facet.normal.as_ref(),
            facet.vertices.as_slice(),
            ridge.as_slice(),
            ridge_data.basis.as_slice(),
            None,
            Some(ridge_data.witness),
            redund_cols,
            redund_mask,
            true,
        ) {
            Ok(step) => step,
            Err(RepairStepFailure::RidgeDirectionNotOneDim) => {
                diagnostics.ridge_direction_not_1d += 1;
                continue;
            }
            Err(RepairStepFailure::OrientationNoWitness) => {
                diagnostics.orientation_no_witness += 1;
                continue;
            }
            Err(RepairStepFailure::NoBlockingVertex) => {
                diagnostics.no_blocking_vertex += 1;
                continue;
            }
            Err(RepairStepFailure::InfeasibleNewFacet) => {
                diagnostics.infeasible_new_facet += 1;
                continue;
            }
        };

        if step.minimizers > 1 {
            diagnostics.tie_steps += 1;
        }

        let hash = hash_key(&step.vertices);
        let existing = buckets.get(&hash).and_then(|candidates| {
            candidates
                .iter()
                .copied()
                .find(|&id| facets[id].vertices == step.vertices)
        });
        if let Some(existing) = existing {
            diagnostics.dedupe_hits += 1;
            if facet_idx != existing {
                let witness = ridge_data.witness;
                let basis = ridge_data.basis.clone();
                bump_ridge(
                    &mut ridge_map,
                    Some(&mut frontier),
                    ridge.clone(),
                    existing,
                    witness,
                    basis,
                );
            }
            continue;
        }

        let id = facets.len();
        buckets.entry(hash).or_default().push(id);
        facets.push(FacetRecord {
            vertices: step.vertices,
            normal: Cow::Owned(step.normal),
        });
        new_facets += 1;

        let facet = facets.get(id).expect("new facet index must be valid");
        if facet.vertices.len() == dimension {
            add_simplicial_facet_ridges(&mut ridge_map, Some(&mut frontier), &facet.vertices, id);
        } else {
            add_general_facet_ridges_int(
                ctx,
                &mut ridge_map,
                Some(&mut frontier),
                facet.normal.as_ref(),
                &facet.vertices,
                id,
                ridge_target_rank,
                redund_cols,
                redund_mask,
                &mut scratch,
                &mut diagnostics,
            )?;
        }
    }

    let remaining_frontier_ridges = ridge_map.values().filter(|ridge| ridge.count == 1).count();
    let step_limit_reached = (steps_attempted >= max_steps && !frontier.is_empty())
        || (facets.len() >= max_facets && remaining_frontier_ridges > 0);

    let edges = (remaining_frontier_ridges == 0 && !step_limit_reached).then(|| {
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(ridge_map.len());
        for ridge in ridge_map.values() {
            if ridge.count != 2 || ridge.facet1 == usize::MAX {
                continue;
            }
            let (a, b) = if ridge.facet0 < ridge.facet1 {
                (ridge.facet0, ridge.facet1)
            } else {
                (ridge.facet1, ridge.facet0)
            };
            if a != b {
                edges.push((a, b));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    });

    Some((
        GeneralFrontierRepairReport {
            initial_facets,
            final_facets: facets.len(),
            remaining_frontier_ridges,
            steps_attempted,
            new_facets,
            step_limit_reached,
            diagnostics,
        },
        edges,
    ))
}

#[allow(clippy::too_many_arguments)]
fn repair_facet_graph_from_inexact_core<'norm, M: Rat, N: Num>(
    exact_poly: &PolyhedronOutput<M, Generator>,
    kept_output_rows: &[Row],
    kept_vertices: &[Vec<Row>],
    int_input_rows: &super::IntRowMatrix<<M as Rat>::Int>,
    int_output_rows: Option<&'norm super::IntRowMatrix<<M as Rat>::Int>>,
    redund_cols: &[usize],
    redund_mask: &[bool],
    inexact: &PolyhedronOutput<N, Generator>,
    options: FacetGraphRepairOptions,
    eps: &impl Epsilon<M>,
) -> Result<FacetGraphRepairResult<M>, FacetGraphRepairError> {
    if kept_vertices.len() != kept_output_rows.len() {
        return Err(FacetGraphRepairError::InputShapeMismatch);
    }
    let Some(inexact_incidence) = inexact.incidence() else {
        return Err(FacetGraphRepairError::MissingInexactOutputIncidence);
    };
    let Some(inexact_adjacency) = inexact.adjacency() else {
        return Err(FacetGraphRepairError::MissingInexactOutputAdjacency);
    };

    if exact_poly.input().row_count() != inexact.input().row_count()
        || exact_poly.input().col_count() != inexact.input().col_count()
    {
        return Err(FacetGraphRepairError::InputShapeMismatch);
    }

    let exact_output_linearity = exact_poly.output().linearity();
    if !exact_output_linearity.is_empty() {
        return Err(FacetGraphRepairError::ExactOutputHasLinearity {
            count: exact_output_linearity.cardinality(),
        });
    }
    let inexact_output_linearity = inexact.output().linearity();
    if !inexact_output_linearity.is_empty() {
        return Err(FacetGraphRepairError::InexactOutputHasLinearity {
            count: inexact_output_linearity.cardinality(),
        });
    }

    let out_rows = inexact.output().row_count();
    if inexact_incidence.family_size() != out_rows || inexact_adjacency.family_size() != out_rows {
        return Err(FacetGraphRepairError::OutputShapeMismatch);
    }

    let input = exact_poly.input();
    let input_rows = input.row_count();
    let cols = input.col_count();

    let ray_rows = count_input_rays(input, eps);
    if ray_rows > 0 {
        return Err(FacetGraphRepairError::InputHasRays { count: ray_rows });
    }

    if inexact.column_mapping().len() != cols {
        return Err(FacetGraphRepairError::InputShapeMismatch);
    }
    if exact_poly.column_mapping().len() != cols {
        return Err(FacetGraphRepairError::InputShapeMismatch);
    }
    if exact_poly.column_mapping() != inexact.column_mapping() {
        return Err(FacetGraphRepairError::ColumnMappingMismatch);
    }

    if redund_mask.len() != cols {
        return Err(FacetGraphRepairError::InputShapeMismatch);
    }
    for &col in redund_cols {
        if col >= cols || !redund_mask[col] {
            return Err(FacetGraphRepairError::InputShapeMismatch);
        }
    }

    let ridge_target_rank = cols.saturating_sub(redund_cols.len() + 2);

    if int_input_rows.row_count() != input_rows || int_input_rows.col_count() != cols {
        return Err(FacetGraphRepairError::InputShapeMismatch);
    }
    let mut int_ctx =
        IntRepairCtx::new(int_input_rows).ok_or(FacetGraphRepairError::InputShapeMismatch)?;

    let mut diagnostics = FacetGraphRepairDiagnostics::default();
    let dd_hints = inexact.dd_repair_hints();
    let mut recover_scratch = FacetGraphRecoverScratch::default();

    let mut facets: Vec<FacetRecord<'norm, <M as Rat>::Int>> = Vec::new();
    let mut buckets: AHashMap<u64, SmallVec<[usize; 4]>> = AHashMap::new();

    let mut node_to_facet: Vec<Option<usize>> = vec![None; out_rows];
    let mut candidate_edges: Vec<(usize, usize)> = Vec::new();

    let mut queue: VecDeque<Row> = VecDeque::new();
    let mut queued: Vec<bool> = vec![false; out_rows];
    let mut processed: Vec<bool> = vec![false; out_rows];

    for (new_idx, &orig_out_idx) in kept_output_rows.iter().enumerate() {
        if orig_out_idx >= out_rows {
            diagnostics.kept_output_row_oob += 1;
            continue;
        }

        let key = kept_vertices
            .get(new_idx)
            .ok_or(FacetGraphRepairError::InputShapeMismatch)?
            .iter()
            .copied()
            .filter(|&v| v < input_rows)
            .collect::<Vec<Row>>();

        let hash = hash_key(&key);
        let existing = buckets.get(&hash).and_then(|candidates| {
            candidates
                .iter()
                .copied()
                .find(|&id| facets[id].vertices == key)
        });
        let facet_id = match existing {
            Some(id) => id,
            None => {
                let id = facets.len();
                buckets.entry(hash).or_default().push(id);
                let normal = if let Some(int_output_rows) = int_output_rows {
                    Cow::Borrowed(
                        int_output_rows
                            .row(new_idx)
                            .ok_or(FacetGraphRepairError::InputShapeMismatch)?,
                    )
                } else {
                    Cow::Owned(
                        super::scaled_integer_vec::<M>(
                            exact_poly
                                .output()
                                .row(new_idx)
                                .ok_or(FacetGraphRepairError::InputShapeMismatch)?,
                        )
                        .ok_or(FacetGraphRepairError::InputShapeMismatch)?,
                    )
                };
                facets.push(FacetRecord {
                    vertices: key,
                    normal,
                });
                id
            }
        };

        node_to_facet[orig_out_idx] = Some(facet_id);
        if !queued[orig_out_idx] {
            queue.push_back(orig_out_idx);
            queued[orig_out_idx] = true;
        }
    }

    let initial_known_nodes = node_to_facet.iter().filter(|v| v.is_some()).count();
    let initial_facets = facets.len();

    // Fast path: if partial resolve already covered every inexact node (no missing facets), then we
    // can skip the BFS repair loop entirely.
    if options.frontier == FrontierRepairMode::Off
        && options.rebuild_polyhedron_output
        && initial_known_nodes == out_rows
        && initial_facets == out_rows
        && exact_poly.output().row_count() == out_rows
        && exact_poly.incidence().is_some()
        && exact_poly.adjacency().is_some()
    {
        let mut repaired_facets: Vec<RepairedFacet<M>> = Vec::with_capacity(out_rows);
        for (idx, facet) in facets.into_iter().enumerate() {
            let normal: Vec<M> = exact_poly
                .output()
                .row(idx)
                .ok_or(FacetGraphRepairError::OutputShapeMismatch)?
                .to_vec();
            repaired_facets.push(RepairedFacet {
                vertices: facet.vertices,
                normal,
            });
        }

        let mut rebuilt = exact_poly.clone();
        rebuilt.status = ComputationStatus::AllFound;
        return Ok(FacetGraphRepairResult {
            facets: repaired_facets,
            report: FacetGraphRepairReport {
                initial_known_nodes,
                initial_facets,
                final_known_nodes: out_rows,
                final_facets: out_rows,
                unresolved_nodes: 0,
                steps_attempted: 0,
                diagnostics,
                frontier: None,
            },
            rebuilt_polyhedron: Some(rebuilt),
        });
    }

    let mut steps_attempted = 0usize;

    // Fast-path: if partial resolution already covers every inexact node, there is no facet-graph
    // work to do. Skipping the BFS avoids scanning the full inexact adjacency.
    if initial_known_nodes < out_rows {
        loop {
            while let Some(i) = queue.pop_front() {
                if processed[i] {
                    continue;
                }
                processed[i] = true;

                let Some(facet_id_i) = node_to_facet[i] else {
                    continue;
                };

                let Some(neighbors) = inexact_adjacency.set(i) else {
                    diagnostics.missing_inexact_adjacency_set += 1;
                    continue;
                };

                let pending: Vec<(Row, RecoveredFacetInt<<M as Rat>::Int>)> = {
                    let facet = &facets[facet_id_i];
                    let normal_i = facet.normal.as_ref();
                    let vertices_i = &facet.vertices;

                    let mut pending: Vec<(Row, RecoveredFacetInt<<M as Rat>::Int>)> = Vec::new();

                    for neighbor in neighbors.iter() {
                        if steps_attempted >= options.max_steps
                            || facets.len() + pending.len() >= options.max_facets
                        {
                            break;
                        }

                        let j = neighbor.as_index();
                        if j >= out_rows {
                            continue;
                        }
                        if let Some(facet_id_j) = node_to_facet[j] {
                            let (a, b) = if facet_id_i < facet_id_j {
                                (facet_id_i, facet_id_j)
                            } else {
                                (facet_id_j, facet_id_i)
                            };
                            if a != b {
                                candidate_edges.push((a, b));
                            }
                            continue;
                        }

                        steps_attempted += 1;

                        let Some(face_j) = inexact_incidence.set(j) else {
                            diagnostics.missing_inexact_incidence_set += 1;
                            continue;
                        };

                        let Some(step) = recover_facet_from_inexact_neighbor::<M>(
                            &mut int_ctx,
                            normal_i,
                            vertices_i,
                            i,
                            j,
                            face_j,
                            input_rows,
                            ridge_target_rank,
                            redund_cols,
                            redund_mask,
                            dd_hints,
                            &mut recover_scratch,
                            &mut diagnostics,
                        ) else {
                            continue;
                        };

                        if step.minimizers > 1 {
                            diagnostics.tie_steps += 1;
                        }

                        pending.push((j, step));
                    }

                    pending
                };

                for (j, step) in pending {
                    if facets.len() >= options.max_facets {
                        break;
                    }

                    let hash = hash_key(&step.vertices);
                    let existing = buckets.get(&hash).and_then(|candidates| {
                        candidates
                            .iter()
                            .copied()
                            .find(|&id| facets[id].vertices == step.vertices)
                    });

                    let (facet_id, is_new) = match existing {
                        Some(existing) => (existing, false),
                        None => {
                            let id = facets.len();
                            buckets.entry(hash).or_default().push(id);
                            facets.push(FacetRecord {
                                vertices: step.vertices,
                                normal: Cow::Owned(step.normal),
                            });
                            (id, true)
                        }
                    };

                    if !is_new {
                        diagnostics.dedupe_hits += 1;
                    } else {
                        diagnostics.new_facets += 1;
                    }

                    if facet_id_i != facet_id {
                        let (a, b) = if facet_id_i < facet_id {
                            (facet_id_i, facet_id)
                        } else {
                            (facet_id, facet_id_i)
                        };
                        if a != b {
                            candidate_edges.push((a, b));
                        }
                    }

                    node_to_facet[j] = Some(facet_id);
                    if !queued[j] {
                        queue.push_back(j);
                        queued[j] = true;
                    }
                }
            }

            if steps_attempted >= options.max_steps || facets.len() >= options.max_facets {
                break;
            }

            let mut recovered: Option<(Row, usize, RecoveredFacetInt<<M as Rat>::Int>)> = None;

            let dd_edge_hints = dd_hints.and_then(|hints| hints.incoming_edge_hints.as_deref());
            let mut best_unresolved: Option<(usize, u8, Row)> = None;

            for j in 0..out_rows {
                if node_to_facet[j].is_some() {
                    continue;
                }
                if inexact_incidence.set(j).is_none() {
                    diagnostics.missing_inexact_incidence_set += 1;
                    continue;
                }
                let Some(neighbors) = inexact_adjacency.set(j) else {
                    diagnostics.missing_inexact_adjacency_set += 1;
                    continue;
                };

                let mut known_neighbors = 0usize;
                let mut best_hint_score = 0u8;
                for neighbor in neighbors.iter() {
                    let i = neighbor.as_index();
                    if i >= out_rows {
                        continue;
                    }
                    if node_to_facet[i].is_none() {
                        continue;
                    }
                    known_neighbors += 1;

                    if let Some(hint) = find_dd_edge_hint(dd_edge_hints, j, i) {
                        let mut score = 1u8;
                        if hint.entered_row.is_some() {
                            score = score.max(2);
                        }
                        if hint.minimizers_complete && !hint.minimizers.is_empty() {
                            score = score.max(3);
                        }
                        best_hint_score = best_hint_score.max(score);
                        if best_hint_score == 3 {
                            // Can't do better than "complete minimizers present".
                            break;
                        }
                    }
                }

                if known_neighbors == 0 {
                    continue;
                }

                match best_unresolved {
                    None => best_unresolved = Some((known_neighbors, best_hint_score, j)),
                    Some((best_known, best_hint, best_j)) => {
                        let better = (known_neighbors, best_hint_score) > (best_known, best_hint)
                            || ((known_neighbors, best_hint_score) == (best_known, best_hint)
                                && j < best_j);
                        if better {
                            best_unresolved = Some((known_neighbors, best_hint_score, j));
                        }
                    }
                }
            }

            if let Some((_, _, j)) = best_unresolved {
                let face_j = inexact_incidence
                    .set(j)
                    .expect("best_unresolved must have incidence present");
                let Some(neighbors) = inexact_adjacency.set(j) else {
                    diagnostics.missing_inexact_adjacency_set += 1;
                    continue;
                };

                'search: for desired_score in (0u8..=3u8).rev() {
                    for neighbor in neighbors.iter() {
                        if steps_attempted >= options.max_steps
                            || facets.len() >= options.max_facets
                        {
                            break 'search;
                        }

                        let i = neighbor.as_index();
                        if i >= out_rows {
                            continue;
                        }
                        let Some(facet_id_i) = node_to_facet[i] else {
                            continue;
                        };

                        let hint_score = find_dd_edge_hint(dd_edge_hints, j, i)
                            .map(|hint| {
                                let mut score = 1u8;
                                if hint.entered_row.is_some() {
                                    score = score.max(2);
                                }
                                if hint.minimizers_complete && !hint.minimizers.is_empty() {
                                    score = score.max(3);
                                }
                                score
                            })
                            .unwrap_or(0);
                        if hint_score != desired_score {
                            continue;
                        }

                        steps_attempted += 1;

                        let facet = &facets[facet_id_i];
                        let Some(step) = recover_facet_from_inexact_neighbor::<M>(
                            &mut int_ctx,
                            facet.normal.as_ref(),
                            &facet.vertices,
                            i,
                            j,
                            face_j,
                            input_rows,
                            ridge_target_rank,
                            redund_cols,
                            redund_mask,
                            dd_hints,
                            &mut recover_scratch,
                            &mut diagnostics,
                        ) else {
                            continue;
                        };

                        if step.minimizers > 1 {
                            diagnostics.tie_steps += 1;
                        }

                        recovered = Some((j, facet_id_i, step));
                        break 'search;
                    }
                }
            }

            let Some((j, from_facet_id, step)) = recovered else {
                break;
            };

            let hash = hash_key(&step.vertices);
            let existing = buckets.get(&hash).and_then(|candidates| {
                candidates
                    .iter()
                    .copied()
                    .find(|&id| facets[id].vertices == step.vertices)
            });

            let (facet_id, is_new) = match existing {
                Some(existing) => (existing, false),
                None => {
                    let id = facets.len();
                    buckets.entry(hash).or_default().push(id);
                    facets.push(FacetRecord {
                        vertices: step.vertices,
                        normal: Cow::Owned(step.normal),
                    });
                    (id, true)
                }
            };

            if !is_new {
                diagnostics.dedupe_hits += 1;
            } else {
                diagnostics.new_facets += 1;
            }

            if from_facet_id != facet_id {
                let (a, b) = if from_facet_id < facet_id {
                    (from_facet_id, facet_id)
                } else {
                    (facet_id, from_facet_id)
                };
                if a != b {
                    candidate_edges.push((a, b));
                }
            }

            node_to_facet[j] = Some(facet_id);
            if !queued[j] {
                queue.push_back(j);
                queued[j] = true;
            }
        }
    }

    let final_known_nodes = node_to_facet.iter().filter(|v| v.is_some()).count();
    let unresolved_nodes = node_to_facet.iter().filter(|v| v.is_none()).count();

    let mut frontier: Option<FrontierRepairReport> = None;
    let mut frontier_added_facets = false;
    let mut frontier_edges: Option<Vec<(usize, usize)>> = None;
    if unresolved_nodes == 0 && options.frontier_max_steps > 0 {
        match options.frontier {
            FrontierRepairMode::Off => {}
            FrontierRepairMode::Simplicial => {
                let before = facets.len();
                if let Some((report, edges)) = repair_simplicial_frontier_on_facet_records_int(
                    &mut int_ctx,
                    &mut facets,
                    &mut buckets,
                    exact_poly.dimension(),
                    redund_cols,
                    redund_mask,
                    options.frontier_max_steps,
                    options.max_facets,
                ) {
                    frontier_added_facets = facets.len() != before;
                    frontier_edges = edges;
                    frontier = Some(FrontierRepairReport::Simplicial(report));
                }
            }
            FrontierRepairMode::General => {
                let before = facets.len();
                if let Some((report, edges)) = repair_general_frontier_on_facet_records_int(
                    &mut int_ctx,
                    &mut facets,
                    &mut buckets,
                    exact_poly.dimension(),
                    ridge_target_rank,
                    redund_cols,
                    redund_mask,
                    options.frontier_max_steps,
                    options.max_facets,
                ) {
                    frontier_added_facets = facets.len() != before;
                    frontier_edges = edges;
                    frontier = Some(FrontierRepairReport::General(report));
                }
            }
        }
    }

    let final_facets = facets.len();

    let mut repaired_facets: Vec<RepairedFacet<M>> = Vec::with_capacity(facets.len());
    for facet in facets {
        repaired_facets.push(RepairedFacet {
            vertices: facet.vertices,
            normal: int_row_to_rat_row::<M>(facet.normal.as_ref()),
        });
    }
    candidate_edges.sort_unstable();
    candidate_edges.dedup();

    let frontier_done = frontier
        .as_ref()
        .map(|r| r.remaining_frontier_ridges() == 0 && !r.step_limit_reached())
        .unwrap_or(true);
    let rebuilt_status = if unresolved_nodes == 0 && frontier_done {
        ComputationStatus::AllFound
    } else {
        ComputationStatus::InProgress
    };

    let (candidate_edges_for_rebuild, candidate_edges_exact) = if let Some(edges) =
        frontier_edges.as_deref()
        && frontier_done
    {
        (Some(edges), true)
    } else if frontier_added_facets {
        (None, false)
    } else {
        (Some(candidate_edges.as_slice()), false)
    };
    let rebuilt_polyhedron = if !options.rebuild_polyhedron_output {
        None
    } else {
        Some(rebuild_polyhedron_output_from_facets(
            exact_poly,
            &repaired_facets,
            rebuilt_status,
            candidate_edges_for_rebuild,
            candidate_edges_exact,
        ))
    };

    Ok(FacetGraphRepairResult {
        facets: repaired_facets,
        report: FacetGraphRepairReport {
            initial_known_nodes,
            initial_facets,
            final_known_nodes,
            final_facets,
            unresolved_nodes,
            steps_attempted,
            diagnostics,
            frontier,
        },
        rebuilt_polyhedron,
    })
}

fn repair_facet_graph_from_inexact<M: Rat, N: Num>(
    partial: &PartialResolveResult<M, Generator>,
    inexact: &PolyhedronOutput<N, Generator>,
    options: FacetGraphRepairOptions,
    eps: &impl Epsilon<M>,
) -> Result<FacetGraphRepairResult<M>, FacetGraphRepairError> {
    let exact_poly = partial.polyhedron();
    let Some(exact_incidence) = exact_poly.incidence() else {
        return Err(FacetGraphRepairError::MissingExactOutputIncidence);
    };
    let input_rows = exact_poly.input().row_count();

    let mut kept_vertices: Vec<Vec<Row>> = Vec::with_capacity(partial.kept_output_rows().len());
    for new_idx in 0..partial.kept_output_rows().len() {
        let Some(face) = exact_incidence.set(new_idx) else {
            return Err(FacetGraphRepairError::MissingExactOutputIncidence);
        };
        kept_vertices.push(
            face.iter()
                .map(|v| v.as_index())
                .filter(|&v| v < input_rows)
                .collect(),
        );
    }

    let cols = exact_poly.input().col_count();
    if inexact.column_mapping().len() != cols {
        return Err(FacetGraphRepairError::InputShapeMismatch);
    }
    let redund_cols = PolyhedronOutput::<N, Generator>::redundant_cols_from_column_mapping(
        inexact.column_mapping(),
    );
    let mut redund_mask = vec![false; cols];
    for &col in &redund_cols {
        redund_mask[col] = true;
    }

    let int_input_rows = super::scaled_integer_rows(exact_poly.input())
        .ok_or(FacetGraphRepairError::InputShapeMismatch)?;

    repair_facet_graph_from_inexact_core(
        exact_poly,
        partial.kept_output_rows(),
        kept_vertices.as_slice(),
        &int_input_rows,
        None,
        &redund_cols,
        &redund_mask,
        inexact,
        options,
        eps,
    )
}

fn repair_facet_graph_from_inexact_prepared<M: Rat, N: Num>(
    prepared: &PreparedPartialResolveResult<M, Generator>,
    inexact: &PolyhedronOutput<N, Generator>,
    options: FacetGraphRepairOptions,
    eps: &impl Epsilon<M>,
) -> Result<FacetGraphRepairResult<M>, FacetGraphRepairError> {
    let exact_poly = prepared.polyhedron();
    let Some(exact_incidence) = exact_poly.incidence() else {
        return Err(FacetGraphRepairError::MissingExactOutputIncidence);
    };
    let input_rows = exact_poly.input().row_count();

    let mut kept_vertices: Vec<Vec<Row>> = Vec::with_capacity(prepared.kept_output_rows().len());
    for new_idx in 0..prepared.kept_output_rows().len() {
        let Some(face) = exact_incidence.set(new_idx) else {
            return Err(FacetGraphRepairError::MissingExactOutputIncidence);
        };
        kept_vertices.push(
            face.iter()
                .map(|v| v.as_index())
                .filter(|&v| v < input_rows)
                .collect(),
        );
    }

    repair_facet_graph_from_inexact_core(
        exact_poly,
        prepared.kept_output_rows(),
        kept_vertices.as_slice(),
        prepared.int_input_rows(),
        Some(prepared.int_output_rows()),
        prepared.redund_cols(),
        prepared.redund_mask(),
        inexact,
        options,
        eps,
    )
}

fn repair_facet_graph_from_inexact_prepared_minimal<M: Rat, N: Num>(
    prepared: &PreparedPartialResolveMinimal<M, Generator>,
    inexact: &PolyhedronOutput<N, Generator>,
    options: FacetGraphRepairOptions,
    eps: &impl Epsilon<M>,
) -> Result<FacetGraphRepairResult<M>, FacetGraphRepairError> {
    repair_facet_graph_from_inexact_core(
        prepared.template(),
        prepared.kept_output_rows(),
        prepared.facet_vertices(),
        prepared.int_input_rows(),
        Some(prepared.int_output_rows()),
        prepared.redund_cols(),
        prepared.redund_mask(),
        inexact,
        options,
        eps,
    )
}

#[derive(Clone, Debug)]
struct RecoveredFacet<N: Num> {
    vertices: Vec<Row>,
    normal: Vec<N>,
    minimizers: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RepairStepFailure {
    RidgeDirectionNotOneDim,
    OrientationNoWitness,
    NoBlockingVertex,
    InfeasibleNewFacet,
}

#[allow(clippy::too_many_arguments)]
fn recover_facet_across_ridge<N: Num>(
    input: &LpMatrix<N, Generator>,
    facet_normal: &[N],
    facet_vertices: &[Row],
    ridge_candidates: &[Row],
    ridge_basis: &[Row],
    candidate_vertices: Option<&[Row]>,
    redund_cols: &[usize],
    redund_mask: &[bool],
    eps: &impl Epsilon<N>,
) -> Result<RecoveredFacet<N>, RepairStepFailure> {
    let cols = input.col_count();
    debug_assert_eq!(facet_normal.len(), cols, "facet normal dimension mismatch");
    debug_assert_eq!(redund_mask.len(), cols, "redund_mask must have length cols");

    let mut unit_cols_base: SmallVec<[usize; 8]> = SmallVec::new();
    unit_cols_base.extend_from_slice(redund_cols);

    let mut direction = None;
    for pivot_col in (0..cols).filter(|&c| !redund_mask[c] && !eps.is_zero(&facet_normal[c])) {
        let mut unit_cols = unit_cols_base.clone();
        unit_cols.push(pivot_col);
        if let Some(vec) =
            input
                .storage()
                .solve_nullspace_1d_rows_with_unit_cols(ridge_basis, &unit_cols, eps)
        {
            direction = Some(vec);
            break;
        }
    }
    let mut direction = direction.ok_or(RepairStepFailure::RidgeDirectionNotOneDim)?;

    orient_direction_away_from_facet(input, &mut direction, facet_vertices, ridge_candidates, eps)
        .ok_or(RepairStepFailure::OrientationNoWitness)?;

    if let Some(candidate_vertices) = candidate_vertices
        && let Some(ratio) = ratio_test_on_vertices(
            input,
            facet_normal,
            &direction,
            candidate_vertices.iter().copied(),
            eps,
        )
        && let Ok(step) = recovered_facet_from_ratio(input, facet_normal, &direction, ratio, eps)
    {
        return Ok(step);
    }

    let ratio = ratio_test_on_vertices(input, facet_normal, &direction, 0..input.row_count(), eps)
        .ok_or(RepairStepFailure::NoBlockingVertex)?;
    recovered_facet_from_global_ratio(input, facet_normal, &direction, facet_vertices, ratio, eps)
}

fn orient_direction_away_from_facet<N: Num>(
    input: &LpMatrix<N, Generator>,
    direction: &mut [N],
    facet_vertices: &[Row],
    ridge_vertices: &[Row],
    eps: &impl Epsilon<N>,
) -> Option<Row> {
    let matrix = input.storage();
    let mut ridge_pos = 0usize;
    for &v in facet_vertices {
        while ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] < v {
            ridge_pos += 1;
        }
        if ridge_pos < ridge_vertices.len() && ridge_vertices[ridge_pos] == v {
            continue;
        }

        let eval = linalg::dot(&matrix[v], direction);
        if eps.is_zero(&eval) {
            continue;
        }
        if eps.is_negative(&eval) {
            for coeff in direction.iter_mut() {
                *coeff = coeff.ref_neg();
            }
        }
        return Some(v);
    }
    None
}

fn recovered_facet_from_ratio<N: Num>(
    input: &LpMatrix<N, Generator>,
    facet_normal: &[N],
    direction: &[N],
    ratio: RatioTestResult<N>,
    eps: &impl Epsilon<N>,
) -> Result<RecoveredFacet<N>, RepairStepFailure> {
    let cols = input.col_count();
    let mut new_normal: Vec<N> = Vec::with_capacity(cols);
    for (n_i, d_i) in facet_normal.iter().zip(direction.iter()) {
        let mut acc = n_i.ref_mul(&ratio.best_den);
        linalg::add_mul_assign(&mut acc, &ratio.best_num, d_i);
        new_normal.push(acc);
    }

    let (vertices, supporting_ok) = supporting_incidence_key(input, &mut new_normal, eps);
    if !supporting_ok {
        return Err(RepairStepFailure::InfeasibleNewFacet);
    }

    Ok(RecoveredFacet {
        vertices,
        normal: new_normal,
        minimizers: ratio.minimizer_vertices.len(),
    })
}

fn recovered_facet_from_global_ratio<N: Num>(
    input: &LpMatrix<N, Generator>,
    facet_normal: &[N],
    direction: &[N],
    facet_vertices: &[Row],
    ratio: RatioTestResult<N>,
    eps: &impl Epsilon<N>,
) -> Result<RecoveredFacet<N>, RepairStepFailure> {
    let cols = input.col_count();
    let mut new_normal: Vec<N> = Vec::with_capacity(cols);
    for (n_i, d_i) in facet_normal.iter().zip(direction.iter()) {
        let mut acc = n_i.ref_mul(&ratio.best_den);
        linalg::add_mul_assign(&mut acc, &ratio.best_num, d_i);
        new_normal.push(acc);
    }

    let mut ridge_vertices: Vec<Row> = Vec::new();
    for &v in facet_vertices {
        if v >= input.row_count() {
            continue;
        }
        let eval = linalg::dot(input.row(v).expect("row index within bounds"), direction);
        if eps.is_zero(&eval) {
            ridge_vertices.push(v);
            continue;
        }
        if eps.is_negative(&eval) {
            return recovered_facet_from_ratio(input, facet_normal, direction, ratio, eps);
        }
    }

    let mut vertices: Vec<Row> = ridge_vertices;
    vertices.extend(ratio.minimizer_vertices.iter().copied());
    vertices.sort_unstable();
    vertices.dedup();

    Ok(RecoveredFacet {
        vertices,
        normal: new_normal,
        minimizers: ratio.minimizer_vertices.len(),
    })
}

fn ratio_test_on_vertices<N: Num, I>(
    input: &LpMatrix<N, Generator>,
    facet_normal: &[N],
    direction: &[N],
    vertices: I,
    eps: &impl Epsilon<N>,
) -> Option<RatioTestResult<N>>
where
    I: IntoIterator<Item = Row>,
{
    let matrix = input.storage();
    let mut has_best = false;
    let mut best_num = N::zero();
    let mut best_den = N::zero();
    let mut minimizer_vertices: SmallVec<[Row; 8]> = SmallVec::new();

    for v in vertices {
        if v >= input.row_count() {
            continue;
        }
        let vertex_row = &matrix[v];
        let (num, dir_eval) = linalg::dot2(vertex_row, facet_normal, direction);
        if !eps.is_positive(&num) {
            continue;
        }
        if !eps.is_negative(&dir_eval) {
            continue;
        }
        let den = dir_eval.ref_neg();

        if !has_best {
            has_best = true;
            best_num = num;
            best_den = den;
            minimizer_vertices.clear();
            minimizer_vertices.push(v);
            continue;
        }

        let lhs = num.ref_mul(&best_den);
        let rhs = best_num.ref_mul(&den);
        match eps.cmp(&lhs, &rhs) {
            Ordering::Less => {
                best_num = num;
                best_den = den;
                minimizer_vertices.clear();
                minimizer_vertices.push(v);
            }
            Ordering::Equal => {
                minimizer_vertices.push(v);
            }
            Ordering::Greater => {}
        }
    }

    has_best.then_some(RatioTestResult {
        best_num,
        best_den,
        minimizer_vertices,
    })
}

fn supporting_incidence_key<N: Num>(
    input: &LpMatrix<N, Generator>,
    normal: &mut [N],
    eps: &impl Epsilon<N>,
) -> (Vec<Row>, bool) {
    let mut vertices: Vec<Row> = Vec::new();
    let mut saw_neg = false;
    let mut saw_pos = false;

    for (v, row) in input.rows().enumerate() {
        let eval = linalg::dot(row, normal);
        match eps.sign(&eval) {
            Sign::Negative => saw_neg = true,
            Sign::Zero => vertices.push(v),
            Sign::Positive => saw_pos = true,
        }
        if saw_neg && saw_pos {
            return (vertices, false);
        }
    }

    if saw_neg {
        for coeff in normal.iter_mut() {
            *coeff = coeff.ref_neg();
        }
    }

    (vertices, true)
}

fn count_input_rays<N: Num>(input: &LpMatrix<N, Generator>, eps: &impl Epsilon<N>) -> usize {
    input
        .rows()
        .filter(|row| row.first().is_none_or(|b| eps.is_zero(b)))
        .count()
}

trait FacetData<N: Num> {
    fn vertices(&self) -> &[Row];
    fn normal(&self) -> &[N];
}

impl<N: Num> FacetData<N> for SimplicialFacet<N> {
    fn vertices(&self) -> &[Row] {
        SimplicialFacet::vertices(self)
    }

    fn normal(&self) -> &[N] {
        SimplicialFacet::normal(self)
    }
}

impl<N: Num> FacetData<N> for RepairedFacet<N> {
    fn vertices(&self) -> &[Row] {
        RepairedFacet::vertices(self)
    }

    fn normal(&self) -> &[N] {
        RepairedFacet::normal(self)
    }
}

fn adjacency_set_family_from_edges(facets: usize, edges: &[(usize, usize)]) -> SetFamily {
    let mut sets: Vec<RowSet> = (0..facets).map(|_| RowSet::new(facets)).collect();

    for &(a, b) in edges {
        if a >= facets || b >= facets || a == b {
            continue;
        }
        sets[a].insert(b);
        sets[b].insert(a);
    }

    let mut builder = SetFamily::builder(facets, facets);
    for (idx, set) in sets.into_iter().enumerate() {
        builder.replace_set(idx, set);
    }
    builder.build()
}

fn rebuild_polyhedron_output_from_facets<N: Num, F: FacetData<N>>(
    template: &PolyhedronOutput<N, Generator>,
    facets: &[F],
    status: ComputationStatus,
    candidate_edges: Option<&[(usize, usize)]>,
    candidate_edges_exact: bool,
) -> PolyhedronOutput<N, Generator> {
    let input = template.input().clone();

    let output_rows: Vec<Vec<N>> = facets.iter().map(|facet| facet.normal().to_vec()).collect();
    let output = if output_rows.is_empty() {
        LpMatrix::<N, Inequality>::new(0, input.col_count())
    } else {
        LpMatrix::<N, Inequality>::from_rows(output_rows)
    };

    let output_size = output.row_count();
    let incidence = if output_size == 0 {
        None
    } else {
        let set_capacity = input.row_count();
        let mut builder = SetFamily::builder(output_size, set_capacity);
        for (idx, facet) in facets.iter().enumerate() {
            let mut set = RowSet::new(set_capacity);
            for &v in facet.vertices() {
                if v < set_capacity {
                    set.insert(v);
                }
            }
            builder.replace_set(idx, set);
        }
        Some(builder.build())
    };

    let adjacency = if output_size < 2 {
        None
    } else if candidate_edges_exact {
        candidate_edges
            .filter(|edges| !edges.is_empty())
            .map(|edges| adjacency_set_family_from_edges(output_size, edges))
    } else {
        incidence.as_ref().and_then(|incidence| {
            let input_rank = input
                .col_count()
                .saturating_sub(template.linearity_dimension());
            let active_rows = RowSet::all(incidence.set_capacity());
            let edges = candidate_edges.filter(|e| !e.is_empty());
            build_adjacency(
                incidence,
                output.linearity(),
                &active_rows,
                input_rank,
                edges,
                false,
                None,
            )
        })
    };

    PolyhedronOutput::<N, Generator> {
        representation: template.representation(),
        homogeneous: template.homogeneous(),
        dimension: template.dimension(),
        input,
        output,
        equality_kinds: template.equality_kinds().to_vec(),
        linearity_dimension: template.linearity_dimension(),
        output_size,
        incidence,
        adjacency,
        input_incidence: None,
        input_adjacency: None,
        redundant_rows: template.redundant_rows().cloned(),
        dominant_rows: template.dominant_rows().cloned(),
        status,
        is_empty: output_size == 0,
        cost_vector: template.cost_vector().map(|v| v.to_vec()),
        row_positions: template.row_positions().clone(),
        column_mapping: template.column_mapping().to_vec(),
        repair_hints: None,
        adjacency_profile: None,
        trace: None,
    }
}

#[cfg(test)]
mod tests {
    use super::recover_facet_across_ridge;
    use crate::matrix::LpMatrixBuilder;
    use calculo::num::Num;
    use hullabaloo::types::Generator;

    #[test]
    #[cfg(feature = "rug")]
    fn cube_edge_pivot_has_two_minimizers() {
        use calculo::num::RugRat;

        let eps = RugRat::default_eps();
        let rat = |value: i32| RugRat::try_from_f64(value as f64).expect("small int to RugRat");

        // Cube vertices in 3D: rows are homogeneous points [1, x, y, z].
        let verts = [
            (-1, -1, -1),
            (-1, -1, 1),
            (-1, 1, -1),
            (-1, 1, 1),
            (1, -1, -1),
            (1, -1, 1),
            (1, 1, -1),
            (1, 1, 1),
        ];
        let rows: Vec<Vec<RugRat>> = verts
            .iter()
            .map(|&(x, y, z)| vec![rat(1), rat(x), rat(y), rat(z)])
            .collect();
        let matrix = LpMatrixBuilder::<RugRat, Generator>::from_rows(rows).build();

        // Facet x = 1: 1 - x >= 0.
        let facet_normal = vec![rat(1), rat(-1), rat(0), rat(0)];
        let facet_vertices = vec![4usize, 5, 6, 7];

        // Ridge x = 1 and y = 1: vertices (1,1,±1).
        let ridge_vertices = vec![6usize, 7];
        let ridge_basis = ridge_vertices.clone();

        let redund_mask = vec![false; matrix.col_count()];
        let step = recover_facet_across_ridge(
            &matrix,
            &facet_normal,
            &facet_vertices,
            &ridge_vertices,
            &ridge_basis,
            None,
            &[],
            &redund_mask,
            &eps,
        )
        .expect("pivot across cube edge must succeed");

        // Moving from x=1 to y=1, both vertices (-1,1,±1) become tight simultaneously.
        assert_eq!(step.minimizers, 2);
        assert_eq!(step.vertices, vec![2usize, 3, 6, 7]);
    }
}
