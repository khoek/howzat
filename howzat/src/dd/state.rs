use std::ops::{Deref, DerefMut};

use crate::dd::ConeOptions;
use crate::dd::builder::Cone;
use crate::dd::index::RayZeroSetIndex;
use crate::dd::mode::{HalfspaceMode, UmpireHalfspaceMode};
use crate::dd::ray::{RayData as _, RayGraph, RayId, RayListHeads, RayPartitionOwned};
use crate::dd::tableau::TableauState;
use crate::dd::zero::ZeroRepr;
use crate::dd::{ConeCtx, SinglePrecisionUmpire, Umpire};
use crate::matrix::LpMatrix;
use calculo::num::Num;
use hullabaloo::matrix::BasisMatrix;
use hullabaloo::types::{Col, ComputationStatus, Representation, Row, RowSet};

#[derive(Clone, Debug)]
pub(crate) struct IterationState {
    pub(crate) iteration: Row,
    pub(crate) col_reduced: bool,
    pub(crate) linearity_dim: Row,
    pub(crate) d_orig: usize,
    pub(crate) newcol: Vec<Option<usize>>,
    pub(crate) initial_ray_index: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ConeCore<
    N: Num,
    R: Representation,
    U = SinglePrecisionUmpire<N>,
    ZR: ZeroRepr = crate::dd::SatRepr,
> where
    U: Umpire<N, ZR>,
{
    pub(crate) input_matrix: LpMatrix<N, R>,
    pub(crate) ctx: ConeCtx<N, R, U::MatrixData<R>>,
    pub(crate) options: ConeOptions,
    pub(crate) recompute_row_order: bool,
    pub(crate) added_halfspaces: RowSet,
    pub(crate) weakly_added_halfspaces: RowSet,
    pub(crate) initial_halfspaces: RowSet,
    pub(crate) ground_set: RowSet,
    pub(crate) equality_set: RowSet,
    pub(crate) _strict_inequality_set: RowSet,
    pub(crate) redundant_halfspaces: RowSet,
    pub(crate) ray_graph: RayGraph<U::RayData>,
    pub(crate) iter_state: IterationState,
    pub(crate) count_intersections: u64,
    pub(crate) count_intersections_good: u64,
    pub(crate) count_intersections_bad: u64,
    pub(crate) comp_status: ComputationStatus,
    pub(crate) tableau: TableauState<N>,
    pub(crate) lists: RayListHeads,
    pub(crate) partitions: RayPartitionOwned,
    pub(crate) adj_face: ZR::Set,
    pub(crate) zero_set_pool: Vec<ZR::Set>,
    pub(crate) ray_index: RayZeroSetIndex,
    pub(crate) ray_incidence: ZR::IncidenceIndex,
    /// Scratch bitset over ray slot indices (used to avoid allocating in hot loops).
    pub(crate) candidate_ray_set: RowSet,
    pub(crate) pending_new_rays: Vec<RayId>,
    pub(crate) active_id_scratch: Vec<RayId>,
    pub(crate) removed_marks: Vec<u32>,
    pub(crate) removed_epoch: u32,
    pub(crate) edges_dirty: bool,
    pub(crate) dedup_drops: u64,
}

#[derive(Clone, Debug)]
pub struct ConeEngine<
    N: Num,
    R: Representation,
    U = SinglePrecisionUmpire<N>,
    ZR: ZeroRepr = crate::dd::SatRepr,
> where
    U: Umpire<N, ZR>,
{
    pub(crate) core: ConeCore<N, R, U, ZR>,
    pub(crate) umpire: U,
}

#[derive(Clone, Debug)]
pub struct ConeBasisPrep<
    N: Num,
    R: Representation,
    U = SinglePrecisionUmpire<N>,
    ZR: ZeroRepr = crate::dd::SatRepr,
> where
    U: Umpire<N, ZR>,
{
    pub(crate) state: ConeEngine<N, R, U, ZR>,
}

#[derive(Clone, Debug)]
pub struct ConeDd<
    N: Num,
    R: Representation,
    U = SinglePrecisionUmpire<N>,
    ZR: ZeroRepr = crate::dd::SatRepr,
> where
    U: Umpire<N, ZR>,
{
    pub(crate) state: ConeEngine<N, R, U, ZR>,
}

#[derive(Clone, Debug)]
pub struct ConeOutput<
    N: Num,
    R: Representation,
    U = SinglePrecisionUmpire<N>,
    ZR: ZeroRepr = crate::dd::SatRepr,
> where
    U: Umpire<N, ZR>,
{
    pub(crate) state: ConeEngine<N, R, U, ZR>,
}

#[derive(Clone, Debug)]
pub struct ColumnReduction<'a, N: Num> {
    original_dimension: usize,
    mapping: &'a [Option<usize>],
    saved_basis: &'a BasisMatrix<N>,
    reduced: bool,
}

impl<'a, N: Num> ColumnReduction<'a, N> {
    pub fn original_dimension(&self) -> usize {
        self.original_dimension
    }

    pub fn mapping(&self) -> &[Option<usize>] {
        self.mapping
    }

    pub fn saved_basis(&self) -> &BasisMatrix<N> {
        self.saved_basis
    }

    pub fn is_reduced(&self) -> bool {
        self.reduced
    }
}

impl<N: Num, R: Representation, U, ZR> ConeEngine<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    #[inline(always)]
    pub(crate) fn split_core_umpire(&mut self) -> (&mut ConeCore<N, R, U, ZR>, &mut U) {
        (&mut self.core, &mut self.umpire)
    }

    pub(crate) fn new_with_umpire(core: Cone<N, R>, umpire: U) -> Self {
        let Cone {
            matrix: input_matrix,
            equality_kinds,
            options,
            ground_set,
            equality_set,
            _strict_inequality_set: strict_inequality_set,
            ..
        } = core;
        let m = input_matrix.row_count();
        let d = input_matrix.col_count();

        let recompute_row_order =
            <UmpireHalfspaceMode<N, U, ZR> as HalfspaceMode>::initial_recompute_row_order();
        let mut umpire = umpire;
        let matrix_data = umpire.ingest(input_matrix.clone());

        let mut core = ConeCore {
            input_matrix,
            ctx: ConeCtx {
                matrix: matrix_data,
                equality_kinds,
                order_vector: Vec::new(),
                row_to_pos: Vec::new(),
                lex_order_cache: None,
                sat_row_to_id: vec![None; m],
                sat_id_to_row: Vec::new(),
                _phantom: std::marker::PhantomData,
            },
            options: options.clone(),
            recompute_row_order,
            added_halfspaces: RowSet::new(m),
            weakly_added_halfspaces: RowSet::new(m),
            initial_halfspaces: RowSet::new(m),
            ground_set,
            equality_set,
            _strict_inequality_set: strict_inequality_set,
            redundant_halfspaces: RowSet::new(m),
            ray_graph: RayGraph::new(),
            iter_state: IterationState {
                iteration: 0,
                col_reduced: false,
                linearity_dim: 0,
                d_orig: d,
                newcol: (0..d).map(Some).collect(),
                initial_ray_index: vec![None; d],
            },
            count_intersections: 0,
            count_intersections_good: 0,
            count_intersections_bad: 0,
            comp_status: ComputationStatus::InProgress,
            tableau: TableauState {
                basis: BasisMatrix::identity(d),
                basis_saved: BasisMatrix::identity(d),
                tableau: Vec::new(),
                tableau_rows: 0,
                tableau_cols: 0,
                tableau_nonbasic: Vec::new(),
                tableau_basic_col_for_row: Vec::new(),
                row_status: Vec::new(),
                pivot_row: Vec::new(),
                factors: Vec::new(),
            },
            lists: RayListHeads::default(),
            partitions: RayPartitionOwned::default(),
            adj_face: ZR::empty_set(m),
            zero_set_pool: Vec::new(),
            ray_index: RayZeroSetIndex::default(),
            ray_incidence: ZR::IncidenceIndex::default(),
            candidate_ray_set: RowSet::new(0),
            pending_new_rays: Vec::new(),
            active_id_scratch: Vec::new(),
            removed_marks: Vec::new(),
            removed_epoch: 0,
            edges_dirty: false,
            dedup_drops: 0,
        };
        for row_id in core.equality_set.iter() {
            let row = row_id.as_index();
            let _ = ZR::ensure_id_for_row(&mut core.ctx, row);
        }
        umpire.recompute_row_order_vector(&mut core.ctx, &core._strict_inequality_set);
        // Initialize any umpire-maintained infeasible-count caches (no-op for policies that don't use them).
        umpire.reset_infeasible_counts(m);
        core.edges_dirty = true;
        Self { core, umpire }
    }

    pub fn matrix(&self) -> &LpMatrix<N, R> {
        self.core.ctx.matrix()
    }

    pub fn input_matrix(&self) -> &LpMatrix<N, R> {
        &self.core.input_matrix
    }

    pub fn added_halfspaces(&self) -> &RowSet {
        &self.core.added_halfspaces
    }

    pub fn weakly_added_halfspaces(&self) -> &RowSet {
        &self.core.weakly_added_halfspaces
    }

    pub fn initial_halfspaces(&self) -> &RowSet {
        &self.core.initial_halfspaces
    }

    pub fn ground_set(&self) -> &RowSet {
        &self.core.ground_set
    }

    pub fn iteration(&self) -> Row {
        self.core.iter_state.iteration
    }

    pub fn linearity_dimension(&self) -> Row {
        self.core.iter_state.linearity_dim
    }

    pub fn column_reduction(&self) -> ColumnReduction<'_, N> {
        ColumnReduction {
            original_dimension: self.core.iter_state.d_orig,
            mapping: &self.core.iter_state.newcol,
            saved_basis: &self.core.tableau.basis_saved,
            reduced: self.core.iter_state.col_reduced,
        }
    }

    pub fn column_mapping(&self) -> &[Option<usize>] {
        &self.core.iter_state.newcol
    }

    pub fn original_dimension(&self) -> usize {
        self.core.iter_state.d_orig
    }

    pub fn is_col_reduced(&self) -> bool {
        self.core.iter_state.col_reduced
    }

    pub fn basis_saved(&self) -> &BasisMatrix<N> {
        &self.core.tableau.basis_saved
    }

    pub fn replace_saved_basis(&mut self, basis: BasisMatrix<N>) {
        self.core.tableau.basis_saved = basis;
    }

    pub fn status(&self) -> ComputationStatus {
        self.core.comp_status
    }

    pub fn nondegenerate_assumed(&self) -> bool {
        self.core.options.assumes_nondegeneracy()
    }

    pub fn relaxed_enumeration(&self) -> bool {
        self.core.options.relaxed_enumeration()
    }

    pub fn init_basis_at_bot(&self) -> bool {
        self.core.options.init_basis_at_bot()
    }

    pub fn rays(&self) -> impl Iterator<Item = &U::RayData> {
        self.core.ray_graph.active_rays()
    }

    pub fn feasible_rays(&self) -> impl Iterator<Item = &U::RayData> {
        self.rays().filter(|ray| ray.is_feasible())
    }

    pub fn row_count(&self) -> Row {
        self.core.ctx.matrix().row_count()
    }

    pub fn col_count(&self) -> Col {
        self.core.ctx.matrix().col_count()
    }

    pub fn tableau_snapshot(&self) -> Option<Vec<Vec<N>>> {
        let rows = self.core.tableau.tableau_rows;
        let cols = self.core.tableau.tableau_cols;
        if rows == 0 || cols == 0 || self.core.tableau.tableau.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let start = r * cols;
            let end = start + cols;
            out.push(self.core.tableau.tableau[start..end].to_vec());
        }
        Some(out)
    }

    pub fn tableau_nonbasic(&self) -> &[isize] {
        &self.core.tableau.tableau_nonbasic
    }

    pub fn tableau_basic_cols(&self) -> &[isize] {
        &self.core.tableau.tableau_basic_col_for_row
    }
}

impl<N: Num, R: Representation, U, ZR> Deref for ConeOutput<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    type Target = ConeEngine<N, R, U, ZR>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<N: Num, R: Representation, U, ZR> Deref for ConeBasisPrep<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    type Target = ConeEngine<N, R, U, ZR>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<N: Num, R: Representation, U, ZR> Deref for ConeDd<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    type Target = ConeEngine<N, R, U, ZR>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<N: Num, R: Representation, U, ZR> DerefMut for ConeOutput<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<N: Num, R: Representation, U, ZR> DerefMut for ConeBasisPrep<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<N: Num, R: Representation, U, ZR> DerefMut for ConeDd<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<N: Num, R: Representation, U, ZR> ConeEngine<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    #[cfg(test)]
    pub(crate) fn sign_sets_for_ray_id(
        &mut self,
        id: RayId,
        relaxed: bool,
        force_infeasible: bool,
    ) -> Option<RowSet> {
        let row_count = self.row_count();
        let mut negative_set = RowSet::new(row_count);
        let ray_data = self.core.ray_graph.ray_data(id)?;
        self.umpire.sign_sets_for_ray(
            &self.core.ctx,
            ray_data,
            relaxed,
            force_infeasible,
            &mut negative_set,
        );
        Some(negative_set)
    }
}

impl<N: Num, R: Representation, U, ZR> ConeEngine<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    pub(crate) fn order_position(&self, row: Row) -> Option<Row> {
        self.core
            .ctx
            .row_to_pos
            .get(row)
            .copied()
            .filter(|pos| *pos < self.row_count())
    }

    pub(crate) fn swap_row_into_iteration_slot(&mut self, row: Row) {
        let iteration = self.core.iter_state.iteration;
        if iteration >= self.core.ctx.order_vector.len() {
            return;
        }
        let Some(pos) = self.order_position(row) else {
            return;
        };
        if pos == iteration {
            return;
        }
        let current = self.core.ctx.order_vector[iteration];
        self.core.ctx.order_vector[iteration] = row;
        self.core.ctx.order_vector[pos] = current;
        self.core.ctx.row_to_pos[row] = iteration;
        self.core.ctx.row_to_pos[current] = pos;
    }
}
