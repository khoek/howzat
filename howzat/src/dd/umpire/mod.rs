use crate::HowzatError as Error;
use crate::dd::umpire::policies::HalfspacePolicy as _;
use crate::dd::RayId;
use crate::dd::sat::SatRowId;
use crate::dd::zero::ZeroRepr;
use crate::lp::LpSolver;
use crate::matrix::{LpMatrix, MatrixRank};
use calculo::linalg;
use calculo::num::{Epsilon, Num, Sign};
use hullabaloo::matrix::BasisMatrix;
use hullabaloo::types::{Col, ColSet, InequalityKind, Representation, Row, RowSet};

mod adaptive_precision;
mod int;
mod multi_precision;
mod kernels {
    use crate::dd::tableau::TableauState;
    use calculo::linalg;
    use calculo::num::{Epsilon, Num};
    use hullabaloo::types::{Col, Row};

    /// Baseline single-precision pivot step (migrated from `ConeEngine::gaussian_column_pivot`).
    #[inline(always)]
    pub(crate) fn gaussian_column_pivot_step<N: Num, E: Epsilon<N>>(
        tableau: &mut TableauState<N>,
        r: Row,
        s: Col,
        eps: &E,
    ) {
        if tableau.is_empty() {
            return;
        }
        let col_count = tableau.tableau_cols;
        let row_count = tableau.tableau_rows;
        debug_assert!(r < row_count, "pivot row out of range");
        debug_assert!(s < col_count, "pivot col out of range");

        let mut pivot_row = std::mem::take(&mut tableau.pivot_row);
        pivot_row.resize(col_count, N::zero());
        pivot_row.clone_from_slice(tableau.row(r));
        let pivot = pivot_row[s].clone();
        tableau.pivot_row = pivot_row;
        if eps.is_zero(&pivot) {
            return;
        }

        tableau.basis.mark_non_identity();
        let pivot_inv = N::one().ref_div(&pivot);
        tableau.factors.resize(col_count, N::zero());
        for (j, factor) in tableau.factors.iter_mut().enumerate() {
            if j == s || eps.is_zero(&tableau.pivot_row[j]) {
                *factor = N::zero();
                continue;
            }
            let scaled = tableau.pivot_row[j].ref_mul(&pivot_inv);
            if eps.is_zero(&scaled) {
                *factor = N::zero();
            } else {
                *factor = scaled;
            }
        }
        let factors = tableau.factors.as_slice();

        // Update basis row-wise to avoid strided access and repeated clones.
        for basis_row in tableau.basis.rows_mut() {
            let pivot_entry = basis_row[s].clone();
            if !eps.is_zero(&pivot_entry) {
                linalg::axpy_sub(basis_row, &pivot_entry, factors);
            }
            basis_row[s] = pivot_entry.ref_mul(&pivot_inv);
        }

        let cols = col_count;
        for row_idx in 0..row_count {
            let start = row_idx * cols;
            let row_slice = &mut tableau.tableau[start..start + cols];
            let pivot_entry = row_slice[s].clone();
            if !eps.is_zero(&pivot_entry) {
                linalg::axpy_sub(row_slice, &pivot_entry, factors);
            }
            row_slice[s] = pivot_entry.ref_mul(&pivot_inv);
        }
    }

    /// Pivot step + tableau book-keeping.
    #[inline(always)]
    pub(crate) fn gaussian_column_pivot<N: Num, E: Epsilon<N>>(
        tableau: &mut TableauState<N>,
        r: Row,
        s: Col,
        eps: &E,
    ) {
        gaussian_column_pivot_step(tableau, r, s, eps);
        let entering = tableau.tableau_nonbasic[s];
        tableau.tableau_basic_col_for_row[r] = s as isize;
        tableau.tableau_nonbasic[s] = r as isize;
        if entering >= 0 {
            let idx = entering as usize;
            debug_assert!(
                idx < tableau.tableau_basic_col_for_row.len(),
                "tableau basis row out of range"
            );
            tableau.tableau_basic_col_for_row[idx] = -1;
        }
    }
}
pub mod policies;
mod single_precision;

pub use adaptive_precision::AdaptivePrecisionUmpire;
pub use int::IntUmpire;
pub use multi_precision::MultiPrecisionUmpire;
pub use single_precision::SinglePrecisionUmpire;
pub use single_precision::{NoPurifier, Purifier, SnapPurifier, UpcastingSnapPurifier};

/// Opaque (umpire-owned) representation for the cone's working matrix.
///
/// The DD core may hold onto this type, but must only use it through:
/// - `AsRef<LpMatrix<..>>` for read-only access to the base numeric matrix, and
/// - `select_columns` for column-reduction.
///
/// More refined tiers (e.g. exact shadows) should live *inside* this representation, not as a
/// separately-managed "shadow matrix" inside the umpire.
pub trait UmpireMatrix<N: Num, R: Representation>: Clone + std::fmt::Debug {
    fn base(&self) -> &LpMatrix<N, R>;

    fn select_columns(&self, columns: &[usize]) -> Result<Self, Error>;
}

impl<N: Num, R: Representation> UmpireMatrix<N, R> for LpMatrix<N, R> {
    #[inline(always)]
    fn base(&self) -> &LpMatrix<N, R> {
        self
    }

    #[inline(always)]
    fn select_columns(&self, columns: &[usize]) -> Result<Self, Error> {
        LpMatrix::select_columns(self, columns)
    }
}

/// Workspace for basis initialization.
///
/// This wrapper keeps `TableauState` crate-private while still letting `Umpire` implementations
/// fill the tableau (and basis) during the basis-prep phase.
pub struct BasisInitTableau<'a, N: Num> {
    tableau: &'a mut crate::dd::tableau::TableauState<N>,
}

impl<'a, N: Num> BasisInitTableau<'a, N> {
    pub(crate) fn new(tableau: &'a mut crate::dd::tableau::TableauState<N>) -> Self {
        Self { tableau }
    }

    #[inline(always)]
    pub fn set_basis(&mut self, basis: BasisMatrix<N>) {
        self.tableau.basis = basis;
    }

    #[inline(always)]
    pub fn set_basis_identity(&mut self, dim: usize) {
        self.tableau.basis = BasisMatrix::identity(dim);
    }

    #[inline(always)]
    pub fn set_tableau_nonbasic_default(&mut self, cols: Col) {
        self.tableau.tableau_nonbasic = (0..cols).map(|c| -((c as isize) + 1)).collect();
    }

    #[inline(always)]
    pub fn set_tableau_basic_col_for_row_len(&mut self, rows: Row) {
        self.tableau.tableau_basic_col_for_row = vec![-1; rows];
    }

    #[inline(always)]
    pub fn reset_storage(&mut self, rows: Row, cols: Col) {
        self.tableau.reset_storage(rows, cols);
    }

    #[inline(always)]
    pub fn row_mut(&mut self, row: Row) -> &mut [N] {
        let cols = self.tableau.tableau_cols;
        debug_assert!(row < self.tableau.tableau_rows, "row out of bounds");
        let start = row * cols;
        &mut self.tableau.tableau[start..start + cols]
    }

    #[inline(always)]
    pub fn row(&self, row: Row) -> &[N] {
        self.tableau.row(row)
    }

    #[inline(always)]
    pub fn clear_tableau_storage(&mut self) {
        self.tableau.tableau.clear();
        self.tableau.tableau_rows = 0;
        self.tableau.tableau_cols = 0;
    }

    #[inline(always)]
    pub fn apply_gaussian_column_pivot<E: Epsilon<N>>(&mut self, r: Row, s: Col, eps: &E) {
        kernels::gaussian_column_pivot(self.tableau, r, s, eps);
    }
}

/// Cone fields the umpire is allowed to consult to make numeric decisions.
///
/// Stored on `ConeEngine` so we can pass `&ConeCtx` directly (no per-call view construction).
#[derive(Clone, Debug)]
pub struct ConeCtx<N: Num, R: Representation, M: UmpireMatrix<N, R> = LpMatrix<N, R>> {
    pub(crate) matrix: M,
    pub(crate) equality_kinds: Vec<InequalityKind>,
    pub(crate) order_vector: Vec<Row>,
    pub(crate) row_to_pos: Vec<Row>,
    pub(crate) lex_order_cache: Option<LexOrderCache>,
    pub(crate) sat_row_to_id: Vec<Option<SatRowId>>,
    pub(crate) sat_id_to_row: Vec<Row>,
    pub(crate) _phantom: std::marker::PhantomData<(N, R)>,
}

#[derive(Clone, Debug)]
pub(crate) struct LexOrderCache {
    pub(crate) row_count: usize,
    pub(crate) col_count: usize,
    pub(crate) strict_sig: u64,
    pub(crate) non_strict: Vec<Row>,
    pub(crate) strict: Vec<Row>,
}

impl<N: Num, R: Representation, M: UmpireMatrix<N, R>> ConeCtx<N, R, M> {
    #[inline(always)]
    pub fn matrix(&self) -> &LpMatrix<N, R> {
        self.matrix.base()
    }

    #[inline(always)]
    pub(crate) fn sat_id_for_row(&self, row: Row) -> Option<SatRowId> {
        self.sat_row_to_id.get(row).copied().flatten()
    }

    #[inline(always)]
    pub(crate) fn sat_row_for_id(&self, id: SatRowId) -> Row {
        self.sat_id_to_row
            .get(id.as_index())
            .copied()
            .unwrap_or_else(|| panic!("sat row id {:?} out of bounds", id))
    }

    pub(crate) fn assign_sat_id_for_row(&mut self, row: Row) -> SatRowId {
        if self.sat_row_to_id.len() != self.matrix().row_count() {
            self.sat_row_to_id.resize(self.matrix().row_count(), None);
        }
        if let Some(id) = self.sat_id_for_row(row) {
            return id;
        }
        let id = SatRowId::from(self.sat_id_to_row.len());
        self.sat_id_to_row.push(row);
        self.sat_row_to_id[row] = Some(id);
        id
    }

    #[inline(always)]
    pub fn row_value(&self, row: Row, ray: &[N]) -> N {
        linalg::dot(
            self.matrix().row(row).expect("row index within bounds"),
            ray,
        )
    }

    #[inline(always)]
    pub(crate) fn refresh_row_to_pos(&mut self) {
        let m = self.matrix().row_count();
        if self.row_to_pos.len() != m {
            self.row_to_pos = vec![m; m];
        }
        for slot in self.row_to_pos.iter_mut() {
            *slot = m;
        }
        for (pos, &row) in self.order_vector.iter().enumerate() {
            self.row_to_pos[row] = pos;
        }
    }
}

/// Umpire strategy controlling how rays are created and classified.
///
/// The DD core is generic over a `ZeroRepr` (saturation-set representation). Umpires should
/// remain representation-agnostic: they simply operate on whatever `ZeroRepr::Set` the core
/// selects.
pub trait Umpire<N: Num, ZR: ZeroRepr = crate::dd::SatRepr>: Sized {
    type Eps: Epsilon<N>;
    type Scalar: Clone + std::fmt::Debug;
    type MatrixData<R: Representation>: UmpireMatrix<N, R>;
    type RayData: crate::dd::ray::RayData<ZeroSet = <ZR as ZeroRepr>::Set>;
    type HalfspacePolicy: policies::HalfspacePolicy<N>;

    /// Convert a base matrix into the umpire-owned working representation.
    fn ingest<R: Representation>(&mut self, matrix: LpMatrix<N, R>) -> Self::MatrixData<R>;

    fn eps(&self) -> &Self::Eps;
    fn halfspace_policy(&mut self) -> &mut Self::HalfspacePolicy;

    fn zero_vector(&self, dim: usize) -> Vec<Self::Scalar>;

    fn basis_column_vector(&mut self, basis: &BasisMatrix<N>, col: usize) -> Vec<Self::Scalar>;

    fn normalize_vector(&mut self, vector: &mut Vec<Self::Scalar>) -> bool;

    fn negate_vector_in_place(&mut self, vector: &mut [Self::Scalar]);

    fn align_vector_in_place(&mut self, reference: &[Self::Scalar], candidate: &mut [Self::Scalar]);

    fn ray_vector_for_output(&self, ray_data: &Self::RayData) -> Vec<N>;

    /// Whether initial rays should be purified from their expected active rows.
    ///
    /// This is intended for snap-style modes that can cheaply reconstruct initial rays from the
    /// initial basis halfspaces (rather than relying on accumulated pivot arithmetic).
    #[inline(always)]
    fn wants_initial_purification(&self) -> bool {
        false
    }

    /// Optionally reconstruct a ray vector from a set of rows expected to evaluate to zero.
    ///
    /// Returning `None` indicates the umpire does not support this purification mode, or that the
    /// selected rows do not yield a 1D nullspace under the current numeric policy.
    #[inline(always)]
    fn purify_vector_from_zero_set<R: Representation>(
        &mut self,
        _cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        _expected_zero: &RowSet,
    ) -> Option<Vec<Self::Scalar>> {
        None
    }

    /// Decide whether two rays should be treated as duplicates.
    ///
    /// The DD core guarantees the two rays share the same zero set before calling this method.
    fn rays_equivalent(&mut self, a: &Self::RayData, b: &Self::RayData) -> bool;

    /// Remap a ray's vector after a column reduction.
    ///
    /// Implementations with shadow vectors/caches should update them here.
    fn remap_ray_after_column_reduction(
        &mut self,
        ray_data: &mut Self::RayData,
        mapping: &[Option<usize>],
        new_dim: usize,
    );

    /// Optional near-zero/ambiguous rows for `ray_data`.
    ///
    /// Umpires that consult a higher-precision shadow only for ambiguous evaluations can return
    /// the rows that required shadow evaluation here.
    #[inline(always)]
    fn near_zero_rows_on_ray<'a>(&self, _ray_data: &'a Self::RayData) -> Option<&'a [Row]> {
        None
    }

    /// Whether `near_zero_rows_on_ray` was truncated due to internal caps.
    #[inline(always)]
    fn near_zero_rows_truncated_on_ray(&self, _ray_data: &Self::RayData) -> bool {
        false
    }

    /// Matrix homogeneity check under the umpire's numeric policy.
    #[inline(always)]
    fn is_homogeneous<R: Representation>(&self, matrix: &LpMatrix<N, R>) -> bool {
        matrix.is_homogeneous(self.eps())
    }

    /// Compute/refresh `cone.order_vector` under this umpire's policy.
    fn recompute_row_order_vector<R: Representation>(
        &mut self,
        cone: &mut ConeCtx<N, R, Self::MatrixData<R>>,
        strict_rows: &RowSet,
    );

    /// Bump the given rows to the front of `order_vector`, preserving relative order.
    ///
    /// Direct migration of the existing `ConeEngine::update_row_order_vector` behavior.
    #[inline(always)]
    fn bump_priority_rows<R: Representation>(
        &mut self,
        cone: &mut ConeCtx<N, R, Self::MatrixData<R>>,
        priority_rows: &RowSet,
    ) {
        let m = cone.matrix().row_count();
        let count = priority_rows.cardinality();
        for target_pos in 0..count {
            if let Some(current_pos) =
                (target_pos..m).find(|pos| priority_rows.contains(cone.order_vector[*pos]))
                && current_pos > target_pos
            {
                let selected = cone.order_vector[current_pos];
                for k in (target_pos + 1..=current_pos).rev() {
                    cone.order_vector[k] = cone.order_vector[k - 1];
                }
                cone.order_vector[target_pos] = selected;
            }
        }
        cone.refresh_row_to_pos();
    }

    /// Choose the next halfspace row to add under this umpire's policy.
    fn choose_next_halfspace<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        excluded: &RowSet,
        iteration: Row,
        active_rays: usize,
    ) -> Option<Row>;

    /// Reset any infeasible-count bookkeeping used by the halfspace policy.
    #[inline(always)]
    fn reset_infeasible_counts(&mut self, row_count: Row) {
        self.halfspace_policy().reset_infeasible_counts(row_count);
    }

    /// Notify the umpire that a ray has been inserted (so any infeasible-count cache can be updated).
    fn on_ray_inserted<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        relaxed: bool,
    );

    /// Notify the umpire that a ray is about to be removed, so any infeasible-count cache can be updated.
    fn on_ray_removed<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        relaxed: bool,
    );

    /// Recycle internal allocations from a ray that is about to be dropped.
    ///
    /// Implementations may move out and retain buffers (vectors, saturation sets, etc.) to
    /// amortize allocation costs across the DD run.
    #[inline(always)]
    fn recycle_ray_data(&mut self, _ray_data: &mut Self::RayData) {}

    /// Compute an initial tableau basis and its pivot halfspaces under the umpire's numeric policy.
    ///
    /// This is called during the basis-prep phase. Implementations must:
    /// - fill `tableau.basis` with the computed basis matrix (dimension `d x d`), and
    /// - leave the tableau storage either in a consistent state for tracing, or empty/cleared.
    ///
    /// The returned `(initial_ray_index, initial_halfspaces)` seeds ray construction:
    /// - `initial_ray_index[col]` is the pivot row providing the initial ray for `col`, if any.
    /// - `initial_halfspaces` is the set of pivot rows.
    ///
    /// Exact/integer umpires should override this to avoid rational-division pivoting.
    fn compute_initial_basis<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        tableau: &mut BasisInitTableau<'_, N>,
        strict_rows: &RowSet,
        equality_set: &RowSet,
    ) -> (Vec<Option<Row>>, RowSet) {
        let row_count = cone.matrix().row_count();
        let col_count = cone.matrix().col_count();

        tableau.set_basis_identity(col_count);
        tableau.set_tableau_nonbasic_default(col_count);
        tableau.set_tableau_basic_col_for_row_len(row_count);
        tableau.reset_storage(row_count, col_count);

        if row_count > 0 && col_count > 0 {
            let matrix = cone.matrix().storage();
            for (row_idx, src) in matrix.iter().enumerate() {
                tableau.row_mut(row_idx).clone_from_slice(src);
            }
        }

        let mut initial_ray_index = vec![None; col_count];
        let mut initial_halfspaces = RowSet::new(row_count);

        let mut col_selected = ColSet::new(col_count);
        let mut nopivot_row = strict_rows.clone();

        let eps = self.eps();
        let mut rank = 0usize;

        loop {
            let mut row_excluded = nopivot_row.clone();

            let pivot = loop {
                let mut candidate = None;
                for i in 0..row_count {
                    if equality_set.contains(i) && !row_excluded.contains(i) {
                        candidate = Some(i);
                        break;
                    }
                }
                if candidate.is_none() {
                    for &row in &cone.order_vector {
                        if !row_excluded.contains(row) {
                            candidate = Some(row);
                            break;
                        }
                    }
                }
                let Some(r) = candidate else {
                    break None;
                };

                let row_slice = tableau.row(r);

                let mut s_opt = None;
                for col_id in col_selected.iter().complement() {
                    let s = col_id.as_index();
                    if !eps.is_zero(&row_slice[s]) {
                        s_opt = Some(s);
                        break;
                    }
                }

                let Some(s) = s_opt else {
                    row_excluded.insert(r);
                    continue;
                };

                tableau.apply_gaussian_column_pivot(r, s, eps);
                break Some((r, s));
            };

            let Some((r, s)) = pivot else {
                break;
            };

            nopivot_row.insert(r);
            col_selected.insert(s);
            initial_halfspaces.insert(r);
            initial_ray_index[s] = Some(r);

            rank += 1;
            if rank == col_count {
                break;
            }
        }

        (initial_ray_index, initial_halfspaces)
    }

    /// Compute matrix rank under the umpire's numeric policy.
    #[inline(always)]
    fn rank<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ignored_rows: &RowSet,
        ignored_cols: &ColSet,
    ) -> MatrixRank {
        cone.matrix()
            .storage()
            .rank(ignored_rows, ignored_cols, self.eps())
    }

    /// Feasibility of a restricted face under the umpire's numeric policy.
    #[inline(always)]
    fn restricted_face_exists<R: Representation>(
        &mut self,
        matrix: &LpMatrix<N, R>,
        equalities: &RowSet,
        strict_inequalities: &RowSet,
        solver: LpSolver,
    ) -> Result<bool, Error> {
        matrix.restricted_face_exists(equalities, strict_inequalities, solver, self.eps())
    }

    /// Compute the sign of a row evaluation on a ray **without** mutating any ray-local caches.
    ///
    /// This exists specifically for hot-loop scans (e.g. cutoff feasibility counts) where
    /// clobbering `RayClass.last_*` would hurt cache efficiency elsewhere.
    fn sign_for_row_on_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray: &Self::RayData,
        row: Row,
    ) -> Sign;

    /// Classify an existing ray against `row`, updating any cached evaluation in `ray_data`.
    fn classify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        row: Row,
    ) -> Sign;

    /// Fully classify a vector as a new ray, writing any needed sign sets into `sets_out`.
    ///
    /// `last_row` indicates a row whose evaluation should be cached on the ray (if provided).
    fn classify_vector<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        vector: Vec<Self::Scalar>,
        relaxed: bool,
        last_row: Option<Row>,
        zero_set: <ZR as ZeroRepr>::Set,
    ) -> Self::RayData;

    /// Compute sign sets for an existing ray under the umpire's numeric policy, writing into
    /// `sets_out` (typically used for infeasible-count bookkeeping).
    fn sign_sets_for_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        relaxed: bool,
        force_infeasible: bool,
        negative_out: &mut RowSet,
    );

    /// Recompute `first_infeasible_row` under the current row order.
    ///
    /// This must not rely on `RayClass.last_*` cache fields (which are used by Phase2's hot loop).
    fn update_first_infeasible_row<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        relaxed: bool,
    );

    /// Recompute ray feasibility + incidence (`zero_set`) under the current row order.
    ///
    /// Implementations should generally preserve the `RayClass.last_*` cache fields.
    fn reclassify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        relaxed: bool,
    );

    /// Construct a new ray from two parent rays at the intersection row.
    ///
    /// Returning `None` indicates the new ray should be discarded.
    fn generate_new_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        parents: (RayId, &Self::RayData, RayId, &Self::RayData),
        row: Row,
        relaxed: bool,
        zero_set: <ZR as ZeroRepr>::Set,
    ) -> Result<Self::RayData, <ZR as ZeroRepr>::Set>;
}
