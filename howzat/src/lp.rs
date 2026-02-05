use crate::HowzatError as Error;
use crate::matrix::{LpMatrix, Matrix};
use calculo::linalg;
use calculo::num::{CoerceFrom, Epsilon, Num};
use hullabaloo::types::{
    Col, ColId, ColSet, Representation, RepresentationKind, Row, RowId, RowIndex, RowSet,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LpObjective {
    None,
    Maximize,
    Minimize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LpSolver {
    CrissCross,
    DualSimplex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LpStatus {
    Undecided,
    Optimal,
    PivotLimitExceeded,
    PivotFailure,
    Inconsistent,
    DualInconsistent,
    StructuralInconsistent,
    StructuralDualInconsistent,
    Unbounded,
    DualUnbounded,
}

#[inline(always)]
pub(crate) fn compare_positive_ratios<N: Num>(
    num_a: &N,
    den_a: &N,
    num_b: &N,
    den_b: &N,
    eps: &impl Epsilon<N>,
) -> std::cmp::Ordering {
    debug_assert!(eps.is_positive(den_a) && eps.is_positive(den_b));
    let lhs = num_a.ref_mul(den_b);
    let rhs = num_b.ref_mul(den_a);
    eps.cmp(&lhs, &rhs)
}

#[derive(Clone, Debug)]
pub(crate) struct SmallMat<N: Num> {
    dim: usize,
    data: Vec<N>,
}

impl<N: Num> SmallMat<N> {
    pub(crate) fn new(dim: usize) -> Self {
        let alloc = dim.checked_mul(dim).expect("small matrix overflow");
        Self {
            dim,
            data: vec![N::zero(); alloc],
        }
    }

    pub(crate) fn identity(dim: usize) -> Self {
        let mut mat = Self::new(dim);
        for i in 0..dim {
            let idx = i * dim + i;
            mat.data[idx] = N::one();
        }
        mat
    }

    #[inline]
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        row * self.dim + col
    }

    pub(crate) fn get(&self, row: usize, col: usize) -> &N {
        let idx = self.index(row, col);
        &self.data[idx]
    }

    #[inline(always)]
    pub(crate) unsafe fn get_unchecked(&self, row: usize, col: usize) -> &N {
        debug_assert!(
            row < self.dim && col < self.dim,
            "smallmat index out of bounds"
        );
        let idx = self.index(row, col);
        unsafe { self.data.get_unchecked(idx) }
    }

    pub(crate) fn row_slice(&self, row: usize) -> &[N] {
        let start = row * self.dim;
        let end = start + self.dim;
        &self.data[start..end]
    }

    #[inline(always)]
    pub(crate) unsafe fn row_slice_unchecked(&self, row: usize) -> &[N] {
        debug_assert!(row < self.dim, "smallmat row out of bounds");
        let start = row * self.dim;
        let end = start + self.dim;
        unsafe { self.data.get_unchecked(start..end) }
    }

    pub(crate) fn invert<Eps: Epsilon<N>>(&self, eps: &Eps) -> Option<Self> {
        let dim = self.dim;
        let mut work = self.data.clone();
        let mut inv: Vec<N> = SmallMat::identity(dim).data;
        for col in 0..dim {
            let pivot_row = (col..dim).find(|&r| {
                let idx = r * dim + col;
                !eps.is_zero(&work[idx])
            })?;
            if pivot_row != col {
                for c in 0..dim {
                    work.swap(col * dim + c, pivot_row * dim + c);
                    inv.swap(col * dim + c, pivot_row * dim + c);
                }
            }
            let pivot_idx = col * dim + col;
            if eps.is_zero(&work[pivot_idx]) {
                return None;
            }
            let pivot = work[pivot_idx].clone();
            let inv_pivot = N::one().ref_div(&pivot);
            let pivot_start = col * dim;
            linalg::scale_assign(&mut work[pivot_start..pivot_start + dim], &inv_pivot);
            linalg::scale_assign(&mut inv[pivot_start..pivot_start + dim], &inv_pivot);
            for r in 0..dim {
                if r == col {
                    continue;
                }
                if eps.is_zero(&work[r * dim + col]) {
                    continue;
                }
                let factor = work[r * dim + col].clone();
                let row_start = r * dim;
                if row_start < pivot_start {
                    let (work_left, work_right) = work.split_at_mut(pivot_start);
                    let work_row = &mut work_left[row_start..row_start + dim];
                    let pivot_row = &work_right[..dim];
                    linalg::axpy_sub(work_row, &factor, pivot_row);

                    let (inv_left, inv_right) = inv.split_at_mut(pivot_start);
                    let inv_row = &mut inv_left[row_start..row_start + dim];
                    let inv_pivot_row = &inv_right[..dim];
                    linalg::axpy_sub(inv_row, &factor, inv_pivot_row);
                } else {
                    let (work_left, work_right) = work.split_at_mut(row_start);
                    let pivot_row = &work_left[pivot_start..pivot_start + dim];
                    let work_row = &mut work_right[..dim];
                    linalg::axpy_sub(work_row, &factor, pivot_row);

                    let (inv_left, inv_right) = inv.split_at_mut(row_start);
                    let inv_pivot_row = &inv_left[pivot_start..pivot_start + dim];
                    let inv_row = &mut inv_right[..dim];
                    linalg::axpy_sub(inv_row, &factor, inv_pivot_row);
                }
            }
        }
        Some(Self { dim, data: inv })
    }

    pub(crate) fn pivot_column<Eps: Epsilon<N>>(
        &mut self,
        tableau_row: &[N],
        pivot_col: usize,
        eps: &Eps,
    ) -> Result<(), ()> {
        debug_assert_eq!(tableau_row.len(), self.dim);
        if pivot_col >= self.dim {
            return Err(());
        }
        if eps.is_zero(&tableau_row[pivot_col]) {
            return Err(());
        }
        let dim = self.dim;
        let pivot = tableau_row[pivot_col].clone();
        let inv_pivot = N::one().ref_div(&pivot);
        for (c, value) in tableau_row.iter().enumerate() {
            if c == pivot_col {
                continue;
            }
            let factor = value.clone().ref_mul(&inv_pivot);
            if eps.is_zero(&factor) {
                continue;
            }
            for r in 0..dim {
                let idx = r * dim + c;
                let pivot_idx = r * dim + pivot_col;
                debug_assert!(pivot_idx < self.data.len(), "smallmat index out of bounds");
                let pivot_entry = unsafe { self.data.get_unchecked(pivot_idx).clone() };
                debug_assert!(idx < self.data.len(), "smallmat index out of bounds");
                let entry = unsafe { self.data.get_unchecked_mut(idx) };
                linalg::sub_mul_assign(entry, &factor, &pivot_entry);
            }
        }
        for r in 0..dim {
            let idx = r * dim + pivot_col;
            debug_assert!(idx < self.data.len(), "smallmat index out of bounds");
            let entry = unsafe { self.data.get_unchecked_mut(idx) };
            *entry = entry.ref_mul(&inv_pivot);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LpSnapshot<N: Num> {
    constraints: Matrix<N>,
    transform: SmallMat<N>,
    nonbasic_index: Vec<isize>,
    bflag: Vec<isize>,
    obj_row: usize,
    rhs_col: usize,
    basis_rows: Vec<usize>,
    equality_rows: RowSet,
    row_order: Vec<usize>,
}

impl<N: Num> LpSnapshot<N> {
    pub fn coerce_as<M>(&self) -> Result<LpSnapshot<M>, Error>
    where
        M: Num + CoerceFrom<N>,
    {
        let constraints = self
            .constraints
            .coerce_as::<M>()
            .map_err(|_| Error::ConversionFailure)?;

        let mut transform_data: Vec<M> = Vec::with_capacity(self.transform.data.len());
        for v in &self.transform.data {
            transform_data.push(M::coerce_from(v).map_err(|_| Error::ConversionFailure)?);
        }
        let transform = SmallMat {
            dim: self.transform.dim,
            data: transform_data,
        };
        Ok(LpSnapshot {
            constraints,
            transform,
            nonbasic_index: self.nonbasic_index.clone(),
            bflag: self.bflag.clone(),
            obj_row: self.obj_row,
            rhs_col: self.rhs_col,
            basis_rows: self.basis_rows.clone(),
            equality_rows: self.equality_rows.clone(),
            row_order: self.row_order.clone(),
        })
    }

    pub fn basis_rows(&self) -> &[usize] {
        &self.basis_rows
    }

    pub fn nonbasic_index(&self) -> &[isize] {
        &self.nonbasic_index
    }

    pub fn bflag(&self) -> &[isize] {
        &self.bflag
    }

    pub fn obj_row(&self) -> usize {
        self.obj_row
    }

    pub fn rhs_col(&self) -> usize {
        self.rhs_col
    }

    pub fn row_order(&self) -> &[usize] {
        &self.row_order
    }
}

#[derive(Clone, Debug)]
pub enum LpBasisStatusIssue<N: Num> {
    RankDeficient { rank: usize, expected: usize },
    BasisCardinalityMismatch { basic: usize, expected: usize },
    NonbasicIndexMismatch,
    ConversionFailed,
    PrimalInfeasibleRhs { row: usize, value: N },
    EqualityRowNotBasic { row: usize },
    EqualityRowOutOfRange { row: usize },
    DualInfeasibleReducedCost { row: usize, value: N },
    MissingSnapshot,
}

#[derive(Clone, Debug)]
pub struct LpBasisStatusResult<N: Num> {
    valid: bool,
    rank: usize,
    expected_rank: usize,
    issues: Vec<LpBasisStatusIssue<N>>,
    conversion_failed: bool,
    verification_pivots: u64,
    _marker: std::marker::PhantomData<N>,
}

impl<N: Num> LpBasisStatusResult<N> {
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn expected_rank(&self) -> usize {
        self.expected_rank
    }

    pub fn issues(&self) -> &[LpBasisStatusIssue<N>] {
        &self.issues
    }

    pub fn conversion_failed(&self) -> bool {
        self.conversion_failed
    }

    pub fn verification_pivots(&self) -> u64 {
        self.verification_pivots
    }
}

#[derive(Clone, Debug, Default)]
pub struct PivotCounts {
    pub setup: u64,
    pub phase_one: u64,
    pub phase_two: u64,
    pub criss_cross: u64,
    pub total: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PivotPhase {
    Setup,
    PhaseOne,
    PhaseTwo,
    CrissCross,
}

impl PivotCounts {
    fn bump(&mut self, phase: PivotPhase) {
        self.total += 1;
        match phase {
            PivotPhase::Setup => self.setup += 1,
            PivotPhase::PhaseOne => self.phase_one += 1,
            PivotPhase::PhaseTwo => self.phase_two += 1,
            PivotPhase::CrissCross => self.criss_cross += 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LpResult<N: Num> {
    objective: LpObjective,
    solver: LpSolver,
    rows: Row,
    cols: Col,
    homogeneous: bool,
    representation: RepresentationKind,
    status: LpStatus,
    optimal_value: N,
    primal: Vec<N>,
    dual: Vec<N>,
    nonbasic_index: RowIndex,
    row_certificate: Option<Row>,
    col_certificate: Option<Col>,
    pivots: PivotCounts,
    equality_rows: RowSet,
    snapshot: Option<LpSnapshot<N>>,
}

impl<N: Num> LpResult<N> {
    pub fn objective(&self) -> LpObjective {
        self.objective
    }

    pub fn solver(&self) -> LpSolver {
        self.solver
    }

    pub fn row_count(&self) -> Row {
        self.rows
    }

    pub fn col_count(&self) -> Col {
        self.cols
    }

    pub fn homogeneous(&self) -> bool {
        self.homogeneous
    }

    pub fn representation(&self) -> RepresentationKind {
        self.representation
    }

    pub fn status(&self) -> LpStatus {
        self.status
    }

    pub fn optimal_value(&self) -> &N {
        &self.optimal_value
    }

    pub fn primal(&self) -> &[N] {
        &self.primal
    }

    pub fn dual(&self) -> &[N] {
        &self.dual
    }

    pub fn nonbasic_index(&self) -> &RowIndex {
        &self.nonbasic_index
    }

    pub fn row_certificate(&self) -> Option<Row> {
        self.row_certificate
    }

    pub fn col_certificate(&self) -> Option<Col> {
        self.col_certificate
    }

    pub fn pivots(&self) -> &PivotCounts {
        &self.pivots
    }

    pub fn equality_rows(&self) -> &RowSet {
        &self.equality_rows
    }

    pub fn snapshot(&self) -> Option<&LpSnapshot<N>> {
        self.snapshot.as_ref()
    }

    pub fn snapshot_as<M>(&self) -> Result<Option<LpSnapshot<M>>, Error>
    where
        M: Num + CoerceFrom<N>,
    {
        self.snapshot
            .as_ref()
            .map(LpSnapshot::coerce_as)
            .transpose()
    }

    pub fn basis_status(&self, eps: &impl Epsilon<N>) -> LpBasisStatusResult<N> {
        self.verify_basis_status_internal::<N>(eps)
    }

    pub fn basis_status_as<M>(&self, eps: &impl Epsilon<M>) -> LpBasisStatusResult<M>
    where
        M: Num + CoerceFrom<N>,
    {
        self.verify_basis_status_internal::<M>(eps)
    }

    pub(crate) fn verify_basis_status_internal<M>(
        &self,
        eps: &impl Epsilon<M>,
    ) -> LpBasisStatusResult<M>
    where
        M: Num + CoerceFrom<N>,
    {
        let Some(snapshot) = &self.snapshot else {
            return LpBasisStatusResult {
                valid: false,
                rank: 0,
                expected_rank: 0,
                issues: vec![LpBasisStatusIssue::MissingSnapshot],
                conversion_failed: false,
                verification_pivots: 0,
                _marker: std::marker::PhantomData,
            };
        };

        let mut verification_pivots = 0u64;
        let mut issues = Vec::new();
        let converted = match snapshot.coerce_as::<M>() {
            Ok(s) => s,
            Err(_) => {
                issues.push(LpBasisStatusIssue::ConversionFailed);
                return LpBasisStatusResult {
                    valid: false,
                    rank: 0,
                    expected_rank: snapshot.transform.dim(),
                    issues,
                    conversion_failed: true,
                    verification_pivots,
                    _marker: std::marker::PhantomData,
                };
            }
        };

        let expected_rank = converted.transform.dim();
        let mut conversion_error = false;

        let invertible = converted.transform.invert(eps).is_some();
        let rank = if invertible {
            verification_pivots = expected_rank as u64;
            expected_rank
        } else {
            issues.push(LpBasisStatusIssue::RankDeficient {
                rank: expected_rank.saturating_sub(1),
                expected: expected_rank,
            });
            expected_rank.saturating_sub(1)
        };

        let mut nb_consistent = true;
        if converted.nonbasic_index.len() != expected_rank
            || converted.bflag.len() < converted.constraints.row_count()
        {
            nb_consistent = false;
        }
        if converted
            .nonbasic_index
            .get(converted.rhs_col)
            .copied()
            .unwrap_or(-1)
            != 0
        {
            nb_consistent = false;
        }
        for (row_idx, flag) in converted.bflag.iter().enumerate() {
            if *flag < 0 {
                continue;
            }
            let col = *flag as usize;
            if col == converted.rhs_col && row_idx == converted.obj_row {
                continue;
            }
            if col >= converted.nonbasic_index.len() {
                nb_consistent = false;
                break;
            }
            let mapped = converted.nonbasic_index[col];
            if mapped <= 0 || mapped as usize != row_idx.saturating_add(1) {
                nb_consistent = false;
                break;
            }
        }
        for (col_idx, &row_val) in converted.nonbasic_index.iter().enumerate() {
            if row_val <= 0 {
                continue;
            }
            let row_idx = row_val.saturating_sub(1) as usize;
            if row_idx >= converted.bflag.len() {
                nb_consistent = false;
                break;
            }
            let mapped = converted.bflag[row_idx];
            if mapped <= 0 || mapped as usize != col_idx {
                nb_consistent = false;
                break;
            }
        }
        if !nb_consistent {
            issues.push(LpBasisStatusIssue::NonbasicIndexMismatch);
        }

        let basic_rows: Vec<usize> = (0..converted.constraints.row_count())
            .filter(|r| {
                *r != converted.obj_row && converted.bflag.get(*r).copied().unwrap_or(-1) == -1
            })
            .collect();
        for row_id in converted.equality_rows.iter() {
            let idx: usize = row_id.into();
            if idx >= converted.constraints.row_count() {
                issues.push(LpBasisStatusIssue::EqualityRowOutOfRange { row: idx });
            } else if converted.bflag.get(idx).copied().unwrap_or(-1) != -1 {
                issues.push(LpBasisStatusIssue::EqualityRowNotBasic { row: idx });
            }
        }

        if matches!(self.status, LpStatus::Optimal) {
            for &row_idx in &basic_rows {
                if row_idx >= converted.constraints.row_count() {
                    conversion_error = true;
                    continue;
                }
                let mut acc = M::zero();
                if let Some(coeffs) = converted.constraints.row(row_idx) {
                    for (a, t) in coeffs.iter().zip(
                        (0..converted.transform.dim())
                            .map(|c| converted.transform.get(c, converted.rhs_col)),
                    ) {
                        linalg::add_mul_assign(&mut acc, a, t);
                    }
                    if eps.is_negative(&acc)
                        && !converted.equality_rows.contains(RowId::new(row_idx))
                    {
                        issues.push(LpBasisStatusIssue::PrimalInfeasibleRhs {
                            row: row_idx,
                            value: acc,
                        });
                    }
                    verification_pivots = verification_pivots.saturating_add(1);
                } else {
                    conversion_error = true;
                }
            }

            for (row_idx, flag) in converted.bflag.iter().enumerate() {
                if row_idx == converted.obj_row || *flag <= 0 {
                    continue;
                }
                let mut acc = M::zero();
                let Some(coeffs) = converted.constraints.row(converted.obj_row) else {
                    conversion_error = true;
                    continue;
                };
                for (a, t) in coeffs.iter().zip(
                    (0..converted.transform.dim())
                        .map(|c| converted.transform.get(c, *flag as usize)),
                ) {
                    linalg::add_mul_assign(&mut acc, a, t);
                }
                let violates = match self.objective {
                    LpObjective::Minimize => eps.is_negative(&acc),
                    LpObjective::Maximize | LpObjective::None => eps.is_positive(&acc),
                };
                if violates {
                    issues.push(LpBasisStatusIssue::DualInfeasibleReducedCost {
                        row: *flag as usize,
                        value: acc,
                    });
                }
                verification_pivots = verification_pivots.saturating_add(1);
            }
        }

        LpBasisStatusResult {
            valid: issues.is_empty() && !conversion_error,
            rank,
            expected_rank,
            issues,
            conversion_failed: conversion_error,
            verification_pivots,
            _marker: std::marker::PhantomData,
        }
    }
}

pub type LpSolution<N> = LpResult<N>;

#[derive(Clone, Debug)]
pub struct LpProblem<N: Num> {
    objective: LpObjective,
    homogeneous: bool,
    rows: Row,
    cols: Col,
    a: Matrix<N>,
    objective_row: Row,
    representation: RepresentationKind,
    lexicographic_pivot: bool,
    max_pivots: Option<usize>,
    row_is_equality: Vec<bool>,
    solve_dual: bool,
}

impl<N: Num> LpProblem<N> {
    pub fn from_matrix<R: Representation>(
        matrix: &LpMatrix<N, R>,
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        let linc = matrix.linearity().cardinality();
        let base_rows = matrix.row_count();
        assert!(
            base_rows < usize::MAX - linc,
            "row count overflow while building LP"
        );
        let m = base_rows + 1 + linc;
        let d = matrix.col_count();
        let mut lp = LpProblem::new(matrix.objective(), m, d);
        lp.representation = matrix.representation();
        assert!(m > 0, "LP must include an objective row");
        lp.objective_row = m - 1;
        lp.homogeneous = true;
        lp.row_is_equality = vec![false; m];

        let mut irev = matrix.row_count();
        for (i, row) in matrix.rows().enumerate() {
            if matrix.linearity().contains(i) {
                irev += 1;
                let dual_row = lp.a.row_mut(irev - 1).expect("dual row in bounds");
                for (j, val) in row.iter().enumerate() {
                    dual_row[j] = -val.clone();
                }
                if i < lp.row_is_equality.len() {
                    lp.row_is_equality[i] = true;
                }
            }
            let lp_row = lp.a.row_mut(i).expect("lp row in bounds");
            for (j, val) in row.iter().enumerate() {
                lp_row[j] = val.clone();
                if j == 0 && !eps.is_zero(val) {
                    lp.homogeneous = false;
                }
            }
        }
        assert_eq!(
            matrix.row_vec().len(),
            d,
            "LP row_vec length mismatch (row_vec={} cols={})",
            matrix.row_vec().len(),
            d
        );
        {
            let obj_row = lp.a.row_mut(m - 1).expect("objective row in bounds");
            for (j, val) in matrix.row_vec().iter().enumerate() {
                obj_row[j] = val.clone();
            }
        }
        if lp.row_count() > 1 && lp.col_count() > lp.row_count() {
            lp.dualize_layout()?;
        }
        Ok(lp)
    }

    pub fn from_feasibility<R: Representation>(
        matrix: &LpMatrix<N, R>,
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        let mut lp = LpProblem::from_matrix(matrix, eps)?;
        lp.objective = LpObjective::Maximize;
        assert!(lp.row_count() > 0, "LP must include an objective row");
        lp.objective_row = lp.row_count() - 1;
        let obj_row =
            lp.a.row_mut(lp.objective_row)
                .expect("objective row in bounds");
        for value in obj_row {
            *value = N::zero();
        }
        if lp.objective_row < lp.row_is_equality.len() {
            lp.row_is_equality[lp.objective_row] = false;
        }
        Ok(lp)
    }

    pub fn from_redundancy_h<R: Representation>(
        matrix: &LpMatrix<N, R>,
        itest: Row,
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        assert!(
            itest < matrix.row_count(),
            "redundancy test row out of bounds (row={itest} rows={})",
            matrix.row_count()
        );
        let linc = matrix.linearity().cardinality();
        let base_rows = matrix.row_count();
        assert!(
            base_rows < usize::MAX - linc,
            "row count overflow while building redundancy test LP"
        );
        let m = base_rows + linc + 1;
        let d = matrix.col_count();
        let mut lp = LpProblem::new(LpObjective::Minimize, m, d);
        lp.representation = RepresentationKind::Inequality;
        assert!(m > 0, "LP must include an objective row");
        lp.objective_row = m - 1;
        lp.homogeneous = true;
        lp.row_is_equality = vec![false; m];

        let mut irev = matrix.row_count();
        for (i, row) in matrix.rows().enumerate() {
            let idx = i;
            if matrix.linearity().contains(idx) {
                irev += 1;
                let dual_row = lp.a.row_mut(irev - 1).expect("dual row in bounds");
                for (j, val) in row.iter().enumerate() {
                    dual_row[j] = -val.clone();
                }
                if i < lp.row_is_equality.len() {
                    lp.row_is_equality[i] = true;
                }
            }
            let lp_row = lp.a.row_mut(i).expect("lp row in bounds");
            for (j, val) in row.iter().enumerate() {
                lp_row[j] = val.clone();
                if j == 0 && !eps.is_zero(val) {
                    lp.homogeneous = false;
                }
            }
        }
        {
            let obj_row = lp.a.row_mut(m - 1).expect("objective row in bounds");
            for (j, val) in matrix
                .row(itest)
                .expect("test row in bounds")
                .iter()
                .enumerate()
            {
                obj_row[j] = val.clone();
            }
        }
        {
            let row = lp.a.row_mut(itest).expect("test row in bounds");
            row[0] = row[0].clone() + N::one();
        }
        Ok(lp)
    }

    pub fn from_redundancy_v<R: Representation>(
        matrix: &LpMatrix<N, R>,
        itest: Row,
    ) -> Result<Self, Error> {
        assert!(
            itest < matrix.row_count(),
            "redundancy test row out of bounds (row={itest} rows={})",
            matrix.row_count()
        );
        let linc = matrix.linearity().cardinality();
        let base_rows = matrix.row_count();
        assert!(
            base_rows < usize::MAX - linc,
            "row count overflow while building generator redundancy LP"
        );
        let m = base_rows + linc + 1;
        let col_count = matrix.col_count();
        assert!(
            col_count < usize::MAX,
            "column count overflow while building generator redundancy LP"
        );
        let d = col_count + 1;
        let mut lp = LpProblem::new(LpObjective::Minimize, m, d);
        lp.representation = RepresentationKind::Generator;
        assert!(m > 0, "LP must include an objective row");
        lp.objective_row = m - 1;
        lp.homogeneous = false;
        lp.row_is_equality = vec![false; m];

        let mut irev = matrix.row_count();
        for (i, row) in matrix.rows().enumerate() {
            let idx = i;
            if matrix.linearity().contains(idx) {
                irev += 1;
                let dual_row = lp.a.row_mut(irev - 1).expect("dual row in bounds");
                for (j, val) in row.iter().enumerate() {
                    dual_row[j + 1] = -val.clone();
                }
                if i < lp.row_is_equality.len() {
                    lp.row_is_equality[i] = true;
                }
            }
            let lp_row = lp.a.row_mut(i).expect("lp row in bounds");
            lp_row[0] = if idx == itest { N::one() } else { N::zero() };
            lp_row[1..d].clone_from_slice(row);
        }
        {
            let obj_row = lp.a.row_mut(m - 1).expect("objective row in bounds");
            obj_row[0] = N::zero();
            for (j, val) in matrix
                .row(itest)
                .expect("test row in bounds")
                .iter()
                .enumerate()
            {
                obj_row[j + 1] = val.clone();
            }
        }
        Ok(lp)
    }

    pub fn from_strong_redundancy_v<R: Representation>(
        matrix: &LpMatrix<N, R>,
        itest: Row,
    ) -> Result<Self, Error> {
        assert!(
            itest < matrix.row_count(),
            "strong redundancy test row out of bounds (row={itest} rows={})",
            matrix.row_count()
        );
        let linc = matrix.linearity().cardinality();
        let base_rows = matrix.row_count();
        assert!(
            base_rows <= usize::MAX - linc - 3,
            "row count overflow while building strong redundancy LP"
        );
        let m = base_rows + linc + 3;
        let col_count = matrix.col_count();
        assert!(
            col_count < usize::MAX,
            "column count overflow while building strong redundancy LP"
        );
        let d = col_count + 1;
        let mut lp = LpProblem::new(LpObjective::Maximize, m, d);
        lp.representation = RepresentationKind::Generator;
        assert!(m > 0, "LP must include an objective row");
        lp.objective_row = m - 1;
        lp.homogeneous = false;
        lp.row_is_equality = vec![false; m];

        let mut irev = matrix.row_count();
        let mut objective_sum = vec![N::zero(); d];
        for (i, row) in matrix.rows().enumerate() {
            let idx = i;
            if matrix.linearity().contains(idx) || idx == itest {
                irev += 1;
                let dual_row = lp.a.row_mut(irev - 1).expect("dual row in bounds");
                for j in 1..d {
                    dual_row[j] = -row[j - 1].clone();
                }
                if i < lp.row_is_equality.len() {
                    lp.row_is_equality[i] = true;
                }
            }
            let lp_row = lp.a.row_mut(i).expect("lp row in bounds");
            lp_row[0] = N::zero();
            for j in 1..d {
                let val = row[j - 1].clone();
                lp_row[j] = val.clone();
                objective_sum[j] = objective_sum[j].clone() + val;
            }
        }

        {
            let obj_row = lp.a.row_mut(m - 1).expect("objective row in bounds");
            obj_row.clone_from_slice(&objective_sum);
        }
        {
            let aux_row = lp.a.row_mut(m - 2).expect("aux row in bounds");
            aux_row[0] = N::one();
            for j in 1..d {
                aux_row[j] = -objective_sum[j].clone();
            }
        }

        Ok(lp)
    }

    pub fn from_feasibility_restricted<R: Representation>(
        matrix: &LpMatrix<N, R>,
        equalities: &RowSet,
        strict_inequalities: &RowSet,
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        let mut lin_union = RowSet::new(matrix.row_count());
        for i in matrix.linearity().iter() {
            lin_union.insert(i);
        }
        for i in equalities.iter() {
            lin_union.insert(i);
        }
        let linc = lin_union.cardinality();
        let base_rows = matrix.row_count();
        assert!(
            base_rows <= usize::MAX - linc - 2,
            "row count overflow while building restricted feasibility LP"
        );
        let m = base_rows + linc + 2;
        let col_count = matrix.col_count();
        assert!(
            col_count < usize::MAX,
            "column count overflow while building restricted feasibility LP"
        );
        let d = col_count + 1;
        let mut lp = LpProblem::new(LpObjective::Maximize, m, d);
        lp.representation = RepresentationKind::Inequality;
        assert!(m > 0, "LP must include an objective row");
        lp.objective_row = m - 1;
        lp.homogeneous = true;
        lp.row_is_equality = vec![false; m];

        let mut irev = matrix.row_count();
        for (i, row) in matrix.rows().enumerate() {
            let idx = i;
            let in_lin_union = lin_union.contains(idx);
            if in_lin_union {
                irev += 1;
                let dual_row = lp.a.row_mut(irev - 1).expect("dual row in bounds");
                for (j, val) in row.iter().enumerate() {
                    dual_row[j] = -val.clone();
                }
                if i < lp.row_is_equality.len() {
                    lp.row_is_equality[i] = true;
                }
            }

            let lp_row = lp.a.row_mut(i).expect("lp row in bounds");
            if !in_lin_union && strict_inequalities.contains(idx) {
                lp_row[d - 1] = -N::one();
            }
            for (j, val) in row.iter().enumerate() {
                lp_row[j] = val.clone();
                if j == 0 && !eps.is_zero(val) {
                    lp.homogeneous = false;
                }
            }
        }
        {
            let row = lp.a.row_mut(m - 2).expect("aux row in bounds");
            row[0] = N::one();
            row[d - 1] = -N::one();
        }
        lp.a.row_mut(m - 1).expect("objective row in bounds")[d - 1] = N::one();
        Ok(lp)
    }

    pub fn from_implicit_linearity_h<R: Representation>(
        matrix: &LpMatrix<N, R>,
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        let linc = matrix.linearity().cardinality();
        let base_rows = matrix.row_count();
        assert!(
            base_rows <= usize::MAX - linc - 2,
            "row count overflow while building implicit linearity LP"
        );
        let m = base_rows + linc + 2;
        let col_count = matrix.col_count();
        assert!(
            col_count < usize::MAX,
            "column count overflow while building implicit linearity LP"
        );
        let d = col_count + 1;
        let mut lp = LpProblem::new(LpObjective::Maximize, m, d);
        lp.representation = RepresentationKind::Inequality;
        assert!(m > 0, "LP must include an objective row");
        lp.objective_row = m - 1;
        lp.homogeneous = true;
        lp.row_is_equality = vec![false; m];

        let mut irev = matrix.row_count();
        for (i, row) in matrix.rows().enumerate() {
            let idx = i;
            let is_linear = matrix.linearity().contains(idx);
            if is_linear {
                irev += 1;
                let dual_row = lp.a.row_mut(irev - 1).expect("dual row in bounds");
                for (j, val) in row.iter().enumerate() {
                    dual_row[j] = -val.clone();
                }
                if i < lp.row_is_equality.len() {
                    lp.row_is_equality[i] = true;
                }
            }
            let lp_row = lp.a.row_mut(i).expect("lp row in bounds");
            if !is_linear {
                lp_row[d - 1] = -N::one();
            }
            for (j, val) in row.iter().enumerate() {
                lp_row[j] = val.clone();
                if j == 0 && !eps.is_zero(val) {
                    lp.homogeneous = false;
                }
            }
        }

        {
            let row = lp.a.row_mut(m - 2).expect("aux row in bounds");
            row[0] = N::one();
            row[d - 1] = -N::one();
        }
        lp.a.row_mut(m - 1).expect("objective row in bounds")[d - 1] = N::one();
        Ok(lp)
    }

    pub fn from_implicit_linearity_v<R: Representation>(
        matrix: &LpMatrix<N, R>,
    ) -> Result<Self, Error> {
        let linc = matrix.linearity().cardinality();
        let base_rows = matrix.row_count();
        assert!(
            base_rows <= usize::MAX - linc - 2,
            "row count overflow while building implicit linearity generator LP"
        );
        let m = base_rows + linc + 2;
        let col_count = matrix.col_count();
        assert!(
            col_count <= usize::MAX - 2,
            "column count overflow while building implicit linearity generator LP"
        );
        let d = col_count + 2;
        let mut lp = LpProblem::new(LpObjective::Maximize, m, d);
        lp.representation = RepresentationKind::Generator;
        assert!(m > 0, "LP must include an objective row");
        lp.objective_row = m - 1;
        lp.homogeneous = false;
        lp.row_is_equality = vec![false; m];

        let mut irev = matrix.row_count();
        for (i, row) in matrix.rows().enumerate() {
            let idx = i;
            let is_linear = matrix.linearity().contains(idx);
            if is_linear {
                irev += 1;
                let dual_row = lp.a.row_mut(irev - 1).expect("dual row in bounds");
                for (j, val) in row.iter().enumerate() {
                    dual_row[j + 1] = -val.clone();
                }
                if i < lp.row_is_equality.len() {
                    lp.row_is_equality[i] = true;
                }
            }

            let lp_row = lp.a.row_mut(i).expect("lp row in bounds");
            lp_row[0] = N::zero();
            if !is_linear {
                lp_row[d - 1] = -N::one();
            }
            for (j, val) in row.iter().enumerate() {
                lp_row[j + 1] = val.clone();
            }
        }
        {
            let row = lp.a.row_mut(m - 2).expect("aux row in bounds");
            row[0] = N::one();
            row[d - 1] = -N::one();
        }
        lp.a.row_mut(m - 1).expect("objective row in bounds")[d - 1] = N::one();
        Ok(lp)
    }

    pub fn new(objective: LpObjective, rows: Row, cols: Col) -> Self {
        assert!(cols > 0, "LP must include at least one column");
        let a = Matrix::new(rows, cols);
        assert!(rows > 0, "LP must include at least one row");
        Self {
            objective,
            homogeneous: true,
            rows,
            cols,
            a,
            objective_row: rows - 1,
            representation: RepresentationKind::Inequality,
            lexicographic_pivot: true,
            max_pivots: None,
            row_is_equality: vec![false; rows],
            solve_dual: false,
        }
    }

    pub fn with_reversed_row(mut self, row: Row) -> Result<Self, Error> {
        assert!(
            row < self.row_count(),
            "row out of bounds (row={row} rows={})",
            self.row_count()
        );
        for value in self.a.row_mut(row).expect("row in bounds") {
            *value = -value.clone();
        }
        Ok(self)
    }

    pub fn with_replaced_row(mut self, row: Row, values: Vec<N>) -> Result<Self, Error> {
        assert!(
            row < self.row_count(),
            "row out of bounds (row={row} rows={})",
            self.row_count()
        );
        assert_eq!(
            values.len(),
            self.col_count(),
            "row length mismatch (len={} cols={})",
            values.len(),
            self.col_count()
        );
        self.a
            .row_mut(row)
            .expect("row in bounds")
            .clone_from_slice(&values);
        Ok(self)
    }

    pub fn row_values(&self, row: Row) -> Result<Vec<N>, Error> {
        assert!(
            row < self.row_count(),
            "row out of bounds (row={row} rows={})",
            self.row_count()
        );
        Ok(self.a.row(row).expect("row in bounds").to_vec())
    }

    pub fn coefficients(&self) -> &Matrix<N> {
        &self.a
    }

    pub fn row_count(&self) -> Row {
        self.rows
    }

    pub fn col_count(&self) -> Col {
        self.cols
    }

    pub fn with_objective(mut self, objective: LpObjective) -> Self {
        self.objective = objective;
        self
    }

    pub fn with_max_pivots(mut self, max_pivots: Option<usize>) -> Self {
        self.max_pivots = max_pivots;
        self
    }

    pub fn with_lexicographic_pivot(mut self, enabled: bool) -> Self {
        self.lexicographic_pivot = enabled;
        self
    }

    fn dualize_layout(&mut self) -> Result<(), Error> {
        let orig_cols = self.a.col_count();
        let orig_rows = self.a.row_count();
        assert!(
            self.objective_row < orig_rows,
            "objective_row out of bounds (objective_row={} rows={})",
            self.objective_row,
            orig_rows
        );
        let constraints: Vec<&[N]> = (0..orig_rows)
            .filter(|&idx| idx != self.objective_row)
            .map(|idx| self.a.row(idx).expect("constraint row in bounds"))
            .collect();
        assert!(
            !constraints.is_empty(),
            "dualize_layout requires at least one constraint row"
        );

        let dual_cols = constraints.len().saturating_add(1);
        let dual_rows = orig_cols;
        let mut dual_matrix = Matrix::new(dual_rows.saturating_add(1), dual_cols);
        let obj_row = self
            .a
            .row(self.objective_row)
            .expect("objective row in bounds");

        for j in 0..orig_cols {
            let dual_row = dual_matrix.row_mut(j).expect("dual row in bounds");
            dual_row[0] = obj_row[j].clone();
            for (k, constraint) in constraints.iter().enumerate() {
                dual_row[k + 1] = constraint[j].clone();
            }
        }
        let dual_objective_row = dual_matrix
            .row_mut(dual_rows)
            .expect("dual objective row in bounds");
        for (k, constraint) in constraints.iter().enumerate() {
            dual_objective_row[k + 1] = constraint[0].clone();
        }

        self.a = dual_matrix;
        self.rows = dual_rows.saturating_add(1);
        self.cols = dual_cols;
        self.objective_row = dual_rows;
        self.row_is_equality = vec![true; dual_rows.saturating_add(1)];
        self.row_is_equality[self.objective_row] = false;
        self.homogeneous = false;
        self.solve_dual = true;
        Ok(())
    }

    pub fn make_interior_finding(&self) -> LpProblem<N> {
        self.clone().with_objective(LpObjective::Maximize)
    }

    pub fn solve(self, solver: LpSolver, eps: &impl Epsilon<N>) -> LpResult<N> {
        let mut engine = LpEngine::from_problem(self, eps);
        engine.solve(solver, false)
    }

    pub fn coerce_as<M>(&self) -> Result<LpProblem<M>, Error>
    where
        M: Num + CoerceFrom<N>,
    {
        let a = self
            .a
            .coerce_as::<M>()
            .map_err(|_| Error::ConversionFailure)?;
        Ok(LpProblem {
            objective: self.objective,
            homogeneous: self.homogeneous,
            rows: self.rows,
            cols: self.cols,
            a,
            objective_row: self.objective_row,
            representation: self.representation,
            lexicographic_pivot: self.lexicographic_pivot,
            max_pivots: self.max_pivots,
            row_is_equality: self.row_is_equality.clone(),
            solve_dual: self.solve_dual,
        })
    }

    pub fn solve_then_resolve_as<M>(
        &self,
        solver: LpSolver,
        eps: &impl Epsilon<N>,
        eps_m: &impl Epsilon<M>,
    ) -> Result<(LpResult<N>, LpResult<M>), Error>
    where
        M: Num + CoerceFrom<N>,
        N: Clone,
    {
        let primary = self.clone().solve_with_snapshot(solver, eps);
        let snapshot = primary
            .snapshot()
            .expect("LP snapshot missing after solve_with_snapshot")
            .coerce_as::<M>()?;
        let rerun_problem = self.coerce_as::<M>()?;
        let rerun = rerun_problem.solve_with_basis(solver, &snapshot, eps_m);
        Ok((primary, rerun))
    }

    pub fn solve_with_snapshot(self, solver: LpSolver, eps: &impl Epsilon<N>) -> LpResult<N> {
        let mut engine = LpEngine::from_problem(self, eps);
        engine.solve(solver, true)
    }

    pub fn solve_with_basis(
        self,
        solver: LpSolver,
        snapshot: &LpSnapshot<N>,
        eps: &impl Epsilon<N>,
    ) -> LpResult<N> {
        let mut engine = LpEngine::from_problem_with_basis(self, eps, snapshot);
        engine.solve(solver, true)
    }
}

#[derive(Clone, Debug)]
struct LpEngine<'a, N: Num, Eps: Epsilon<N>> {
    problem: LpProblem<N>,
    eps: &'a Eps,
    constraints: Matrix<N>,
    obj_row: usize,
    rhs_col: usize,
    row_order: Vec<usize>,
    equality_rows: RowSet,
    transform: SmallMat<N>,
    nbindex: Vec<isize>,
    nbindex_ref: Option<Vec<isize>>,
    bflag: Vec<isize>,
    pivots: PivotCounts,
    seeded: bool,
    scratch_obj_row: Vec<N>,
    scratch_pivot_row: Vec<N>,
    scratch_pivot_row_for: usize,
    scratch_aux_row: Vec<N>,
    scratch_ties: ColSet,
    scratch_working: ColSet,
    scratch_next: ColSet,
}

#[derive(Clone, Debug)]
struct PivotChoice {
    status: LpStatus,
    pivot: Option<(usize, usize)>,
    row_evidence: Option<usize>,
    col_evidence: Option<usize>,
}

impl<'a, N: Num, Eps: Epsilon<N>> LpEngine<'a, N, Eps> {
    fn from_problem(mut problem: LpProblem<N>, eps: &'a Eps) -> Self {
        let (constraints, obj_row, rhs_col, equality_rows) = Self::build_constraints(&mut problem);
        let mut lp = Self {
            problem,
            eps,
            constraints,
            obj_row,
            rhs_col,
            row_order: Vec::new(),
            equality_rows,
            transform: SmallMat::new(0),
            nbindex: Vec::new(),
            nbindex_ref: None,
            bflag: Vec::new(),
            pivots: PivotCounts::default(),
            seeded: false,
            scratch_obj_row: Vec::new(),
            scratch_pivot_row: Vec::new(),
            scratch_pivot_row_for: usize::MAX,
            scratch_aux_row: Vec::new(),
            scratch_ties: ColSet::new(0),
            scratch_working: ColSet::new(0),
            scratch_next: ColSet::new(0),
        };
        lp.reset_tableau(false);
        lp
    }

    fn from_problem_with_basis(problem: LpProblem<N>, eps: &'a Eps, seed: &LpSnapshot<N>) -> Self {
        let mut lp = Self::from_problem(problem, eps);
        lp.seed_from_snapshot(seed);
        lp
    }

    fn seed_from_snapshot(&mut self, seed: &LpSnapshot<N>) -> bool {
        if seed.constraints.col_count() != self.constraints.col_count()
            || seed.constraints.row_count() != self.constraints.row_count()
            || seed.obj_row != self.obj_row
            || seed.rhs_col != self.rhs_col
        {
            return false;
        }
        self.transform = seed.transform.clone();
        self.nbindex = seed.nonbasic_index.clone();
        self.nbindex_ref = Some(seed.nonbasic_index.clone());
        self.bflag = seed.bflag.clone();
        self.equality_rows = seed.equality_rows.clone();
        self.row_order = seed.row_order.clone();
        self.rebuild_row_order();
        self.seeded = true;
        true
    }

    fn build_constraints(problem: &mut LpProblem<N>) -> (Matrix<N>, usize, usize, RowSet) {
        // Equalities are already expanded during LP construction; only the original equality rows
        // belong to the equality set, matching cddlib.

        let objective_row = problem.objective_row;

        let mut a = std::mem::replace(&mut problem.a, Matrix::new(0, 0));
        let row_is_equality = std::mem::take(&mut problem.row_is_equality);
        assert_eq!(
            a.row_count(),
            row_is_equality.len(),
            "row_is_equality must align with LP rows"
        );

        let row_count = a.row_count();
        assert!(objective_row < row_count, "objective row out of range");

        let mut order = Vec::with_capacity(row_count);
        for idx in 0..row_count {
            if idx != objective_row {
                order.push(idx);
            }
        }
        order.push(objective_row);
        a.reorder_rows_by_order(&order);

        let mut eq = RowSet::new(row_count);
        for (idx, &is_eq) in row_is_equality.iter().enumerate() {
            if !is_eq || idx == objective_row {
                continue;
            }
            let new_idx = if idx < objective_row { idx } else { idx - 1 };
            eq.insert(new_idx);
        }

        let obj_row = row_count - 1;
        (a, obj_row, 0, eq)
    }

    fn reset_tableau(&mut self, preserve_basis: bool) {
        let dim = self.constraints.col_count();
        let rows = self.constraints.row_count();
        let rows_with_aux = rows.saturating_add(1);
        self.pivots = PivotCounts::default();
        if preserve_basis
            && self.transform.dim() == dim
            && self.nbindex.len() == dim
            && self.bflag.len() >= rows_with_aux
        {
            self.bflag.resize(rows_with_aux, -1);
            self.rebuild_row_order();
            return;
        }
        self.transform = SmallMat::identity(dim);
        self.nbindex = (0..dim)
            .map(|j| {
                if j == self.rhs_col {
                    0
                } else {
                    -((j as isize) + 1)
                }
            })
            .collect();
        self.bflag = vec![-1; rows_with_aux];
        if self.obj_row < rows_with_aux {
            self.bflag[self.obj_row] = 0;
        }
        self.nbindex_ref = None;
        self.seeded = false;
        self.scratch_pivot_row_for = usize::MAX;
        self.rebuild_row_order();
    }

    fn rebuild_row_order(&mut self) {
        let mut order = Vec::with_capacity(self.row_count());
        for idx in self.equality_rows.iter() {
            let i: usize = idx.into();
            if i < self.row_count() && i != self.obj_row {
                order.push(i);
            }
        }
        for i in 0..self.row_count() {
            if self.equality_rows.contains(RowId::new(i)) || i == self.obj_row {
                continue;
            }
            order.push(i);
        }
        if self.obj_row < self.row_count() {
            order.push(self.obj_row);
        }
        self.row_order = order;
    }

    fn pivot_cap(&self) -> Option<usize> {
        if let Some(cap) = self.problem.max_pivots {
            return Some(cap);
        }
        let dim = self.dim().max(1);
        Some(dim.saturating_mul(20))
    }

    fn row_count(&self) -> usize {
        self.constraints.row_count()
    }

    fn dim(&self) -> usize {
        self.constraints.col_count()
    }

    #[inline(always)]
    fn ensure_scratch_dim(&mut self) {
        let dim = self.dim();
        if self.scratch_obj_row.len() != dim {
            self.scratch_obj_row.resize(dim, N::zero());
        }
        if self.scratch_pivot_row.len() != dim {
            self.scratch_pivot_row.resize(dim, N::zero());
        }
        if self.scratch_aux_row.len() != dim {
            self.scratch_aux_row.resize(dim, N::zero());
        }
        self.scratch_ties.resize(dim);
        self.scratch_working.resize(dim);
        self.scratch_next.resize(dim);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tableau_row_into(
        constraints: &Matrix<N>,
        transform: &SmallMat<N>,
        row_count: usize,
        dim: usize,
        row: usize,
        aux_row: Option<&[N]>,
        eps: &impl Epsilon<N>,
        out: &mut [N],
    ) {
        debug_assert_eq!(out.len(), dim);

        let zero = N::zero();
        out.fill(zero);

        let coeffs = unsafe {
            if row < row_count {
                Some(constraints.row_unchecked(row))
            } else {
                aux_row
            }
        };
        let Some(coeffs) = coeffs else {
            return;
        };
        debug_assert_eq!(coeffs.len(), dim);

        for (k, a) in coeffs.iter().enumerate() {
            if eps.is_zero(a) {
                continue;
            }
            // SAFETY: k < dim because coeffs has length dim.
            let t_row = unsafe { transform.row_slice_unchecked(k) };
            linalg::axpy_add(out, a, t_row);
        }
    }

    fn tableau_entry_with_row(&self, row: usize, col: usize, aux_row: Option<&[N]>) -> N {
        let mut acc = N::zero();
        let coeffs = unsafe {
            if row < self.row_count() {
                Some(self.constraints.row_unchecked(row))
            } else {
                aux_row
            }
        };
        let Some(coeffs) = coeffs else {
            return acc;
        };
        let dim = self.dim();
        debug_assert_eq!(coeffs.len(), dim);
        debug_assert!(col < dim, "tableau column out of bounds");

        for (k, a) in coeffs.iter().enumerate() {
            if self.eps.is_zero(a) {
                continue;
            }
            // SAFETY: k < dim and col < dim by construction.
            let t_entry = unsafe { self.transform.get_unchecked(k, col) };
            linalg::add_mul_assign(&mut acc, a, t_entry);
        }
        acc
    }

    fn pivot_with_row(&mut self, row: usize, col: usize, pivot_row: &[N]) -> Result<(), LpStatus> {
        if self
            .transform
            .pivot_column(pivot_row, col, self.eps)
            .is_err()
        {
            return Err(LpStatus::PivotFailure);
        }
        if row >= self.bflag.len() {
            self.bflag.resize(row + 1, -1);
        }
        debug_assert!(col < self.nbindex.len(), "pivot column out of range");
        let entering = self.nbindex[col];
        self.bflag[row] = col as isize;
        self.nbindex[col] = (row as isize) + 1;
        if entering > 0 {
            let leaving_row = (entering - 1) as usize;
            debug_assert!(
                leaving_row < self.bflag.len(),
                "leaving row out of range while updating basis"
            );
            self.bflag[leaving_row] = -1;
        }
        self.scratch_pivot_row_for = usize::MAX;
        Ok(())
    }

    #[inline(always)]
    fn pivot_cached_row(&mut self, row: usize, col: usize) -> Result<(), LpStatus> {
        debug_assert_eq!(self.scratch_pivot_row_for, row);

        // SAFETY: scratch_pivot_row is not mutated during the pivot, so taking a
        // raw slice avoids borrow checker churn without additional allocations.
        let len = self.scratch_pivot_row.len();
        let ptr = self.scratch_pivot_row.as_ptr();
        let pivot_row = unsafe { std::slice::from_raw_parts(ptr, len) };
        self.pivot_with_row(row, col, pivot_row)
    }

    #[inline(always)]
    fn record_pivot(&mut self, phase: PivotPhase) {
        self.pivots.bump(phase);
    }

    fn select_pivot2(
        &mut self,
        row_selected: &RowSet,
        col_selected: &ColSet,
        row_limit: usize,
    ) -> Option<(usize, usize)> {
        let rows = self.row_count();
        let dim = self.dim();
        self.ensure_scratch_dim();

        for pass in 0..2 {
            for &r in &self.row_order {
                if r >= row_limit {
                    continue;
                }
                if row_selected.contains(r) {
                    continue;
                }
                let is_eq = self.equality_rows.contains(RowId::new(r));
                if (pass == 0 && !is_eq) || (pass == 1 && is_eq) {
                    continue;
                }

                for c in 0..dim {
                    if col_selected.contains(c) {
                        continue;
                    }
                    let val = self.tableau_entry_with_row(r, c, None);
                    if !self.eps.is_zero(&val) {
                        Self::tableau_row_into(
                            &self.constraints,
                            &self.transform,
                            rows,
                            dim,
                            r,
                            None,
                            self.eps,
                            &mut self.scratch_pivot_row,
                        );
                        self.scratch_pivot_row_for = r;
                        return Some((r, c));
                    }
                }
            }
        }
        None
    }

    fn find_lp_basis(&mut self, pivot_limit: Option<usize>) -> Result<Option<Col>, LpStatus> {
        let rows = self.row_count();
        let cols = self.dim();
        self.ensure_scratch_dim();
        let mut row_selected = RowSet::new(rows);
        let mut col_selected = ColSet::new(cols);
        row_selected.insert(self.obj_row);
        col_selected.insert(self.rhs_col);
        let mut rank = 0usize;
        let mut evidence_col = None;

        loop {
            if let Some(limit) = pivot_limit
                && self.pivots.total >= limit as u64
            {
                return Err(LpStatus::PivotLimitExceeded);
            }
            if rank + 1 >= cols {
                break;
            }
            if let Some((r, c)) = self.select_pivot2(&row_selected, &col_selected, rows) {
                debug_assert_eq!(self.scratch_pivot_row_for, r);
                self.pivot_cached_row(r, c)?;
                self.record_pivot(PivotPhase::Setup);
                row_selected.insert(r);
                col_selected.insert(c);
                rank += 1;
            } else {
                for j in 0..cols {
                    if j == self.rhs_col {
                        continue;
                    }
                    if self.nbindex.get(j).copied().unwrap_or(-1) < 0 {
                        let val = self.tableau_entry_with_row(self.obj_row, j, None);
                        if !self.eps.is_zero(&val) {
                            evidence_col = Some(j);
                            return Ok(evidence_col);
                        }
                    }
                }
                break;
            }
        }
        Ok(evidence_col)
    }

    fn build_aux_row_into(&self, aux_row: &mut [N]) {
        let dim = self.dim();
        debug_assert_eq!(aux_row.len(), dim);
        aux_row.fill(N::zero());
        for (col_idx, &nb) in self.nbindex.iter().enumerate() {
            if nb <= 0 {
                continue;
            }
            let scale = N::from_u64(col_idx as u64 + 11);
            let row_idx = nb.saturating_sub(1) as usize;
            let Some(row) = self.constraints.row(row_idx) else {
                continue;
            };
            debug_assert_eq!(row.len(), dim);
            linalg::axpy_sub(aux_row, &scale, row);
        }
    }

    fn find_dual_feasible_basis(
        &mut self,
        pivot_limit: Option<usize>,
    ) -> Result<Option<Col>, LpStatus> {
        let dim = self.dim();
        let row_count = self.row_count();
        self.ensure_scratch_dim();
        self.scratch_pivot_row_for = usize::MAX;
        Self::tableau_row_into(
            &self.constraints,
            &self.transform,
            row_count,
            dim,
            self.obj_row,
            None,
            self.eps,
            &mut self.scratch_obj_row,
        );
        let rcost = self.scratch_obj_row.clone();
        let mut any_positive = false;
        for (j, cost) in rcost.iter().enumerate() {
            if j != self.rhs_col && self.eps.is_positive(cost) {
                any_positive = true;
                break;
            }
        }
        if !any_positive {
            return Ok(None);
        }

        let mut aux_row = std::mem::take(&mut self.scratch_aux_row);
        self.build_aux_row_into(&mut aux_row);
        let result = (|| -> Result<Option<Col>, LpStatus> {
            let aux_idx = row_count;
            if self.bflag.len() <= aux_idx {
                self.bflag.resize(aux_idx + 1, -1);
            }

            Self::tableau_row_into(
                &self.constraints,
                &self.transform,
                row_count,
                dim,
                aux_idx,
                Some(&aux_row),
                self.eps,
                &mut self.scratch_pivot_row,
            );
            self.scratch_pivot_row_for = aux_idx;

            let mut best_col: Option<usize> = None;
            let mut best_den: Option<N> = None;
            for j in 0..dim {
                if j == self.rhs_col {
                    continue;
                }
                if !self.eps.is_positive(&rcost[j]) {
                    continue;
                }
                let ax_entry = &self.scratch_pivot_row[j];
                if !self.eps.is_negative(ax_entry) {
                    return Err(LpStatus::PivotFailure);
                }
                let den = ax_entry.ref_neg();
                match best_col {
                    None => {
                        best_col = Some(j);
                        best_den = Some(den);
                    }
                    Some(best_j) => {
                        let best_den_ref = best_den.as_ref().expect("best denominator exists");
                        if compare_positive_ratios(
                            &rcost[j],
                            &den,
                            &rcost[best_j],
                            best_den_ref,
                            self.eps,
                        )
                        .is_gt()
                        {
                            best_col = Some(j);
                            best_den = Some(den);
                        }
                    }
                }
            }
            let ms = match best_col {
                Some(col) => col,
                None => return Ok(None),
            };

            if let Some(limit) = pivot_limit
                && self.pivots.total >= limit as u64
            {
                return Err(LpStatus::PivotLimitExceeded);
            }
            self.pivot_cached_row(aux_idx, ms)?;
            self.record_pivot(PivotPhase::PhaseOne);
            self.nbindex_ref = Some(self.nbindex.clone());

            loop {
                if let Some(limit) = pivot_limit
                    && self.pivots.total >= limit as u64
                {
                    return Err(LpStatus::PivotLimitExceeded);
                }
                let choice = self.select_dual_simplex_pivot(row_count + 1, true, Some(aux_idx));
                match choice.pivot {
                    Some((r, c)) => {
                        if r == aux_idx {
                            Self::tableau_row_into(
                                &self.constraints,
                                &self.transform,
                                row_count,
                                dim,
                                r,
                                Some(&aux_row),
                                self.eps,
                                &mut self.scratch_pivot_row,
                            );
                            self.scratch_pivot_row_for = r;
                        } else {
                            debug_assert_eq!(self.scratch_pivot_row_for, r);
                        }
                        self.pivot_cached_row(r, c)?;
                        self.record_pivot(PivotPhase::PhaseOne);
                        if self.bflag[aux_idx] < 0 {
                            break;
                        }
                    }
                    None => match choice.status {
                        LpStatus::DualInconsistent => return Ok(choice.col_evidence),
                        LpStatus::Inconsistent => return Err(LpStatus::PivotFailure),
                        _ => {
                            // Force auxiliary variable into basis if needed.
                            let mut minval = None;
                            let mut leave_row = None;
                            for r in 0..row_count + 1 {
                                if self.bflag[r] < 0 {
                                    let val = self.tableau_entry_with_row(r, ms, Some(&aux_row));
                                    if minval
                                        .as_ref()
                                        .map(|m| self.eps.cmp(&val, m).is_lt())
                                        .unwrap_or(true)
                                    {
                                        minval = Some(val.clone());
                                        leave_row = Some(r);
                                    }
                                }
                            }
                            let r = leave_row.ok_or(LpStatus::PivotFailure)?;
                            if r == aux_idx {
                                Self::tableau_row_into(
                                    &self.constraints,
                                    &self.transform,
                                    row_count,
                                    dim,
                                    r,
                                    Some(&aux_row),
                                    self.eps,
                                    &mut self.scratch_pivot_row,
                                );
                            } else {
                                Self::tableau_row_into(
                                    &self.constraints,
                                    &self.transform,
                                    row_count,
                                    dim,
                                    r,
                                    None,
                                    self.eps,
                                    &mut self.scratch_pivot_row,
                                );
                            }
                            self.scratch_pivot_row_for = r;
                            if let Some(limit) = pivot_limit
                                && self.pivots.total >= limit as u64
                            {
                                return Err(LpStatus::PivotLimitExceeded);
                            }
                            self.pivot_cached_row(r, ms)?;
                            self.record_pivot(PivotPhase::PhaseOne);
                            break;
                        }
                    },
                }
            }
            Ok(None)
        })();
        self.scratch_aux_row = aux_row;
        result
    }

    fn select_dual_simplex_pivot(
        &mut self,
        row_limit: usize,
        phase_one: bool,
        aux_row: Option<usize>,
    ) -> PivotChoice {
        self.ensure_scratch_dim();
        self.scratch_pivot_row_for = usize::MAX;
        let dim = self.dim();
        let row_count = self.row_count();

        Self::tableau_row_into(
            &self.constraints,
            &self.transform,
            row_count,
            dim,
            self.obj_row,
            None,
            self.eps,
            &mut self.scratch_obj_row,
        );
        for j in 0..dim {
            if j == self.rhs_col {
                continue;
            }
            if self.eps.is_positive(&self.scratch_obj_row[j]) {
                return PivotChoice {
                    status: LpStatus::Undecided,
                    pivot: None,
                    row_evidence: None,
                    col_evidence: None,
                };
            }
        }

        let mut minval: Option<N> = None;
        let mut leaving_row = None;
        for i in self.row_order.iter().copied() {
            if i >= row_limit {
                continue;
            }
            if i == self.obj_row || self.bflag[i] != -1 {
                continue;
            }

            let val = if phase_one {
                let aux_col = aux_row
                    .map(|idx| self.bflag[idx])
                    .unwrap_or(self.rhs_col as isize) as usize;
                self.tableau_entry_with_row(i, aux_col, None).ref_neg()
            } else {
                self.tableau_entry_with_row(i, self.rhs_col, None)
            };
            if minval
                .as_ref()
                .map(|m| self.eps.cmp(&val, m).is_lt())
                .unwrap_or(true)
            {
                minval = Some(val);
                leaving_row = Some(i);
            }
        }

        if minval
            .as_ref()
            .map(|v| !self.eps.is_negative(v))
            .unwrap_or(true)
        {
            return PivotChoice {
                status: LpStatus::Optimal,
                pivot: None,
                row_evidence: None,
                col_evidence: None,
            };
        }

        let r = leaving_row.expect("leaving row exists");
        Self::tableau_row_into(
            &self.constraints,
            &self.transform,
            row_count,
            dim,
            r,
            None,
            self.eps,
            &mut self.scratch_pivot_row,
        );
        self.scratch_pivot_row_for = r;

        self.scratch_ties.clear();
        self.scratch_working.clear();
        self.scratch_next.clear();
        let mut s_choice: Option<usize> = None;
        let mut best_num: Option<N> = None;

        for j in 0..dim {
            if j == self.rhs_col {
                continue;
            }
            let val = &self.scratch_pivot_row[j];
            if !self.eps.is_positive(val) {
                continue;
            }

            let num = self.scratch_obj_row[j].ref_neg();
            match s_choice {
                None => {
                    s_choice = Some(j);
                    best_num = Some(num);
                    self.scratch_ties.insert(j);
                }
                Some(best_j) => {
                    let best_den = &self.scratch_pivot_row[best_j];
                    let best_num_ref = best_num.as_ref().expect("best numerator exists");
                    match compare_positive_ratios(&num, val, best_num_ref, best_den, self.eps) {
                        std::cmp::Ordering::Less => {
                            s_choice = Some(j);
                            best_num = Some(num);
                            self.scratch_ties.clear();
                            self.scratch_ties.insert(j);
                        }
                        std::cmp::Ordering::Equal => self.scratch_ties.insert(j),
                        std::cmp::Ordering::Greater => {}
                    }
                }
            }
        }

        if self.problem.lexicographic_pivot
            && self.scratch_ties.cardinality() > 1
            && let Some(refs) = self.nbindex_ref.as_deref()
        {
            self.scratch_working.clone_from(&self.scratch_ties);

            for &iref in refs.iter().skip(1) {
                if iref <= 0 {
                    continue;
                }
                let row_idx = iref.saturating_sub(1) as usize;

                let col_basic = self.bflag[row_idx];
                if col_basic > 0 {
                    let col_basic = col_basic as usize;
                    if self.scratch_working.cardinality() == 1
                        && self.scratch_working.contains(col_basic)
                    {
                        s_choice = Some(col_basic);
                        break;
                    }
                    self.scratch_working.remove(col_basic);
                    if self.scratch_working.cardinality() == 1 {
                        s_choice = self.scratch_working.iter().next().map(ColId::as_index);
                        break;
                    }
                    continue;
                }

                let mut best_col: Option<usize> = None;
                self.scratch_next.clear();

                let Some(coeffs) = self.constraints.row(row_idx) else {
                    continue;
                };
                for j in self.scratch_working.iter() {
                    self.scratch_obj_row[j.as_index()] = N::zero();
                }
                for (k, a) in coeffs.iter().enumerate() {
                    if self.eps.is_zero(a) {
                        continue;
                    }
                    let t_row = self.transform.row_slice(k);
                    for j in self.scratch_working.iter() {
                        let j = j.as_index();
                        linalg::add_mul_assign(&mut self.scratch_obj_row[j], a, &t_row[j]);
                    }
                }

                for j in self.scratch_working.iter() {
                    let j = j.as_index();
                    let denom = &self.scratch_pivot_row[j];
                    if !self.eps.is_positive(denom) {
                        continue;
                    }
                    let num = &self.scratch_obj_row[j];
                    match best_col {
                        None => {
                            best_col = Some(j);
                            self.scratch_next.insert(j);
                        }
                        Some(best_j) => {
                            let best_denom = &self.scratch_pivot_row[best_j];
                            let best_num_ref = &self.scratch_obj_row[best_j];
                            match compare_positive_ratios(
                                num,
                                denom,
                                best_num_ref,
                                best_denom,
                                self.eps,
                            ) {
                                std::cmp::Ordering::Less => {
                                    best_col = Some(j);
                                    self.scratch_next.clear();
                                    self.scratch_next.insert(j);
                                }
                                std::cmp::Ordering::Equal => self.scratch_next.insert(j),
                                std::cmp::Ordering::Greater => {}
                            }
                        }
                    };
                }

                if !self.scratch_next.is_empty() {
                    std::mem::swap(&mut self.scratch_working, &mut self.scratch_next);
                    self.scratch_next.clear();
                    if self.scratch_working.cardinality() == 1 {
                        s_choice = self.scratch_working.iter().next().map(ColId::as_index);
                        break;
                    }
                }
            }

            if s_choice.is_none() && !self.scratch_working.is_empty() {
                s_choice = self.scratch_working.iter().next().map(ColId::as_index);
            }
        }

        if let Some(s) = s_choice {
            PivotChoice {
                status: LpStatus::Undecided,
                pivot: Some((r, s)),
                row_evidence: None,
                col_evidence: None,
            }
        } else {
            PivotChoice {
                status: LpStatus::Inconsistent,
                pivot: None,
                row_evidence: Some(r),
                col_evidence: None,
            }
        }
    }

    fn select_criss_cross_pivot(&mut self) -> PivotChoice {
        self.ensure_scratch_dim();
        self.scratch_pivot_row_for = usize::MAX;
        let row_count = self.row_count();
        let dim = self.dim();

        Self::tableau_row_into(
            &self.constraints,
            &self.transform,
            row_count,
            dim,
            self.obj_row,
            None,
            self.eps,
            &mut self.scratch_obj_row,
        );

        self.scratch_working.clear();
        for &col in self.bflag.iter().take(row_count) {
            if col > 0 {
                self.scratch_working.insert(col as usize);
            }
        }
        self.scratch_working.remove(self.rhs_col);

        for &row in &self.row_order {
            if row == self.obj_row {
                continue;
            }
            let flag = self.bflag[row];
            if flag == -1 {
                let rhs = self.tableau_entry_with_row(row, self.rhs_col, None);
                if self.eps.is_negative(&rhs) {
                    Self::tableau_row_into(
                        &self.constraints,
                        &self.transform,
                        row_count,
                        dim,
                        row,
                        None,
                        self.eps,
                        &mut self.scratch_pivot_row,
                    );
                    self.scratch_pivot_row_for = row;

                    for col in self.scratch_working.iter() {
                        let col_idx = col.as_index();
                        if self.eps.is_positive(&self.scratch_pivot_row[col_idx]) {
                            return PivotChoice {
                                status: LpStatus::Undecided,
                                pivot: Some((row, col_idx)),
                                row_evidence: None,
                                col_evidence: None,
                            };
                        }
                    }
                    return PivotChoice {
                        status: LpStatus::Inconsistent,
                        pivot: None,
                        row_evidence: Some(row),
                        col_evidence: None,
                    };
                }
            } else if flag > 0 {
                let col = flag as usize;
                let rcost = &self.scratch_obj_row[col];
                if self.eps.is_positive(rcost) {
                    for probe in 0..row_count {
                        if probe == self.obj_row {
                            continue;
                        }
                        if self.bflag[probe] == -1 {
                            let val = self.tableau_entry_with_row(probe, col, None);
                            if self.eps.is_negative(&val) {
                                Self::tableau_row_into(
                                    &self.constraints,
                                    &self.transform,
                                    row_count,
                                    dim,
                                    probe,
                                    None,
                                    self.eps,
                                    &mut self.scratch_pivot_row,
                                );
                                self.scratch_pivot_row_for = probe;
                                return PivotChoice {
                                    status: LpStatus::Undecided,
                                    pivot: Some((probe, col)),
                                    row_evidence: None,
                                    col_evidence: None,
                                };
                            }
                        }
                    }
                    return PivotChoice {
                        status: LpStatus::DualInconsistent,
                        pivot: None,
                        row_evidence: None,
                        col_evidence: Some(col),
                    };
                }
            }
        }
        PivotChoice {
            status: LpStatus::Optimal,
            pivot: None,
            row_evidence: None,
            col_evidence: None,
        }
    }

    fn dual_simplex(&mut self, pivot_limit: Option<usize>) -> (LpStatus, Option<Row>, Option<Col>) {
        let mut row_cert = None;
        let mut col_cert = None;

        if !self.seeded {
            let basis_evidence = match self.find_lp_basis(pivot_limit) {
                Ok(col) => col,
                Err(status) => return (status, None, None),
            };
            if let Some(col) = basis_evidence {
                return (LpStatus::Unbounded, None, Some(col));
            }

            match self.find_dual_feasible_basis(pivot_limit) {
                Ok(Some(col)) => {
                    col_cert = Some(col);
                    return (LpStatus::DualInconsistent, row_cert, col_cert);
                }
                Ok(None) => {}
                Err(status) => return (status, row_cert, col_cert),
            }
        }

        if self.nbindex_ref.is_none() {
            self.nbindex_ref = Some(self.nbindex.clone());
        }

        loop {
            if let Some(limit) = pivot_limit
                && self.pivots.total >= limit as u64
            {
                return (LpStatus::PivotLimitExceeded, row_cert, col_cert);
            }
            let choice = self.select_dual_simplex_pivot(self.row_count(), false, None);
            match choice.pivot {
                Some((r, c)) => {
                    debug_assert_eq!(self.scratch_pivot_row_for, r);
                    let result = self.pivot_cached_row(r, c);
                    if let Err(status) = result {
                        return (status, row_cert, col_cert);
                    }
                    self.record_pivot(PivotPhase::PhaseTwo);
                }
                None => match choice.status {
                    LpStatus::Optimal => return (LpStatus::Optimal, row_cert, col_cert),
                    LpStatus::Inconsistent => {
                        row_cert = choice.row_evidence;
                        return (LpStatus::Inconsistent, row_cert, col_cert);
                    }
                    LpStatus::DualInconsistent => {
                        col_cert = choice.col_evidence;
                        return (LpStatus::DualInconsistent, row_cert, col_cert);
                    }
                    _ => {
                        let cc = self.select_criss_cross_pivot();
                        match cc.pivot {
                            Some((r, c)) => {
                                debug_assert_eq!(self.scratch_pivot_row_for, r);
                                let result = self.pivot_cached_row(r, c);
                                if let Err(status) = result {
                                    return (status, row_cert, col_cert);
                                }
                                self.record_pivot(PivotPhase::CrissCross);
                            }
                            None => match cc.status {
                                LpStatus::Optimal => {
                                    return (LpStatus::Optimal, row_cert, col_cert);
                                }
                                LpStatus::Inconsistent => {
                                    row_cert = cc.row_evidence;
                                    return (LpStatus::Inconsistent, row_cert, col_cert);
                                }
                                LpStatus::DualInconsistent => {
                                    col_cert = cc.col_evidence;
                                    return (LpStatus::DualInconsistent, row_cert, col_cert);
                                }
                                other => return (other, row_cert, col_cert),
                            },
                        }
                    }
                },
            }
        }
    }

    fn criss_cross(&mut self, pivot_limit: Option<usize>) -> (LpStatus, Option<Row>, Option<Col>) {
        let mut row_cert = None;
        let mut col_cert = None;
        self.ensure_scratch_dim();

        let basis_evidence = if self.seeded {
            None
        } else {
            match self.find_lp_basis(pivot_limit) {
                Ok(col) => col,
                Err(status) => return (status, row_cert, col_cert),
            }
        };
        if let Some(col) = basis_evidence {
            return (LpStatus::Unbounded, row_cert, Some(col));
        }

        loop {
            if let Some(limit) = pivot_limit
                && self.pivots.total >= limit as u64
            {
                return (LpStatus::PivotLimitExceeded, row_cert, col_cert);
            }
            let choice = self.select_criss_cross_pivot();
            match choice.pivot {
                Some((r, c)) => {
                    debug_assert_eq!(self.scratch_pivot_row_for, r);
                    let result = self.pivot_cached_row(r, c);
                    if let Err(status) = result {
                        return (status, row_cert, col_cert);
                    }
                    self.record_pivot(PivotPhase::CrissCross);
                }
                None => match choice.status {
                    LpStatus::Optimal => return (LpStatus::Optimal, row_cert, col_cert),
                    LpStatus::Inconsistent => {
                        row_cert = choice.row_evidence;
                        return (LpStatus::Inconsistent, row_cert, col_cert);
                    }
                    LpStatus::DualInconsistent => {
                        col_cert = choice.col_evidence;
                        return (LpStatus::DualInconsistent, row_cert, col_cert);
                    }
                    other => return (other, row_cert, col_cert),
                },
            }
        }
    }

    fn solve(&mut self, solver: LpSolver, capture_snapshot: bool) -> LpResult<N> {
        self.reset_tableau(self.seeded);
        if self.seeded && self.nbindex_ref.is_none() {
            self.nbindex_ref = Some(self.nbindex.clone());
        }
        let minimize = matches!(self.problem.objective, LpObjective::Minimize);
        let pivot_limit = match solver {
            LpSolver::DualSimplex => self.pivot_cap(),
            LpSolver::CrissCross => self.pivot_cap().map(|cap| cap.saturating_mul(5)), /* mirror cddlib 100*d cap */
        };
        if minimize && let Some(obj_row) = self.constraints.row_mut(self.obj_row) {
            for v in obj_row {
                *v = v.ref_neg();
            }
        }
        let (status, row_cert, col_cert) = match solver {
            LpSolver::DualSimplex => self.dual_simplex(pivot_limit),
            LpSolver::CrissCross => self.criss_cross(pivot_limit),
        };
        if minimize && let Some(obj_row) = self.constraints.row_mut(self.obj_row) {
            for v in obj_row {
                *v = v.ref_neg();
            }
        }

        let mut primal_vec = vec![N::zero(); self.problem.cols];
        for j in 0..self.dim().min(primal_vec.len()) {
            primal_vec[j] = self.transform.get(j, self.rhs_col).clone();
        }

        self.ensure_scratch_dim();
        let row_count = self.row_count();
        let dim = self.dim();
        Self::tableau_row_into(
            &self.constraints,
            &self.transform,
            row_count,
            dim,
            self.obj_row,
            None,
            self.eps,
            &mut self.scratch_obj_row,
        );
        let optimal_value = self.scratch_obj_row[self.rhs_col].clone();
        let mut dual = vec![N::zero(); dim];
        for (dst, src) in dual.iter_mut().zip(&self.scratch_obj_row) {
            *dst = src.ref_neg();
        }

        self.build_result(
            status,
            optimal_value,
            primal_vec,
            dual,
            row_cert,
            col_cert,
            solver,
            capture_snapshot,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        status: LpStatus,
        optimal_value: N,
        primal: Vec<N>,
        dual: Vec<N>,
        row_certificate: Option<Row>,
        col_certificate: Option<Col>,
        solver: LpSolver,
        capture_snapshot: bool,
    ) -> LpResult<N> {
        let snapshot = capture_snapshot.then(|| {
            let mut basis_rows = Vec::new();
            for (row_idx, flag) in self.bflag.iter().enumerate().take(self.row_count()) {
                if row_idx != self.obj_row && *flag == -1 {
                    basis_rows.push(row_idx);
                }
            }
            LpSnapshot {
                constraints: self.constraints.clone(),
                transform: self.transform.clone(),
                nonbasic_index: self.nbindex.clone(),
                bflag: self.bflag.clone(),
                obj_row: self.obj_row,
                rhs_col: self.rhs_col,
                basis_rows,
                equality_rows: self.equality_rows.clone(),
                row_order: self.row_order.clone(),
            }
        });
        LpResult {
            objective: self.problem.objective,
            solver,
            rows: self.problem.rows,
            cols: self.problem.cols,
            homogeneous: self.problem.homogeneous,
            representation: self.problem.representation,
            status,
            optimal_value,
            primal,
            dual,
            nonbasic_index: self.nbindex.clone(),
            row_certificate,
            col_certificate,
            pivots: self.pivots.clone(),
            equality_rows: self.equality_rows.clone(),
            snapshot,
        }
    }
}
