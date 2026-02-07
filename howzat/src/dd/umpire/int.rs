use crate::HowzatError as Error;
use crate::dd::ray::RayData;
use crate::dd::RayId;
use crate::dd::zero::{ZeroRepr, ZeroSet};
use crate::matrix::{LpMatrix, MatrixRank};
use crate::polyhedron::{bareiss_solve_det_times_matrix_in_place, scaled_integer_rows, IntRowMatrix};
use calculo::num::{Epsilon, Int, Rat, Sign};
use hullabaloo::matrix::BasisMatrix;
use hullabaloo::types::{ColSet, Representation, Row, RowSet};

use super::policies::{HalfspacePolicy, LexMin};
use super::{ConeCtx, Umpire, UmpireMatrix};

#[derive(Debug)]
pub struct IntMatrix<N: Rat, R: Representation> {
    pub(crate) base: LpMatrix<N, R>,
    pub(crate) int_rows: IntRowMatrix<N::Int>,
}

impl<N: Rat, R: Representation> Clone for IntMatrix<N, R> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            int_rows: self.int_rows.clone(),
        }
    }
}

impl<N: Rat, R: Representation> IntMatrix<N, R> {
    #[inline(always)]
    fn int_row(&self, row: Row) -> &[N::Int] {
        self.int_rows.row(row).expect("row index within bounds")
    }
}

impl<N: Rat, R: Representation> UmpireMatrix<N, R> for IntMatrix<N, R> {
    #[inline(always)]
    fn base(&self) -> &LpMatrix<N, R> {
        &self.base
    }

    fn select_columns(&self, columns: &[usize]) -> Result<Self, Error> {
        let base = self.base.select_columns(columns)?;
        let old_cols = self.int_rows.col_count();
        let rows = self.int_rows.row_count();
        let new_cols = columns.len();

        let mut data = Vec::with_capacity(rows * new_cols);
        for row in 0..rows {
            let src = self
                .int_rows
                .row(row)
                .unwrap_or_else(|| panic!("row {row} out of bounds"));
            debug_assert_eq!(src.len(), old_cols, "int row width mismatch");
            for &col in columns {
                data.push(src[col].clone());
            }
        }
        let int_rows = IntRowMatrix::new(rows, new_cols, data).ok_or(Error::DimensionTooLarge)?;
        Ok(Self { base, int_rows })
    }
}

#[derive(Clone, Debug)]
pub struct IntRay<N: Rat, ZS: ZeroSet = crate::dd::SatSet> {
    pub(crate) vector: Vec<N::Int>,
    pub(crate) row_signs: Vec<Option<Sign>>,
    pub(crate) zero_set: ZS,
    pub(crate) zero_set_sig: u64,
    pub(crate) zero_set_count: usize,
    pub(crate) first_infeasible_row: Option<Row>,
    pub(crate) feasible: bool,
    pub(crate) weakly_feasible: bool,
    pub(crate) last_eval_row: Option<Row>,
    pub(crate) last_eval: N::Int,
    pub(crate) last_sign: Sign,
}

impl<N: Rat, ZS: ZeroSet> IntRay<N, ZS> {
    #[inline(always)]
    fn vector(&self) -> &[N::Int] {
        &self.vector
    }

    #[inline(always)]
    fn cached_sign(&self, row: Row) -> Option<Sign> {
        (self.last_eval_row == Some(row)).then_some(self.last_sign)
    }

    #[inline(always)]
    fn cached_row_sign(&self, row: Row) -> Option<Sign> {
        self.row_signs.get(row).copied().flatten()
    }

    #[inline(always)]
    fn set_cached_row_sign(&mut self, row: Row, sign: Sign) {
        if let Some(slot) = self.row_signs.get_mut(row) {
            *slot = Some(sign);
        }
    }
}

impl<N: Rat, ZS: ZeroSet> crate::dd::ray::RayData for IntRay<N, ZS> {
    type ZeroSet = ZS;

    #[inline(always)]
    fn zero_set(&self) -> &ZS {
        &self.zero_set
    }

    #[inline(always)]
    fn zero_set_mut(&mut self) -> &mut ZS {
        &mut self.zero_set
    }

    #[inline(always)]
    fn zero_set_signature(&self) -> u64 {
        self.zero_set_sig
    }

    #[inline(always)]
    fn set_zero_set_signature(&mut self, sig: u64) {
        self.zero_set_sig = sig;
    }

    #[inline(always)]
    fn zero_set_count(&self) -> usize {
        self.zero_set_count
    }

    #[inline(always)]
    fn set_zero_set_count(&mut self, count: usize) {
        self.zero_set_count = count;
    }

    #[inline(always)]
    fn first_infeasible_row(&self) -> Option<Row> {
        self.first_infeasible_row
    }

    #[inline(always)]
    fn set_first_infeasible_row(&mut self, row: Option<Row>) {
        self.first_infeasible_row = row;
    }

    #[inline(always)]
    fn is_feasible(&self) -> bool {
        self.feasible
    }

    #[inline(always)]
    fn is_weakly_feasible(&self) -> bool {
        self.weakly_feasible
    }

    #[inline(always)]
    fn last_sign(&self) -> Sign {
        self.last_sign
    }
}

#[derive(Clone, Debug)]
pub struct IntUmpire<
    N: Rat,
    E: Epsilon<N> = calculo::num::DynamicEpsilon<N>,
    H: HalfspacePolicy<N> = LexMin,
> {
    eps: E,
    halfspace: H,
    dot_acc: N::Int,
    dot_acc2: N::Int,
    dot_tmp: N::Int,
    vector_pool: Vec<Vec<N::Int>>,
    row_sign_pool: Vec<Vec<Option<Sign>>>,
    phantom: std::marker::PhantomData<N>,
}

impl<N: Rat, E: Epsilon<N>> IntUmpire<N, E, LexMin> {
    pub fn new(eps: E) -> Self {
        Self::with_halfspace_policy(eps, LexMin)
    }
}

impl<N: Rat, E: Epsilon<N>, H: HalfspacePolicy<N>> IntUmpire<N, E, H> {
    pub fn with_halfspace_policy(eps: E, halfspace: H) -> Self {
        Self {
            eps,
            halfspace,
            dot_acc: N::Int::zero(),
            dot_acc2: N::Int::zero(),
            dot_tmp: N::Int::zero(),
            vector_pool: Vec::new(),
            row_sign_pool: Vec::new(),
            phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    fn take_vector(&mut self, dim: usize) -> Vec<N::Int> {
        let mut v = self.vector_pool.pop().unwrap_or_default();
        if v.len() != dim {
            v.resize(dim, N::Int::zero());
        }
        v
    }

    #[inline]
    fn take_row_signs(&mut self, row_count: usize) -> Vec<Option<Sign>> {
        let mut out = self.row_sign_pool.pop().unwrap_or_default();
        if out.len() != row_count {
            out.resize(row_count, None);
        } else {
            out.fill(None);
        }
        out
    }

    #[inline(always)]
    fn int_sign(value: &N::Int) -> Sign {
        if value.is_zero() {
            Sign::Zero
        } else if value.is_negative() {
            Sign::Negative
        } else {
            Sign::Positive
        }
    }

    #[inline(always)]
    fn combine_nonnegative_signs(a: Sign, b: Sign) -> Option<Sign> {
        match (a, b) {
            (Sign::Negative, Sign::Positive) | (Sign::Positive, Sign::Negative) => None,
            (Sign::Zero, sign) | (sign, Sign::Zero) => Some(sign),
            (Sign::Negative, Sign::Negative) => Some(Sign::Negative),
            (Sign::Positive, Sign::Positive) => Some(Sign::Positive),
        }
    }

    #[inline(always)]
    fn dot_int_in_acc<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        row: Row,
        ray: &[N::Int],
    ) {
        N::Int::assign_from(&mut self.dot_acc, &N::Int::zero());
        let row_vec = cone.matrix.int_row(row);
        debug_assert_eq!(row_vec.len(), ray.len(), "dot product dimension mismatch");
        for (a, b) in row_vec.iter().zip(ray.iter()) {
            N::Int::assign_from(&mut self.dot_tmp, a);
            self.dot_tmp
                .mul_assign(b)
                .expect("exact integer dot product requires multiplication");
            self.dot_acc += &self.dot_tmp;
        }
    }

    #[inline(always)]
    fn dot_int_clone<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        row: Row,
        ray: &[N::Int],
    ) -> N::Int {
        self.dot_int_in_acc(cone, row, ray);
        self.dot_acc.clone()
    }

    #[inline(always)]
    fn dot2_int_in_accs<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        row: Row,
        ray_a: &[N::Int],
        ray_b: &[N::Int],
    ) {
        N::Int::assign_from(&mut self.dot_acc, &N::Int::zero());
        N::Int::assign_from(&mut self.dot_acc2, &N::Int::zero());
        let row_vec = cone.matrix.int_row(row);
        debug_assert_eq!(
            row_vec.len(),
            ray_a.len(),
            "dot2 dimension mismatch (row={}, ray_a={})",
            row_vec.len(),
            ray_a.len()
        );
        debug_assert_eq!(
            row_vec.len(),
            ray_b.len(),
            "dot2 dimension mismatch (row={}, ray_b={})",
            row_vec.len(),
            ray_b.len()
        );

        for (a, (b1, b2)) in row_vec.iter().zip(ray_a.iter().zip(ray_b.iter())) {
            N::Int::assign_from(&mut self.dot_tmp, a);
            self.dot_tmp
                .mul_assign(b1)
                .expect("exact integer dot product requires multiplication");
            self.dot_acc += &self.dot_tmp;

            N::Int::assign_from(&mut self.dot_tmp, a);
            self.dot_tmp
                .mul_assign(b2)
                .expect("exact integer dot product requires multiplication");
            self.dot_acc2 += &self.dot_tmp;
        }
    }

    #[inline(always)]
    fn dot2_int_clone<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        row: Row,
        ray_a: &[N::Int],
        ray_b: &[N::Int],
    ) -> (N::Int, N::Int) {
        self.dot2_int_in_accs(cone, row, ray_a, ray_b);
        (self.dot_acc.clone(), self.dot_acc2.clone())
    }

    fn normalize_vector_in_place(vector: &mut [N::Int]) -> bool {
        let z0 = N::Int::zero();
        let z1 = N::Int::one();

        let mut any = false;
        let mut gcd: Option<N::Int> = None;
        for v in vector.iter() {
            if v.is_zero() {
                continue;
            }
            any = true;
            let abs = v.abs().expect("exact integer normalization requires abs");
            match gcd.as_mut() {
                None => gcd = Some(abs),
                Some(g) => g
                    .gcd_assign(&abs)
                    .expect("exact integer normalization requires gcd"),
            }
        }

        if !any {
            for v in vector.iter_mut() {
                N::Int::assign_from(v, &z0);
            }
            return false;
        }

        let Some(g) = gcd else {
            return false;
        };
        if g == z0 || g == z1 {
            return true;
        }
        for v in vector.iter_mut() {
            if v.is_zero() {
                continue;
            }
            v.div_assign_exact(&g)
                .expect("exact integer normalization requires exact division");
        }
        true
    }

    fn build_standard_from_vector(vector: &[N::Int]) -> Vec<N> {
        let one = N::Int::one();
        vector
            .iter()
            .map(|v| N::from_frac(v.clone(), one.clone()))
            .collect()
    }

    fn build_ray_from_vector<ZR: ZeroRepr, R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        vector: Vec<N::Int>,
        relaxed: bool,
        last_row: Option<Row>,
        mut zero_set: <ZR as ZeroRepr>::Set,
        seeded_zero_set: bool,
        mut preseeded_row_signs: Option<Vec<Option<Sign>>>,
    ) -> H::WrappedRayData<IntRay<N, <ZR as ZeroRepr>::Set>> {
        let m = cone.matrix().row_count();
        let mut negative_rows = self.halfspace.take_negative_rows(m);
        let track_negatives = negative_rows.len() == m;

        zero_set.ensure_domain(m);
        if !seeded_zero_set {
            zero_set.clear();
        }
        let mut row_signs = preseeded_row_signs
            .take()
            .unwrap_or_else(|| self.take_row_signs(m));
        if row_signs.len() != m {
            row_signs.resize(m, None);
        }
        let mut zero_set_count = zero_set.cardinality();
        let mut feasible = true;
        let mut weakly_feasible = true;
        let mut first_infeasible_row = None;

        let mut last_eval_row = None;
        let mut last_eval = N::Int::zero();
        let mut last_sign = Sign::Zero;

        for &row_idx in &cone.order_vector {
            let sign = if let Some(preseeded) = row_signs[row_idx] {
                if preseeded == Sign::Zero {
                    N::Int::assign_from(&mut self.dot_acc, &N::Int::zero());
                } else if Some(row_idx) == last_row {
                    self.dot_int_in_acc(cone, row_idx, &vector);
                }
                preseeded
            } else if seeded_zero_set
                && <ZR as ZeroRepr>::zero_set_contains_row(cone, &zero_set, row_idx)
            {
                N::Int::assign_from(&mut self.dot_acc, &N::Int::zero());
                Sign::Zero
            } else {
                self.dot_int_in_acc(cone, row_idx, &vector);
                Self::int_sign(&self.dot_acc)
            };
            row_signs[row_idx] = Some(sign);
            if track_negatives && sign == Sign::Negative {
                negative_rows.insert(row_idx);
            }
            if Some(row_idx) == last_row {
                last_eval_row = Some(row_idx);
                N::Int::assign_from(&mut last_eval, &self.dot_acc);
                last_sign = sign;
            }

            if sign == Sign::Zero
                && let Some(id) = <ZR as ZeroRepr>::id_for_row(cone, row_idx)
                && !zero_set.contains(id)
            {
                zero_set.insert(id);
                zero_set_count += 1;
            }

            let kind = cone.equality_kinds[row_idx];
            if kind.weakly_violates_sign(sign, relaxed) {
                if first_infeasible_row.is_none() {
                    first_infeasible_row = Some(row_idx);
                }
                weakly_feasible = false;
            }
            if kind.violates_sign(sign, relaxed) {
                feasible = false;
            }
        }

        if relaxed {
            feasible = weakly_feasible;
        }

        let zero_set_sig = zero_set.signature_u64();
        let ray_data = IntRay {
            vector,
            row_signs,
            zero_set,
            zero_set_sig,
            zero_set_count,
            first_infeasible_row,
            feasible,
            weakly_feasible,
            last_eval_row,
            last_eval,
            last_sign,
        };

        self.halfspace.wrap_ray_data(ray_data, negative_rows)
    }

    fn convert_vector_to_int(&mut self, vector: Vec<N>) -> Vec<N::Int> {
        let one = N::Int::one();
        let mut out: Vec<N::Int> = Vec::with_capacity(vector.len());
        let mut denoms: Option<Vec<N::Int>> = None;
        let mut scale = one.clone();

        for v in vector {
            let (mut numer, mut denom) = v.into_parts();
            if denom.is_negative() {
                denom
                    .neg_mut()
                    .expect("exact integer conversion requires negation");
                numer
                    .neg_mut()
                    .expect("exact integer conversion requires negation");
            }

            if denom != one {
                scale
                    .lcm_assign(&denom)
                    .expect("exact integer conversion requires LCM");
                let denoms = denoms.get_or_insert_with(|| vec![one.clone(); out.len()]);
                denoms.push(denom);
            } else if let Some(denoms) = denoms.as_mut() {
                denoms.push(one.clone());
            }

            out.push(numer);
        }

        let Some(denoms) = denoms else {
            return out;
        };
        debug_assert_eq!(
            denoms.len(),
            out.len(),
            "denominator bookkeeping mismatch ({} denoms, {} nums)",
            denoms.len(),
            out.len()
        );

        for (numer, denom) in out.iter_mut().zip(denoms.iter()) {
            if numer.is_zero() || *denom == one {
                continue;
            }
            N::Int::assign_from(&mut self.dot_tmp, &scale);
            self.dot_tmp
                .div_assign_exact(denom)
                .expect("exact integer conversion requires exact division");
            numer
                .mul_assign(&self.dot_tmp)
                .expect("exact integer conversion requires multiplication");
        }
        out
    }

    fn rank_basis_int<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        ignored_rows: &RowSet,
        ignored_cols: &ColSet,
    ) -> MatrixRank {
        let rows = cone.matrix().row_count();
        let cols = cone.matrix().col_count();
        if cols == 0 {
            return MatrixRank {
                rank: 0,
                row_basis: RowSet::new(rows),
                col_basis: ColSet::new(cols),
            };
        }

        let mut ignored_mask = vec![false; cols];
        for col in ignored_cols.iter() {
            ignored_mask[col.as_index()] = true;
        }
        let available_cols = cols - ignored_mask.iter().filter(|v| **v).count();

        let mut basis: Vec<Vec<N::Int>> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        let mut chosen_rows: Vec<Row> = Vec::new();
        let mut work: Vec<N::Int> = vec![N::Int::zero(); cols];

        for row in 0..rows {
            if ignored_rows.contains(row) {
                continue;
            }
            let src = cone.matrix.int_row(row);
            debug_assert_eq!(src.len(), cols);
            for (dst, v) in work.iter_mut().zip(src.iter()) {
                N::Int::assign_from(dst, v);
            }
            for (col, ignored) in ignored_mask.iter().enumerate() {
                if *ignored {
                    N::Int::assign_from(&mut work[col], &N::Int::zero());
                }
            }

            for (basis_row, &pivot_col) in basis.iter().zip(pivots.iter()) {
                let pivot = work[pivot_col].clone();
                if pivot.is_zero() {
                    continue;
                }
                let basis_pivot = basis_row[pivot_col].clone();
                if basis_pivot.is_zero() {
                    continue;
                }

                let mut g = basis_pivot.clone();
                g.gcd_assign(&pivot)
                    .expect("exact integer rank requires gcd");
                if g.is_zero() {
                    continue;
                }

                let mut a = basis_pivot;
                a.div_assign_exact(&g)
                    .expect("exact integer rank requires exact division");
                let mut b = pivot;
                b.div_assign_exact(&g)
                    .expect("exact integer rank requires exact division");

                for (c, ignored) in ignored_mask.iter().enumerate().skip(pivot_col) {
                    if *ignored {
                        continue;
                    }
                    let mut left = work[c].clone();
                    left.mul_assign(&a)
                        .expect("exact integer rank requires multiplication");
                    let mut right = basis_row[c].clone();
                    right.mul_assign(&b)
                        .expect("exact integer rank requires multiplication");
                    left -= &right;
                    N::Int::assign_from(&mut work[c], &left);
                }
            }

            let pivot_col = (0..cols).find(|&c| !ignored_mask[c] && !work[c].is_zero());
            let Some(pivot_col) = pivot_col else {
                continue;
            };

            basis.push(work.clone());
            pivots.push(pivot_col);
            chosen_rows.push(row);
            if basis.len() == available_cols {
                break;
            }
        }

        let mut row_basis = RowSet::new(rows);
        for row in chosen_rows {
            row_basis.insert(row);
        }
        let mut col_basis = ColSet::new(cols);
        for col in pivots {
            col_basis.insert(col);
        }

        MatrixRank {
            rank: basis.len(),
            row_basis,
            col_basis,
        }
    }

    fn choose_basis_rows_and_cols<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        strict_rows: &RowSet,
        equality_set: &RowSet,
    ) -> (Vec<Row>, Vec<usize>) {
        let cols = cone.matrix().col_count();
        if cols == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut basis: Vec<Vec<N::Int>> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        let mut chosen_rows: Vec<Row> = Vec::new();

        let mut work: Vec<N::Int> = vec![N::Int::zero(); cols];

        let mut candidates: Vec<Row> = Vec::new();
        for row in 0..cone.matrix().row_count() {
            if equality_set.contains(row) && !strict_rows.contains(row) {
                candidates.push(row);
            }
        }
        for &row in &cone.order_vector {
            if equality_set.contains(row) || strict_rows.contains(row) {
                continue;
            }
            candidates.push(row);
        }

        let z0 = N::Int::zero();
        let z1 = N::Int::one();

        for row in candidates {
            let src = cone.matrix.int_row(row);
            for (dst, v) in work.iter_mut().zip(src.iter()) {
                N::Int::assign_from(dst, v);
            }

            for (basis_row, &pivot_col) in basis.iter().zip(pivots.iter()) {
                let pivot = work[pivot_col].clone();
                if pivot.is_zero() {
                    continue;
                }
                let basis_pivot = basis_row[pivot_col].clone();
                if basis_pivot.is_zero() {
                    continue;
                }

                let mut g = basis_pivot.clone();
                g.gcd_assign(&pivot)
                    .expect("exact basis selection requires gcd");
                if g.is_zero() {
                    continue;
                }

                let mut a = basis_pivot;
                a.div_assign_exact(&g)
                    .expect("exact basis selection requires exact division");
                let mut b = pivot;
                b.div_assign_exact(&g)
                    .expect("exact basis selection requires exact division");

                for c in pivot_col..cols {
                    let mut left = work[c].clone();
                    left.mul_assign(&a)
                        .expect("exact basis selection requires multiplication");
                    let mut right = basis_row[c].clone();
                    right.mul_assign(&b)
                        .expect("exact basis selection requires multiplication");
                    left -= &right;
                    N::Int::assign_from(&mut work[c], &left);
                }
            }

            let pivot_col = (0..cols).find(|&c| !work[c].is_zero());
            let Some(pivot_col) = pivot_col else {
                continue;
            };

            let mut row_gcd: Option<N::Int> = None;
            for value in work.iter() {
                if value.is_zero() {
                    continue;
                }
                let abs = value.abs().expect("exact basis selection requires abs");
                match row_gcd.as_mut() {
                    None => row_gcd = Some(abs),
                    Some(g) => g
                        .gcd_assign(&abs)
                        .expect("exact basis selection requires gcd"),
                }
            }
            if let Some(g) = row_gcd.filter(|g| *g != z0 && *g != z1) {
                for value in work.iter_mut() {
                    if value.is_zero() {
                        continue;
                    }
                    value
                        .div_assign_exact(&g)
                        .expect("exact basis selection requires exact division");
                }
            }

            if work[pivot_col].is_negative() {
                for value in work.iter_mut() {
                    if value.is_zero() {
                        continue;
                    }
                    value
                        .neg_mut()
                        .expect("exact basis selection requires negation");
                }
            }

            basis.push(work.clone());
            pivots.push(pivot_col);
            chosen_rows.push(row);
            if chosen_rows.len() == cols {
                break;
            }
        }

        (chosen_rows, pivots)
    }

    fn compute_basis_matrix_from_pivots<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, IntMatrix<N, R>>,
        pivot_rows: &[Row],
        pivot_cols: &[usize],
    ) -> BasisMatrix<N> {
        let d = cone.matrix().col_count();
        if d == 0 {
            return BasisMatrix::identity(0);
        }
        if pivot_rows.is_empty() {
            return BasisMatrix::identity(d);
        }

        let k = pivot_rows.len();
        debug_assert_eq!(k, pivot_cols.len(), "pivot rows/cols length mismatch");

        let mut a: Vec<N::Int> = vec![N::Int::zero(); k * k];
        for (i, &row) in pivot_rows.iter().enumerate() {
            let src = cone.matrix.int_row(row);
            for (j, &col) in pivot_cols.iter().enumerate() {
                N::Int::assign_from(&mut a[i * k + j], &src[col]);
            }
        }

        let mut adj: Vec<N::Int> = vec![N::Int::zero(); k * k];
        for i in 0..k {
            adj[i * k + i] = N::Int::one();
        }
        let mut pivot_scratch = <N::Int as Int>::PivotScratch::default();
        let mut det =
            bareiss_solve_det_times_matrix_in_place(&mut a, &mut adj, k, &mut pivot_scratch)
                .unwrap_or_else(|| {
                    panic!("exact basis computation failed (singular pivot submatrix)")
                });

        if det.is_negative() {
            det.neg_mut()
                .expect("exact basis computation requires negation");
            for v in adj.iter_mut() {
                if v.is_zero() {
                    continue;
                }
                v.neg_mut()
                    .expect("exact basis computation requires negation");
            }
        }

        let mut pivot_pos_of_col: Vec<Option<usize>> = vec![None; d];
        for (pos, &col) in pivot_cols.iter().enumerate() {
            pivot_pos_of_col[col] = Some(pos);
        }

        let mut data: Vec<N> = vec![N::zero(); d * d];
        let det_n = N::from_frac(det.clone(), N::Int::one());

        // Pivot columns: set the pivot block to `adj` (det * inv(A)).
        for (j_pos, &col_j) in pivot_cols.iter().enumerate() {
            for (i_pos, &col_i) in pivot_cols.iter().enumerate() {
                let v = adj[i_pos * k + j_pos].clone();
                if v.is_zero() {
                    continue;
                }
                data[col_i * d + col_j] = N::from_frac(v, N::Int::one());
            }
        }

        // Non-pivot columns: set `basis[col][col] = det` and pivot rows to `-adj * col`.
        let mut col_vec: Vec<N::Int> = vec![N::Int::zero(); k];
        for col in 0..d {
            if pivot_pos_of_col[col].is_some() {
                continue;
            }
            data[col * d + col] = det_n.clone();

            for (i_pos, &row) in pivot_rows.iter().enumerate() {
                N::Int::assign_from(&mut col_vec[i_pos], &cone.matrix.int_row(row)[col]);
            }

            for i in 0..k {
                N::Int::assign_from(&mut self.dot_acc, &N::Int::zero());
                for j in 0..k {
                    N::Int::assign_from(&mut self.dot_tmp, &adj[i * k + j]);
                    self.dot_tmp
                        .mul_assign(&col_vec[j])
                        .expect("exact basis computation requires multiplication");
                    self.dot_acc += &self.dot_tmp;
                }
                if self.dot_acc.is_zero() {
                    continue;
                }
                let mut v = self.dot_acc.clone();
                v.neg_mut()
                    .expect("exact basis computation requires negation");
                let row_idx = pivot_cols[i];
                data[row_idx * d + col] = N::from_frac(v, N::Int::one());
            }
        }

        BasisMatrix::from_flat(d, data)
    }
}

impl<N: Rat, E: Epsilon<N>, H: HalfspacePolicy<N>, ZR: crate::dd::mode::PreorderedBackend>
    Umpire<N, ZR> for IntUmpire<N, E, H>
{
    type Eps = E;
    type Scalar = N::Int;
    type MatrixData<R: Representation> = IntMatrix<N, R>;
    type RayData = H::WrappedRayData<IntRay<N, <ZR as ZeroRepr>::Set>>;
    type HalfspacePolicy = H;

    fn ingest<R: Representation>(&mut self, matrix: LpMatrix<N, R>) -> Self::MatrixData<R> {
        let mut int_rows = scaled_integer_rows(&matrix)
            .unwrap_or_else(|| panic!("exact integer umpire cannot scale matrix to integers"));

        // Reduce each row by its content to keep dot products and pivoting smaller.
        let cols = int_rows.col_count();
        for row in 0..int_rows.row_count() {
            let Some(row_slice) = int_rows.row_mut(row) else {
                continue;
            };
            if cols == 0 {
                continue;
            }
            let mut gcd: Option<N::Int> = None;
            for v in row_slice.iter() {
                if v.is_zero() {
                    continue;
                }
                let abs = v.abs().expect("exact integer row reduction requires abs");
                match gcd.as_mut() {
                    None => gcd = Some(abs),
                    Some(g) => g
                        .gcd_assign(&abs)
                        .expect("exact integer row reduction requires gcd"),
                }
            }
            let Some(g) = gcd else {
                continue;
            };
            let z0 = N::Int::zero();
            let z1 = N::Int::one();
            if g == z0 || g == z1 {
                continue;
            }
            for v in row_slice.iter_mut() {
                if v.is_zero() {
                    continue;
                }
                v.div_assign_exact(&g)
                    .expect("exact integer row reduction requires exact division");
            }
        }

        IntMatrix {
            base: matrix,
            int_rows,
        }
    }

    fn eps(&self) -> &Self::Eps {
        &self.eps
    }

    fn halfspace_policy(&mut self) -> &mut Self::HalfspacePolicy {
        &mut self.halfspace
    }

    fn zero_vector(&self, dim: usize) -> Vec<Self::Scalar> {
        vec![N::Int::zero(); dim]
    }

    fn basis_column_vector(
        &mut self,
        basis: &BasisMatrix<N>,
        col: usize,
    ) -> Vec<Self::Scalar> {
        self.convert_vector_to_int(basis.column(col))
    }

    fn normalize_vector(&mut self, vector: &mut Vec<Self::Scalar>) -> bool {
        Self::normalize_vector_in_place(vector)
    }

    fn negate_vector_in_place(&mut self, vector: &mut [Self::Scalar]) {
        for v in vector {
            if v.is_zero() {
                continue;
            }
            v.neg_mut().expect("exact integer negation requires neg_mut");
        }
    }

    fn align_vector_in_place(&mut self, reference: &[Self::Scalar], candidate: &mut [Self::Scalar]) {
        N::Int::assign_from(&mut self.dot_acc, &N::Int::zero());
        debug_assert_eq!(
            reference.len(),
            candidate.len(),
            "align_vector_in_place dimension mismatch"
        );
        for (a, b) in reference.iter().zip(candidate.iter()) {
            N::Int::assign_from(&mut self.dot_tmp, a);
            self.dot_tmp
                .mul_assign(b)
                .expect("exact integer alignment requires multiplication");
            self.dot_acc += &self.dot_tmp;
        }
        if self.dot_acc.is_negative() {
            <Self as Umpire<N, ZR>>::negate_vector_in_place(self, candidate);
        }
    }

    fn ray_vector_for_output(&self, ray_data: &Self::RayData) -> Vec<N> {
        Self::build_standard_from_vector(ray_data.vector())
    }

    fn recompute_row_order_vector<R: Representation>(
        &mut self,
        cone: &mut ConeCtx<N, R, Self::MatrixData<R>>,
        strict_rows: &RowSet,
    ) {
        self.halfspace
            .recompute_row_order_vector(&self.eps, cone, strict_rows);
    }

    fn choose_next_halfspace<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        excluded: &RowSet,
        iteration: Row,
        active_rays: usize,
    ) -> Option<Row> {
        self.halfspace
            .choose_next_halfspace(cone, excluded, iteration, active_rays)
    }

    fn rays_equivalent(&mut self, a: &Self::RayData, b: &Self::RayData) -> bool {
        a.vector() == b.vector()
    }

    fn remap_ray_after_column_reduction(
        &mut self,
        ray_data: &mut Self::RayData,
        mapping: &[Option<usize>],
        new_dim: usize,
    ) {
        let old_vector = std::mem::take(&mut ray_data.vector);
        let mut new_vector = self.take_vector(new_dim);
        new_vector.fill(N::Int::zero());
        for (old_idx, new_idx) in mapping.iter().enumerate() {
            let Some(idx) = *new_idx else {
                continue;
            };
            N::Int::assign_from(&mut new_vector[idx], &old_vector[old_idx]);
        }
        self.vector_pool.push(old_vector);
        ray_data.vector = new_vector;
        ray_data.row_signs.clear();
        ray_data.last_eval_row = None;
        ray_data.last_eval = N::Int::zero();
        ray_data.last_sign = Sign::Zero;
    }

    fn compute_initial_basis<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        tableau: &mut crate::dd::umpire::BasisInitTableau<'_, N>,
        strict_rows: &RowSet,
        equality_set: &RowSet,
    ) -> (Vec<Option<Row>>, RowSet) {
        let d = cone.matrix().col_count();
        let m = cone.matrix().row_count();
        let mut initial_ray_index = vec![None; d];
        let mut initial_halfspaces = RowSet::new(m);

        let (pivot_rows, pivot_cols) =
            self.choose_basis_rows_and_cols(cone, strict_rows, equality_set);
        for (&row, &col) in pivot_rows.iter().zip(pivot_cols.iter()) {
            if col >= d {
                continue;
            }
            initial_ray_index[col] = Some(row);
            initial_halfspaces.insert(row);
        }
        tableau.set_basis(self.compute_basis_matrix_from_pivots(cone, &pivot_rows, &pivot_cols));
        tableau.set_tableau_nonbasic_default(d);
        tableau.set_tableau_basic_col_for_row_len(m);
        tableau.clear_tableau_storage();

        (initial_ray_index, initial_halfspaces)
    }

    fn rank<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ignored_rows: &RowSet,
        ignored_cols: &ColSet,
    ) -> MatrixRank {
        self.rank_basis_int(cone, ignored_rows, ignored_cols)
    }

    #[inline(always)]
    fn sign_for_row_on_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray: &Self::RayData,
        row: Row,
    ) -> Sign {
        if let Some(sign) = ray.cached_sign(row) {
            return sign;
        }
        if let Some(sign) = ray.cached_row_sign(row) {
            return sign;
        }
        self.dot_int_in_acc(cone, row, ray.vector());
        Self::int_sign(&self.dot_acc)
    }

    fn on_ray_inserted<R: Representation>(
        &mut self,
        _cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        _relaxed: bool,
    ) {
        self.halfspace.on_ray_inserted(ray_data);
    }

    fn on_ray_removed<R: Representation>(
        &mut self,
        _cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        _relaxed: bool,
    ) {
        self.halfspace.on_ray_removed(ray_data);
    }

    fn recycle_ray_data(&mut self, ray_data: &mut Self::RayData) {
        self.halfspace.recycle_wrapped_ray_data(ray_data);
        self.vector_pool.push(std::mem::take(&mut ray_data.vector));
        let mut row_signs = std::mem::take(&mut ray_data.row_signs);
        row_signs.clear();
        self.row_sign_pool.push(row_signs);
        ray_data.zero_set_sig = 0;
        ray_data.zero_set_count = 0;
    }

    fn classify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        row: Row,
    ) -> Sign {
        if let Some(sign) = ray_data.cached_sign(row) {
            return sign;
        }

        let sign = if let Some(sign) = ray_data.cached_row_sign(row) {
            if sign == Sign::Zero {
                N::Int::assign_from(&mut self.dot_acc, &N::Int::zero());
            } else {
                self.dot_int_in_acc(cone, row, &ray_data.vector);
            }
            sign
        } else {
            self.dot_int_in_acc(cone, row, &ray_data.vector);
            Self::int_sign(&self.dot_acc)
        };

        ray_data.last_eval_row = Some(row);
        N::Int::assign_from(&mut ray_data.last_eval, &self.dot_acc);
        ray_data.last_sign = sign;
        ray_data.set_cached_row_sign(row, sign);

        sign
    }

    fn classify_vector<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        mut vector: Vec<Self::Scalar>,
        relaxed: bool,
        last_row: Option<Row>,
        zero_set: <ZR as ZeroRepr>::Set,
    ) -> Self::RayData {
        Self::normalize_vector_in_place(&mut vector);
        self.build_ray_from_vector::<ZR, R>(
            cone,
            vector,
            relaxed,
            last_row,
            zero_set,
            false,
            None,
        )
    }

    fn sign_sets_for_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        _relaxed: bool,
        force_infeasible: bool,
        negative_out: &mut RowSet,
    ) {
        let m = cone.matrix().row_count();
        negative_out.resize(m);
        negative_out.clear();

        let force_negative_row = ray_data
            .first_infeasible_row
            .filter(|_| force_infeasible && !ray_data.feasible);
        let floor_pos = force_negative_row
            .and_then(|r| cone.row_to_pos.get(r).copied())
            .filter(|pos| *pos < cone.order_vector.len());

        for (pos, &row_idx) in cone.order_vector.iter().enumerate() {
            let forced = floor_pos.is_some_and(|floor| pos >= floor);
            if forced {
                negative_out.insert(row_idx);
                continue;
            }
            self.dot_int_in_acc(cone, row_idx, ray_data.vector());
            if Self::int_sign(&self.dot_acc) == Sign::Negative {
                negative_out.insert(row_idx);
            }
        }
    }

    fn update_first_infeasible_row<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        relaxed: bool,
    ) {
        if ray_data.weakly_feasible {
            ray_data.first_infeasible_row = None;
            return;
        }

        let mut first = None;
        for &row_idx in &cone.order_vector {
            self.dot_int_in_acc(cone, row_idx, &ray_data.vector);
            let sign = Self::int_sign(&self.dot_acc);
            let kind = cone.equality_kinds[row_idx];
            if kind.weakly_violates_sign(sign, relaxed) {
                first = Some(row_idx);
                break;
            }
        }
        ray_data.first_infeasible_row = first;
    }

    fn reclassify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        relaxed: bool,
    ) {
        let last_eval_row = ray_data.last_eval_row;
        let mut last_eval = ray_data.last_eval.clone();
        let mut last_sign = ray_data.last_sign;

        let m = cone.matrix().row_count();
        if ray_data.row_signs.len() != m {
            ray_data.row_signs.resize(m, None);
        } else {
            ray_data.row_signs.fill(None);
        }
        ray_data.zero_set.ensure_domain(m);
        ray_data.zero_set.clear();
        let mut zero_set_count = 0usize;
        ray_data.first_infeasible_row = None;
        ray_data.feasible = true;
        ray_data.weakly_feasible = true;

        for &row_idx in &cone.order_vector {
            self.dot_int_in_acc(cone, row_idx, &ray_data.vector);
            let sign = Self::int_sign(&self.dot_acc);
            ray_data.set_cached_row_sign(row_idx, sign);

            if Some(row_idx) == last_eval_row {
                last_eval = self.dot_acc.clone();
                last_sign = sign;
            }

            if sign == Sign::Zero
                && let Some(id) = <ZR as ZeroRepr>::id_for_row(cone, row_idx)
            {
                ray_data.zero_set.insert(id);
                zero_set_count += 1;
            }

            let kind = cone.equality_kinds[row_idx];
            if kind.weakly_violates_sign(sign, relaxed) {
                if ray_data.first_infeasible_row.is_none() {
                    ray_data.first_infeasible_row = Some(row_idx);
                }
                ray_data.weakly_feasible = false;
            }
            if kind.violates_sign(sign, relaxed) {
                ray_data.feasible = false;
            }
        }

        if relaxed {
            ray_data.feasible = ray_data.weakly_feasible;
        }

        ray_data.zero_set_sig = ray_data.zero_set.signature_u64();
        ray_data.zero_set_count = zero_set_count;
        ray_data.last_eval_row = last_eval_row;
        ray_data.last_eval = last_eval;
        ray_data.last_sign = last_sign;
    }

    fn generate_new_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        parents: (RayId, &Self::RayData, RayId, &Self::RayData),
        row: Row,
        relaxed: bool,
        zero_set: <ZR as ZeroRepr>::Set,
    ) -> Result<Self::RayData, <ZR as ZeroRepr>::Set> {
        let (_id1, ray1, _id2, ray2) = parents;
        let mut inherited_zero_set = zero_set;
        inherited_zero_set.ensure_domain(cone.matrix().row_count());
        inherited_zero_set.copy_from(ray1.zero_set());
        inherited_zero_set.intersection_inplace(ray2.zero_set());
        if let Some(id) = <ZR as ZeroRepr>::id_for_row(cone, row) {
            inherited_zero_set.insert(id);
        }
        let mut inherited_row_signs = self.take_row_signs(cone.matrix().row_count());
        for &row_idx in &cone.order_vector {
            let parent_a_sign = ray1.cached_row_sign(row_idx).or_else(|| {
                <ZR as ZeroRepr>::zero_set_contains_row(cone, ray1.zero_set(), row_idx)
                    .then_some(Sign::Zero)
            });
            let parent_b_sign = ray2.cached_row_sign(row_idx).or_else(|| {
                <ZR as ZeroRepr>::zero_set_contains_row(cone, ray2.zero_set(), row_idx)
                    .then_some(Sign::Zero)
            });
            let (Some(parent_a_sign), Some(parent_b_sign)) = (parent_a_sign, parent_b_sign) else {
                continue;
            };
            if let Some(sign) = Self::combine_nonnegative_signs(parent_a_sign, parent_b_sign) {
                inherited_row_signs[row_idx] = Some(sign);
            }
        }
        inherited_row_signs[row] = Some(Sign::Zero);

        let (val1, val2) = match (
            ray1.last_eval_row == Some(row),
            ray2.last_eval_row == Some(row),
        ) {
            (true, true) => (ray1.last_eval.clone(), ray2.last_eval.clone()),
            (true, false) => (
                ray1.last_eval.clone(),
                self.dot_int_clone(cone, row, ray2.vector()),
            ),
            (false, true) => (
                self.dot_int_clone(cone, row, ray1.vector()),
                ray2.last_eval.clone(),
            ),
            (false, false) => self.dot2_int_clone(cone, row, ray1.vector(), ray2.vector()),
        };

        let mut a1 = val1.abs().expect("exact ray generation requires abs");
        let mut a2 = val2.abs().expect("exact ray generation requires abs");
        if a1.is_zero() && a2.is_zero() {
            return Err(inherited_zero_set);
        }

        let mut g = a1.clone();
        g.gcd_assign(&a2)
            .expect("exact ray generation requires gcd");
        let z1 = N::Int::one();
        if !g.is_zero() && g != z1 {
            a1.div_assign_exact(&g)
                .expect("exact ray generation requires exact division");
            a2.div_assign_exact(&g)
                .expect("exact ray generation requires exact division");
        }

        let dim = ray1.vector().len();
        if dim != ray2.vector().len() {
            return Err(inherited_zero_set);
        }

        let mut new_vector = self.take_vector(dim);
        for i in 0..dim {
            N::Int::assign_from(&mut self.dot_acc, &ray1.vector()[i]);
            self.dot_acc
                .mul_assign(&a2)
                .expect("exact ray generation requires multiplication");
            N::Int::assign_from(&mut self.dot_tmp, &ray2.vector()[i]);
            self.dot_tmp
                .mul_assign(&a1)
                .expect("exact ray generation requires multiplication");
            self.dot_acc += &self.dot_tmp;
            N::Int::assign_from(&mut new_vector[i], &self.dot_acc);
        }

        if !Self::normalize_vector_in_place(&mut new_vector) {
            self.vector_pool.push(new_vector);
            return Err(inherited_zero_set);
        }

        Ok(self.build_ray_from_vector::<ZR, R>(
            cone,
            new_vector,
            relaxed,
            Some(row),
            inherited_zero_set,
            true,
            Some(inherited_row_signs),
        ))
    }
}
