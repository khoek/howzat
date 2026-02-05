use crate::dd::DefaultNormalizer;
use crate::dd::{Ray, RayClass, RayId};
use crate::dd::zero::{ZeroRepr, ZeroSet as _};
use crate::matrix::{LpMatrix, Matrix};
use calculo::linalg;
use calculo::num::{CoerceFrom, Epsilon, Normalizer, Num, Sign};
use hullabaloo::types::{Representation, Row, RowSet};
use std::cmp::Ordering;

use super::policies::{HalfspacePolicy, LexMin};
use super::{ConeCtx, Umpire};

pub trait Purifier<N: Num>: Clone {
    const ENABLED: bool = true;

    fn purify<E: Epsilon<N>>(
        &mut self,
        rows: &Matrix<N>,
        eps: &E,
        row: Row,
        ray1_zero_set: &RowSet,
        ray2_zero_set: &RowSet,
    ) -> Option<Vec<N>>;

    #[inline(always)]
    fn purify_from_zero_set<E: Epsilon<N>>(
        &mut self,
        _rows: &Matrix<N>,
        _eps: &E,
        _expected_zero: &RowSet,
    ) -> Option<Vec<N>> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoPurifier;

impl<N: Num> Purifier<N> for NoPurifier {
    const ENABLED: bool = false;

    #[inline(always)]
    fn purify<E: Epsilon<N>>(
        &mut self,
        _rows: &Matrix<N>,
        _eps: &E,
        _row: Row,
        _ray1_zero_set: &RowSet,
        _ray2_zero_set: &RowSet,
    ) -> Option<Vec<N>> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct SnapPurifier {
    expected_zero: RowSet,
}

impl SnapPurifier {
    #[inline]
    pub fn new() -> Self {
        Self {
            expected_zero: RowSet::new(0),
        }
    }
}

impl Default for SnapPurifier {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Num> Purifier<N> for SnapPurifier {
    fn purify<E: Epsilon<N>>(
        &mut self,
        rows: &Matrix<N>,
        eps: &E,
        row: Row,
        ray1_zero_set: &RowSet,
        ray2_zero_set: &RowSet,
    ) -> Option<Vec<N>> {
        self.expected_zero.copy_from(ray1_zero_set);
        self.expected_zero.intersection_inplace(ray2_zero_set);
        self.expected_zero.insert(row);
        rows.solve_nullspace_1d(&self.expected_zero, eps)
    }

    fn purify_from_zero_set<E: Epsilon<N>>(
        &mut self,
        rows: &Matrix<N>,
        eps: &E,
        expected_zero: &RowSet,
    ) -> Option<Vec<N>> {
        rows.solve_nullspace_1d(expected_zero, eps)
    }
}

#[derive(Clone, Debug)]
pub struct UpcastingSnapPurifier<M: Num, EM: Epsilon<M>> {
    expected_zero: RowSet,
    eps: EM,
    phantom: std::marker::PhantomData<M>,
}

impl<M: Num, EM: Epsilon<M>> UpcastingSnapPurifier<M, EM> {
    pub fn new(eps: EM) -> Self {
        Self {
            expected_zero: RowSet::new(0),
            eps,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<N, M, EM> Purifier<N> for UpcastingSnapPurifier<M, EM>
where
    N: Num + CoerceFrom<M>,
    M: Num + CoerceFrom<N>,
    EM: Epsilon<M> + Clone,
{
    fn purify<E: Epsilon<N>>(
        &mut self,
        rows: &Matrix<N>,
        _eps: &E,
        row: Row,
        ray1_zero_set: &RowSet,
        ray2_zero_set: &RowSet,
    ) -> Option<Vec<N>> {
        self.expected_zero.copy_from(ray1_zero_set);
        self.expected_zero.intersection_inplace(ray2_zero_set);
        self.expected_zero.insert(row);

        let cols = rows.col_count();
        if cols == 0 {
            return None;
        }

        let selected = self.expected_zero.cardinality();
        if selected == 0 {
            return None;
        }

        let mut data = Vec::with_capacity(selected * cols);
        for row_id in self.expected_zero.iter() {
            let src = &rows[row_id.as_index()];
            debug_assert_eq!(
                src.len(),
                cols,
                "matrix row width mismatch (row={}, got={}, expected={})",
                row_id.as_index(),
                src.len(),
                cols
            );
            for v in src.iter() {
                data.push(M::coerce_from(v).ok()?);
            }
        }

        let lifted = Matrix::from_flat(selected, cols, data);
        let lifted_rows = RowSet::all(selected);
        let purified = lifted.solve_nullspace_1d(&lifted_rows, &self.eps)?;
        purified.iter().map(|v| N::coerce_from(v).ok()).collect()
    }

    fn purify_from_zero_set<E: Epsilon<N>>(
        &mut self,
        rows: &Matrix<N>,
        _eps: &E,
        expected_zero: &RowSet,
    ) -> Option<Vec<N>> {
        let cols = rows.col_count();
        if cols == 0 {
            return None;
        }

        let selected = expected_zero.cardinality();
        if selected == 0 {
            return None;
        }

        let mut data = Vec::with_capacity(selected * cols);
        for row_id in expected_zero.iter() {
            let src = &rows[row_id.as_index()];
            debug_assert_eq!(
                src.len(),
                cols,
                "matrix row width mismatch (row={}, got={}, expected={})",
                row_id.as_index(),
                src.len(),
                cols
            );
            for v in src.iter() {
                data.push(M::coerce_from(v).ok()?);
            }
        }

        let lifted = Matrix::from_flat(selected, cols, data);
        let lifted_rows = RowSet::all(selected);
        let purified = lifted.solve_nullspace_1d(&lifted_rows, &self.eps)?;
        purified.iter().map(|v| N::coerce_from(v).ok()).collect()
    }
}

#[derive(Clone, Debug)]
pub struct SinglePrecisionUmpire<
    N: Num,
    E: Epsilon<N> = calculo::num::DynamicEpsilon<N>,
    NM: Normalizer<N> = <N as DefaultNormalizer>::Norm,
    H: HalfspacePolicy<N> = LexMin,
    P: Purifier<N> = NoPurifier,
> {
    eps: E,
    normalizer: NM,
    halfspace: H,
    purifier: P,
    expected_zero: RowSet,
    vector_pool: Vec<Vec<N>>,
}

impl<N: DefaultNormalizer, E: Epsilon<N>>
    SinglePrecisionUmpire<N, E, <N as DefaultNormalizer>::Norm, LexMin, NoPurifier>
{
    pub fn new(eps: E) -> Self {
        Self::with_normalizer(eps, <N as DefaultNormalizer>::Norm::default())
    }
}

impl<N: Num, E: Epsilon<N>, NM: Normalizer<N>> SinglePrecisionUmpire<N, E, NM, LexMin, NoPurifier>
{
    pub fn with_normalizer(eps: E, normalizer: NM) -> Self {
        Self {
            eps,
            normalizer,
            halfspace: LexMin,
            purifier: NoPurifier,
            expected_zero: RowSet::new(0),
            vector_pool: Vec::new(),
        }
    }
}

impl<N: Num, E: Epsilon<N>, NM: Normalizer<N>, P: Purifier<N>>
    SinglePrecisionUmpire<N, E, NM, LexMin, P>
{
    pub fn with_purifier(eps: E, normalizer: NM, purifier: P) -> Self {
        Self {
            eps,
            normalizer,
            halfspace: LexMin,
            purifier,
            expected_zero: RowSet::new(0),
            vector_pool: Vec::new(),
        }
    }
}

impl<N: DefaultNormalizer, E: Epsilon<N>, H: HalfspacePolicy<N>>
    SinglePrecisionUmpire<N, E, <N as DefaultNormalizer>::Norm, H, NoPurifier>
{
    pub fn with_halfspace_policy(eps: E, halfspace: H) -> Self {
        Self {
            eps,
            normalizer: <N as DefaultNormalizer>::Norm::default(),
            halfspace,
            purifier: NoPurifier,
            expected_zero: RowSet::new(0),
            vector_pool: Vec::new(),
        }
    }
}

impl<N: Num, E: Epsilon<N>, NM: Normalizer<N>, H: HalfspacePolicy<N>, P: Purifier<N>>
    SinglePrecisionUmpire<N, E, NM, H, P>
{
    pub fn eps(&self) -> &E {
        &self.eps
    }

    pub fn normalizer(&mut self) -> &mut NM {
        &mut self.normalizer
    }

    pub fn eps_and_normalizer(&mut self) -> (&E, &mut NM) {
        (&self.eps, &mut self.normalizer)
    }

    #[inline(always)]
    fn normalize_vector(&mut self, vector: &mut [N]) -> bool {
        let (eps, normalizer) = self.eps_and_normalizer();
        normalizer.normalize(eps, vector)
    }

    #[inline]
    fn take_vector(&mut self, dim: usize) -> Vec<N> {
        let mut v = self.vector_pool.pop().unwrap_or_default();
        if v.len() != dim {
            v.resize(dim, N::zero());
        }
        v
    }
}

impl<
    N: Num,
    E: Epsilon<N>,
    NM: Normalizer<N>,
    H: HalfspacePolicy<N>,
    P: Purifier<N>,
    ZR: crate::dd::mode::PreorderedBackend,
> Umpire<N, ZR> for SinglePrecisionUmpire<N, E, NM, H, P>
{
    type Eps = E;
    type Scalar = N;
    type MatrixData<R: Representation> = LpMatrix<N, R>;
    type RayData = H::WrappedRayData<Ray<N, <ZR as ZeroRepr>::Set>>;
    type HalfspacePolicy = H;

    #[inline(always)]
    fn ingest<R: Representation>(&mut self, matrix: LpMatrix<N, R>) -> Self::MatrixData<R> {
        matrix
    }

    fn eps(&self) -> &Self::Eps {
        &self.eps
    }

    fn halfspace_policy(&mut self) -> &mut Self::HalfspacePolicy {
        &mut self.halfspace
    }

    fn zero_vector(&self, dim: usize) -> Vec<Self::Scalar> {
        vec![N::zero(); dim]
    }

    fn basis_column_vector(&mut self, basis: &hullabaloo::matrix::BasisMatrix<N>, col: usize) -> Vec<Self::Scalar> {
        basis.column(col)
    }

    fn normalize_vector(&mut self, vector: &mut Vec<Self::Scalar>) -> bool {
        let (eps, normalizer) = self.eps_and_normalizer();
        normalizer.normalize(eps, vector)
    }

    fn negate_vector_in_place(&mut self, vector: &mut [Self::Scalar]) {
        for v in vector {
            *v = v.ref_neg();
        }
    }

    fn align_vector_in_place(&mut self, reference: &[Self::Scalar], candidate: &mut [Self::Scalar]) {
        let align = linalg::dot(reference, candidate);
        if self.eps.sign(&align) == Sign::Negative {
            <Self as Umpire<N, ZR>>::negate_vector_in_place(self, candidate);
        }
    }

    fn ray_vector_for_output(&self, ray_data: &Self::RayData) -> Vec<N> {
        ray_data.vector.clone()
    }

    fn wants_initial_purification(&self) -> bool {
        P::ENABLED
    }

    fn purify_vector_from_zero_set<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        expected_zero: &RowSet,
    ) -> Option<Vec<Self::Scalar>> {
        if !P::ENABLED {
            return None;
        }
        self.purifier
            .purify_from_zero_set(cone.matrix().storage(), &self.eps, expected_zero)
    }

    fn rays_equivalent(&mut self, a: &Self::RayData, b: &Self::RayData) -> bool {
        let eps = self.eps();
        let va = a.vector();
        let vb = b.vector();
        va.len() == vb.len()
            && va
                .iter()
                .zip(vb.iter())
                .all(|(lhs, rhs)| eps.cmp(lhs, rhs) == Ordering::Equal)
    }

    fn remap_ray_after_column_reduction(
        &mut self,
        ray_data: &mut Self::RayData,
        mapping: &[Option<usize>],
        new_dim: usize,
    ) {
        let mut new_vec = vec![N::zero(); new_dim];
        for (old_idx, new_idx) in mapping.iter().enumerate() {
            let Some(idx) = *new_idx else {
                continue;
            };
            new_vec[idx] = ray_data.vector[old_idx].clone();
        }
        ray_data.vector = new_vec;
        ray_data.class.last_eval_row = None;
        ray_data.class.last_eval = N::zero();
        ray_data.class.last_sign = Sign::Zero;
    }

    fn sign_for_row_on_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray: &Self::RayData,
        row: Row,
    ) -> Sign {
        if ray.class.last_eval_row == Some(row) {
            return ray.class.last_sign;
        }
        let value = cone.row_value(row, ray.vector());
        self.eps.sign(&value)
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
        ray_data.class.zero_set_sig = 0;
        ray_data.class.zero_set_count = 0;
    }

    fn classify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        row: Row,
    ) -> Sign {
        if ray_data.class.last_eval_row == Some(row) {
            return ray_data.class.last_sign;
        }
        let row_vec = &cone.matrix().storage()[row];
        let value = linalg::dot(row_vec, &ray_data.vector);
        let sign = self.eps.sign(&value);
        ray_data.class.last_eval_row = Some(row);
        ray_data.class.last_eval = value;
        ray_data.class.last_sign = sign;
        sign
    }

    fn classify_vector<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        vector: Vec<Self::Scalar>,
        relaxed: bool,
        last_row: Option<Row>,
        mut zero_set: <ZR as ZeroRepr>::Set,
    ) -> Self::RayData {
        let m = cone.matrix().row_count();
        let mut negative_rows = self.halfspace.take_negative_rows(m);
        let track_negatives = negative_rows.len() == m;

        zero_set.ensure_domain(m);
        zero_set.clear();
        let mut zero_set_count = 0usize;
        let mut feasible = true;
        let mut weakly_feasible = true;
        let mut first_infeasible_row = None;
        let mut last_eval = None;

        let matrix = cone.matrix().storage();
        for &row_idx in cone.order_vector.iter() {
            let row_vec = &matrix[row_idx];
            let value = linalg::dot(row_vec, &vector);
            if Some(row_idx) == last_row {
                last_eval = Some(value.clone());
            }
            let sign = self.eps.sign(&value);
            if track_negatives && sign == Sign::Negative {
                negative_rows.insert(row_idx);
            }
            if sign == Sign::Zero
                && let Some(id) = <ZR as ZeroRepr>::id_for_row(cone, row_idx)
            {
                zero_set.insert(id);
                zero_set_count += 1;
            }
            let kind = cone.equality_kinds[row_idx];
            let weak_violation = kind.weakly_violates_sign(sign, relaxed);
            if weak_violation {
                if first_infeasible_row.is_none() {
                    first_infeasible_row = Some(row_idx);
                }
                weakly_feasible = false;
            }
            let strict_violation = kind.violates_sign(sign, relaxed);
            if strict_violation {
                feasible = false;
            }
        }

        if relaxed {
            feasible = weakly_feasible;
        }

        let zero_set_sig = zero_set.signature_u64();
        let last_eval_row = last_row;
        let last_eval = last_eval.unwrap_or_else(|| {
            last_eval_row
                .map(|row| cone.row_value(row, &vector))
                .unwrap_or_else(N::zero)
        });
        let last_sign = self.eps.sign(&last_eval);

        let ray_data = Ray {
            vector,
            class: RayClass {
                zero_set,
                zero_set_sig,
                zero_set_count,
                first_infeasible_row,
                feasible,
                weakly_feasible,
                last_eval_row,
                last_eval,
                last_sign,
            },
        };

        self.halfspace.wrap_ray_data(ray_data, negative_rows)
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

        let matrix = cone.matrix().storage();
        let force_negative_row = ray_data
            .class
            .first_infeasible_row
            .filter(|_| force_infeasible && !ray_data.class.feasible);
        let floor_pos = force_negative_row
            .and_then(|r| cone.row_to_pos.get(r).copied())
            .filter(|pos| *pos < cone.order_vector.len());

        for (pos, &row_idx) in cone.order_vector.iter().enumerate() {
            let forced = floor_pos.is_some_and(|floor| pos >= floor);
            if forced {
                negative_out.insert(row_idx);
                continue;
            }
            let row_vec = &matrix[row_idx];
            let value = linalg::dot(row_vec, &ray_data.vector);
            let sign = self.eps.sign(&value);
            if sign == Sign::Negative {
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
        if ray_data.class.weakly_feasible {
            ray_data.class.first_infeasible_row = None;
            return;
        }

        let mut first = None;
        let matrix = cone.matrix().storage();
        for &row_idx in cone.order_vector.iter() {
            let row_vec = &matrix[row_idx];
            let value = linalg::dot(row_vec, &ray_data.vector);
            let sign = self.eps.sign(&value);
            let kind = cone.equality_kinds[row_idx];
            if kind.weakly_violates_sign(sign, relaxed) {
                first = Some(row_idx);
                break;
            }
        }
        ray_data.class.first_infeasible_row = first;
    }

    fn reclassify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        relaxed: bool,
    ) {
        let last_eval_row = ray_data.class.last_eval_row;
        let mut last_eval = ray_data.class.last_eval.clone();
        let mut last_sign = ray_data.class.last_sign;

        let matrix = cone.matrix().storage();
        ray_data.class.zero_set.clear();
        let mut zero_set_count = 0usize;
        ray_data.class.first_infeasible_row = None;
        ray_data.class.feasible = true;
        ray_data.class.weakly_feasible = true;

        for &row_idx in cone.order_vector.iter() {
            let row_vec = &matrix[row_idx];
            let value = linalg::dot(row_vec, &ray_data.vector);
            let sign = self.eps.sign(&value);
            if Some(row_idx) == last_eval_row {
                last_eval = value.clone();
                last_sign = sign;
            }
            if sign == Sign::Zero
                && let Some(id) = <ZR as ZeroRepr>::id_for_row(cone, row_idx)
            {
                ray_data.class.zero_set.insert(id);
                zero_set_count += 1;
            }

            let kind = cone.equality_kinds[row_idx];
            if kind.weakly_violates_sign(sign, relaxed) {
                if ray_data.class.first_infeasible_row.is_none() {
                    ray_data.class.first_infeasible_row = Some(row_idx);
                }
                ray_data.class.weakly_feasible = false;
            }
            if kind.violates_sign(sign, relaxed) {
                ray_data.class.feasible = false;
            }
        }

        if relaxed {
            ray_data.class.feasible = ray_data.class.weakly_feasible;
        }

        ray_data.class.last_eval_row = last_eval_row;
        ray_data.class.last_eval = last_eval;
        ray_data.class.last_sign = last_sign;
        ray_data.class.zero_set_sig = ray_data.class.zero_set.signature_u64();
        ray_data.class.zero_set_count = zero_set_count;
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
        let val1 = if ray1.class.last_eval_row == Some(row) {
            ray1.class.last_eval.clone()
        } else {
            let row_vec = &cone.matrix().storage()[row];
            linalg::dot(row_vec, &ray1.vector)
        };
        let val2 = if ray2.class.last_eval_row == Some(row) {
            ray2.class.last_eval.clone()
        } else {
            let row_vec = &cone.matrix().storage()[row];
            linalg::dot(row_vec, &ray2.vector)
        };

        let a1 = val1.abs();
        let a2 = val2.abs();

        let mut new_vector = self.take_vector(ray1.vector.len());
        linalg::lin_comb2_into(&mut new_vector, &ray1.vector, &a2, &ray2.vector, &a1);
        if !self.normalize_vector(&mut new_vector) {
            return Err(zero_set);
        }

        if P::ENABLED {
            <ZR as ZeroRepr>::fill_purify_expected_zero(
                cone,
                &mut self.expected_zero,
                row,
                ray1.zero_set(),
                ray2.zero_set(),
            );
            if let Some(mut purified) =
                self.purifier
                    .purify_from_zero_set(cone.matrix().storage(), &self.eps, &self.expected_zero)
                    && self.normalize_vector(&mut purified)
            {
                let align = linalg::dot(&purified, &new_vector);
                if self.eps.sign(&align) == Sign::Negative {
                    for v in &mut purified {
                        *v = v.ref_neg();
                    }
                }
                new_vector = purified;
            }
        }

        Ok(<Self as Umpire<N, ZR>>::classify_vector::<R>(
            self,
            cone,
            new_vector,
            relaxed,
            Some(row),
            zero_set,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{SinglePrecisionUmpire, SnapPurifier, UpcastingSnapPurifier};
    use crate::dd::ray::RayNoNegatives;
    use crate::dd::{ConeCtx, Ray, RayClass, RayId, SatSet, Umpire};
    use crate::matrix::LpMatrix;
    use calculo::num::{DynamicEpsilon, Epsilon, NoNormalizer, Num, Sign};
    use hullabaloo::types::{Inequality, InequalityKind};

    #[test]
    fn generate_new_ray_survives_no_normalizer() {
        let eps = DynamicEpsilon::new(0.0);
        let matrix =
            LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let cone = ConeCtx {
            matrix,
            equality_kinds: vec![InequalityKind::Inequality; 2],
            order_vector: vec![0, 1],
            row_to_pos: vec![0, 1],
            lex_order_cache: None,
            sat_row_to_id: vec![Some(0usize.into()), Some(1usize.into())],
            sat_id_to_row: vec![0, 1],
            _phantom: std::marker::PhantomData,
        };

        let mut zero = SatSet::default();
        zero.insert(0usize.into());
        let zero_sig = zero.signature_u64();
        let class = RayClass {
            zero_set: zero.clone(),
            zero_set_sig: zero_sig,
            zero_set_count: 1,
            first_infeasible_row: None,
            feasible: true,
            weakly_feasible: true,
            last_eval_row: None,
            last_eval: 0.0,
            last_sign: Sign::Zero,
        };
        let ray1 = RayNoNegatives {
            inner: Ray {
                vector: vec![1e-3, -1.0, 1.0],
                class: class.clone(),
            },
        };
        let ray2 = RayNoNegatives {
            inner: Ray {
                vector: vec![1e-3, 2.0, 1.0],
                class,
            },
        };

        let mut umpire = SinglePrecisionUmpire::with_normalizer(eps, NoNormalizer);
        let new_ray = <_ as Umpire<f64>>::generate_new_ray::<Inequality>(
            &mut umpire,
            &cone,
            (RayId(0), &ray1, RayId(1), &ray2),
            1,
            false,
            SatSet::default(),
        );
        assert!(new_ray.is_ok());
    }

    #[test]
    fn snap_purifier_reconstructs_ray_from_zero_sets() {
        let eps = f64::default_eps();
        let matrix =
            LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let cone = ConeCtx {
            matrix,
            equality_kinds: vec![InequalityKind::Inequality; 2],
            order_vector: vec![0, 1],
            row_to_pos: vec![0, 1],
            lex_order_cache: None,
            sat_row_to_id: vec![Some(0usize.into()), Some(1usize.into())],
            sat_id_to_row: vec![0, 1],
            _phantom: std::marker::PhantomData,
        };

        let mut zero = SatSet::default();
        zero.insert(0usize.into());
        let zero_sig = zero.signature_u64();
        let class = RayClass {
            zero_set: zero.clone(),
            zero_set_sig: zero_sig,
            zero_set_count: 1,
            first_infeasible_row: None,
            feasible: true,
            weakly_feasible: true,
            last_eval_row: None,
            last_eval: 0.0,
            last_sign: Sign::Zero,
        };
        let ray1 = RayNoNegatives {
            inner: Ray {
                vector: vec![1e-3, -1.0, 1.0],
                class: class.clone(),
            },
        };
        let ray2 = RayNoNegatives {
            inner: Ray {
                vector: vec![1e-3, 2.0, 1.0],
                class,
            },
        };

        let mut umpire =
            SinglePrecisionUmpire::with_purifier(eps.clone(), NoNormalizer, SnapPurifier::new());
        let ray = <_ as Umpire<f64>>::generate_new_ray::<Inequality>(
            &mut umpire,
            &cone,
            (RayId(0), &ray1, RayId(1), &ray2),
            1,
            false,
            SatSet::default(),
        )
        .expect("ray should be generated");

        assert!(eps.is_zero(&ray.vector[0]));
        assert!(eps.is_zero(&ray.vector[1]));
        assert!(!eps.is_zero(&ray.vector[2]));
        assert!(ray.zero_set().contains(0usize.into()));
        assert!(ray.zero_set().contains(1usize.into()));
    }

    #[test]
    #[cfg(feature = "rug")]
    fn upcasting_snap_purifier_reconstructs_ray_from_zero_sets() {
        use calculo::num::RugRat;

        let eps = f64::default_eps();
        let matrix =
            LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let cone = ConeCtx {
            matrix,
            equality_kinds: vec![InequalityKind::Inequality; 2],
            order_vector: vec![0, 1],
            row_to_pos: vec![0, 1],
            lex_order_cache: None,
            sat_row_to_id: vec![Some(0usize.into()), Some(1usize.into())],
            sat_id_to_row: vec![0, 1],
            _phantom: std::marker::PhantomData,
        };

        let mut zero = SatSet::default();
        zero.insert(0usize.into());
        let zero_sig = zero.signature_u64();
        let class = RayClass {
            zero_set: zero.clone(),
            zero_set_sig: zero_sig,
            zero_set_count: 1,
            first_infeasible_row: None,
            feasible: true,
            weakly_feasible: true,
            last_eval_row: None,
            last_eval: 0.0,
            last_sign: Sign::Zero,
        };
        let ray1 = RayNoNegatives {
            inner: Ray {
                vector: vec![1e-3, -1.0, 1.0],
                class: class.clone(),
            },
        };
        let ray2 = RayNoNegatives {
            inner: Ray {
                vector: vec![1e-3, 2.0, 1.0],
                class,
            },
        };

        let mut umpire = SinglePrecisionUmpire::with_purifier(
            eps.clone(),
            NoNormalizer,
            UpcastingSnapPurifier::new(DynamicEpsilon::new(RugRat::zero())),
        );
        let ray = <_ as Umpire<f64>>::generate_new_ray::<Inequality>(
            &mut umpire,
            &cone,
            (RayId(0), &ray1, RayId(1), &ray2),
            1,
            false,
            SatSet::default(),
        )
        .expect("ray should be generated");

        assert!(eps.is_zero(&ray.vector[0]));
        assert!(eps.is_zero(&ray.vector[1]));
        assert!(!eps.is_zero(&ray.vector[2]));
        assert!(ray.zero_set().contains(0usize.into()));
        assert!(ray.zero_set().contains(1usize.into()));
    }
}
