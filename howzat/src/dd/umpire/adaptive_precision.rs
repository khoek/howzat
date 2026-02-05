use crate::dd::DefaultNormalizer;
use crate::dd::zero::{ZeroRepr, ZeroSet};
use crate::dd::{Ray, RayClass, RayId};
use crate::matrix::LpMatrix;
use calculo::linalg;
use calculo::num::{CoerceFrom, Epsilon, Normalizer, Num, Sign};
use hullabaloo::types::{Representation, Row, RowSet};

use super::multi_precision::{CachedRay, ShadowedMatrix};
use super::policies::{HalfspacePolicy, LexMin};
use super::{ConeCtx, Umpire};
use std::cmp::Ordering;

const MAX_NEAR_ZERO_ROWS: usize = 64;

/// Adaptive-precision umpire: use fast `N` signs unless "near zero", then consult a shadow `M`.
///
/// This is intended to close the gap between `SinglePrecisionUmpire` and `MultiPrecisionUmpire`
/// by paying the shadow cost only for ambiguous evaluations.
#[derive(Clone, Debug)]
pub struct AdaptivePrecisionUmpire<
    N: Num,
    M: Num,
    E: Epsilon<N> = calculo::num::DynamicEpsilon<N>,
    ET: Epsilon<N> = calculo::num::DynamicEpsilon<N>,
    EM: Epsilon<M> = calculo::num::DynamicEpsilon<M>,
    NM: Normalizer<N> = <N as DefaultNormalizer>::Norm,
    H: HalfspacePolicy<N> = LexMin,
> {
    eps: E,
    trigger_eps: ET,
    shadow_eps: EM,
    normalizer: NM,
    halfspace: H,
    standard_vector_pool: Vec<Vec<N>>,
    shadow_vector_pool: Vec<Vec<M>>,
    near_zero_rows_pool: Vec<Vec<Row>>,
}

impl<
    N: DefaultNormalizer,
    M: Num + CoerceFrom<N>,
    E: Epsilon<N>,
    ET: Epsilon<N>,
    EM: Epsilon<M>,
> AdaptivePrecisionUmpire<N, M, E, ET, EM, <N as DefaultNormalizer>::Norm, LexMin>
{
    pub fn new(eps: E, trigger_eps: ET, shadow_eps: EM) -> Self {
        Self::with_halfspace_policy(eps, trigger_eps, shadow_eps, LexMin)
    }
}

impl<
    N: Num,
    M: Num + CoerceFrom<N>,
    E: Epsilon<N>,
    ET: Epsilon<N>,
    EM: Epsilon<M>,
    NM: Normalizer<N>,
> AdaptivePrecisionUmpire<N, M, E, ET, EM, NM, LexMin>
{
    pub fn with_normalizer(eps: E, trigger_eps: ET, shadow_eps: EM, normalizer: NM) -> Self {
        Self {
            eps,
            trigger_eps,
            shadow_eps,
            normalizer,
            halfspace: LexMin,
            standard_vector_pool: Vec::new(),
            shadow_vector_pool: Vec::new(),
            near_zero_rows_pool: Vec::new(),
        }
    }
}

impl<
    N: DefaultNormalizer,
    M: Num + CoerceFrom<N>,
    E: Epsilon<N>,
    ET: Epsilon<N>,
    EM: Epsilon<M>,
    H: HalfspacePolicy<N>,
> AdaptivePrecisionUmpire<N, M, E, ET, EM, <N as DefaultNormalizer>::Norm, H>
{
    pub fn with_halfspace_policy(eps: E, trigger_eps: ET, shadow_eps: EM, halfspace: H) -> Self {
        Self {
            eps,
            trigger_eps,
            shadow_eps,
            normalizer: <N as DefaultNormalizer>::Norm::default(),
            halfspace,
            standard_vector_pool: Vec::new(),
            shadow_vector_pool: Vec::new(),
            near_zero_rows_pool: Vec::new(),
        }
    }
}

impl<
    N: Num + 'static,
    M: Num + CoerceFrom<N> + 'static,
    E: Epsilon<N>,
    ET: Epsilon<N>,
    EM: Epsilon<M>,
    NM: Normalizer<N>,
    H: HalfspacePolicy<N>,
> AdaptivePrecisionUmpire<N, M, E, ET, EM, NM, H>
{
    #[inline(always)]
    fn shadow_sign(&self, value: &M) -> Sign {
        self.shadow_eps.sign(value)
    }

    #[inline(always)]
    fn should_consult_shadow(&self, base_value: &N, base_sign: Sign) -> bool {
        base_sign == Sign::Zero || self.trigger_eps.is_zero(base_value)
    }

    #[inline(always)]
    fn adaptive_sign<R: Representation>(
        &self,
        cone: &ConeCtx<N, R, ShadowedMatrix<N, M, R>>,
        shadow_vec: &[M],
        row: Row,
        base_value: &N,
        base_sign: Sign,
    ) -> Sign {
        if !self.should_consult_shadow(base_value, base_sign) {
            return base_sign;
        }
        self.shadow_sign(&cone.matrix.shadow_row_value(row, shadow_vec))
    }

    #[inline]
    fn take_standard_vector(&mut self, dim: usize) -> Vec<N> {
        let mut v = self.standard_vector_pool.pop().unwrap_or_default();
        if v.len() != dim {
            v.resize(dim, N::zero());
        }
        v
    }

    #[inline]
    fn take_shadow_vector(&mut self, dim: usize) -> Vec<M> {
        let mut v = self.shadow_vector_pool.pop().unwrap_or_default();
        if v.len() != dim {
            v.resize(dim, M::zero());
        }
        v
    }

    #[inline]
    fn take_near_zero_rows(&mut self) -> Vec<Row> {
        self.near_zero_rows_pool.pop().unwrap_or_default()
    }

    fn build_ray_from_shadow<ZR: ZeroRepr, R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, ShadowedMatrix<N, M, R>>,
        standard_vector: Vec<N>,
        shadow: Vec<M>,
        relaxed: bool,
        last_row: Option<Row>,
        mut zero_set: <ZR as ZeroRepr>::Set,
    ) -> H::WrappedRayData<CachedRay<N, M, <ZR as ZeroRepr>::Set>> {
        let m = cone.matrix.base.row_count();
        let mut negative_rows = self.halfspace.take_negative_rows(m);
        let track_negatives = negative_rows.len() == m;

        zero_set.ensure_domain(m);
        zero_set.clear();
        let mut zero_set_count = 0usize;
        let mut feasible = true;
        let mut weakly_feasible = true;
        let mut first_infeasible_row = None;

        let mut standard_last_eval: Option<N> = None;
        let mut shadow_last_eval_row: Option<Row> = None;
        let mut shadow_last_eval = M::zero();
        let mut shadow_last_sign = Sign::Zero;
        let mut last_sign = Sign::Zero;

        let mut near_zero_rows = self.take_near_zero_rows();
        near_zero_rows.clear();
        let mut near_zero_truncated = false;

        for &row_idx in &cone.order_vector {
            let base_value = cone.row_value(row_idx, &standard_vector);
            if Some(row_idx) == last_row {
                standard_last_eval = Some(base_value.clone());
            }

            let base_sign = self.eps.sign(&base_value);
            let consulted_shadow = self.should_consult_shadow(&base_value, base_sign);
            let sign = if consulted_shadow {
                let shadow_value = cone.matrix.shadow_row_value(row_idx, &shadow);
                let sign = self.shadow_sign(&shadow_value);
                if Some(row_idx) == last_row {
                    shadow_last_eval_row = Some(row_idx);
                    shadow_last_eval = shadow_value.clone();
                    shadow_last_sign = sign;
                }
                sign
            } else {
                base_sign
            };
            if consulted_shadow {
                if near_zero_rows.len() < MAX_NEAR_ZERO_ROWS {
                    near_zero_rows.push(row_idx);
                } else {
                    near_zero_truncated = true;
                }
            }
            if Some(row_idx) == last_row {
                last_sign = sign;
            }
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

        let standard_last_eval = match last_row {
            Some(row_idx) => {
                standard_last_eval.unwrap_or_else(|| cone.row_value(row_idx, &standard_vector))
            }
            None => N::zero(),
        };

        let zero_set_sig = zero_set.signature_u64();
        let standard = Ray {
            vector: standard_vector,
            class: RayClass {
                zero_set,
                zero_set_sig,
                zero_set_count,
                first_infeasible_row,
                feasible,
                weakly_feasible,
                last_eval_row: last_row,
                last_eval: standard_last_eval,
                last_sign,
            },
        };

        let ray_data = CachedRay {
            standard,
            shadow,
            shadow_last_eval_row,
            shadow_last_eval,
            shadow_last_sign,
            near_zero_rows,
            near_zero_truncated,
        };

        self.halfspace.wrap_ray_data(ray_data, negative_rows)
    }
}

impl<
    N: Num + 'static,
    M: Num + CoerceFrom<N> + 'static,
    E: Epsilon<N>,
    ET: Epsilon<N>,
    EM: Epsilon<M>,
    NM: Normalizer<N>,
    H: HalfspacePolicy<N>,
    ZR: crate::dd::mode::PreorderedBackend,
> Umpire<N, ZR> for AdaptivePrecisionUmpire<N, M, E, ET, EM, NM, H>
{
    type Eps = E;
    type Scalar = N;
    type MatrixData<R: Representation> = ShadowedMatrix<N, M, R>;
    type RayData = H::WrappedRayData<CachedRay<N, M, <ZR as ZeroRepr>::Set>>;
    type HalfspacePolicy = H;

    fn ingest<R: Representation>(&mut self, matrix: LpMatrix<N, R>) -> Self::MatrixData<R> {
        let cols = matrix.col_count();
        let rows = matrix.row_count();
        let mut shadow = Vec::with_capacity(rows * cols);
        for row in matrix.rows() {
            for v in row {
                shadow.push(M::coerce_from(v).expect("matrix entries must be convertible"));
            }
        }
        ShadowedMatrix { base: matrix, shadow }
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
        self.normalizer.normalize(&self.eps, vector)
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
        ray_data.standard.vector.clone()
    }

    #[inline(always)]
    fn near_zero_rows_on_ray<'a>(&self, ray_data: &'a Self::RayData) -> Option<&'a [Row]> {
        Some(ray_data.near_zero_rows())
    }

    #[inline(always)]
    fn near_zero_rows_truncated_on_ray(&self, ray_data: &Self::RayData) -> bool {
        ray_data.near_zero_truncated()
    }

    fn rays_equivalent(&mut self, a: &Self::RayData, b: &Self::RayData) -> bool {
        let eps = &self.eps;
        let va = a.as_ref().vector();
        let vb = b.as_ref().vector();
        if va.len() != vb.len()
            || va
                .iter()
                .zip(vb.iter())
                .any(|(lhs, rhs)| eps.cmp(lhs, rhs) != Ordering::Equal)
        {
            return false;
        }

        a.shadow().len() == b.shadow().len()
            && a.shadow()
                .iter()
                .zip(b.shadow().iter())
                .all(|(lhs, rhs)| self.shadow_eps.cmp(lhs, rhs) == Ordering::Equal)
    }

    fn remap_ray_after_column_reduction(
        &mut self,
        ray_data: &mut Self::RayData,
        mapping: &[Option<usize>],
        new_dim: usize,
    ) {
        let old_standard = std::mem::take(&mut ray_data.standard.vector);
        let mut new_vec = self.take_standard_vector(new_dim);
        new_vec.fill(N::zero());
        for (old_idx, new_idx) in mapping.iter().enumerate() {
            let Some(idx) = *new_idx else {
                continue;
            };
            new_vec[idx] = old_standard[old_idx].clone();
        }
        self.standard_vector_pool.push(old_standard);
        ray_data.standard.vector = new_vec;
        ray_data.standard.class.last_eval_row = None;
        ray_data.standard.class.last_eval = N::zero();
        ray_data.standard.class.last_sign = Sign::Zero;

        let old_shadow = std::mem::take(&mut ray_data.shadow);
        let mut new_shadow = self.take_shadow_vector(new_dim);
        new_shadow.fill(M::zero());
        for (old_idx, new_idx) in mapping.iter().enumerate() {
            let Some(idx) = *new_idx else {
                continue;
            };
            new_shadow[idx] = old_shadow[old_idx].clone();
        }
        self.shadow_vector_pool.push(old_shadow);
        ray_data.shadow = new_shadow;
        ray_data.shadow_last_eval_row = None;
        ray_data.shadow_last_eval = M::zero();
        ray_data.shadow_last_sign = Sign::Zero;
        ray_data.near_zero_rows.clear();
        ray_data.near_zero_truncated = false;
    }

    fn recycle_ray_data(&mut self, ray_data: &mut Self::RayData) {
        self.halfspace.recycle_wrapped_ray_data(ray_data);
        self.standard_vector_pool
            .push(std::mem::take(&mut ray_data.standard.vector));
        self.shadow_vector_pool
            .push(std::mem::take(&mut ray_data.shadow));

        let mut near_zero_rows = std::mem::take(&mut ray_data.near_zero_rows);
        near_zero_rows.clear();
        self.near_zero_rows_pool.push(near_zero_rows);

        ray_data.standard.class.zero_set_sig = 0;
        ray_data.standard.class.zero_set_count = 0;
        ray_data.standard.class.first_infeasible_row = None;
        ray_data.standard.class.last_eval_row = None;
        ray_data.standard.class.last_eval = N::zero();

        ray_data.shadow_last_eval_row = None;
        ray_data.shadow_last_eval = M::zero();
        ray_data.shadow_last_sign = Sign::Zero;
        ray_data.near_zero_truncated = false;
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

    #[inline(always)]
    fn sign_for_row_on_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray: &Self::RayData,
        row: Row,
    ) -> Sign {
        let ray_ref = ray.as_ref();
        if ray_ref.class.last_eval_row == Some(row) {
            return ray_ref.class.last_sign;
        }
        if let Some(sign) = ray.cached_shadow_sign(row) {
            return sign;
        }

        let base_value = cone.row_value(row, ray_ref.vector());
        let base_sign = self.eps.sign(&base_value);
        self.adaptive_sign(cone, ray.shadow(), row, &base_value, base_sign)
    }

    fn classify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        row: Row,
    ) -> Sign {
        ray_data.ensure_shadow_matches_standard();

        if ray_data.standard.class.last_eval_row == Some(row) {
            return ray_data.standard.class.last_sign;
        }

        let base_value = cone.row_value(row, ray_data.standard.vector());
        let base_sign = self.eps.sign(&base_value);
        let consulted_shadow = self.should_consult_shadow(&base_value, base_sign);
        let sign = if consulted_shadow {
            let shadow_value = cone.matrix.shadow_row_value(row, ray_data.shadow.as_slice());
            let sign = self.shadow_sign(&shadow_value);
            ray_data.shadow_last_eval_row = Some(row);
            ray_data.shadow_last_eval = shadow_value;
            ray_data.shadow_last_sign = sign;
            sign
        } else {
            base_sign
        };
        if consulted_shadow {
            if ray_data.near_zero_rows.len() < MAX_NEAR_ZERO_ROWS {
                if !ray_data.near_zero_rows.contains(&row) {
                    ray_data.near_zero_rows.push(row);
                }
            } else {
                ray_data.near_zero_truncated = true;
            }
        }

        ray_data.standard.class.last_eval_row = Some(row);
        ray_data.standard.class.last_eval = base_value;
        ray_data.standard.class.last_sign = sign;
        sign
    }

    fn classify_vector<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        vector: Vec<Self::Scalar>,
        relaxed: bool,
        last_row: Option<Row>,
        zero_set: <ZR as ZeroRepr>::Set,
    ) -> Self::RayData {
        let mut shadow = self.take_shadow_vector(vector.len());
        for (dst, v) in shadow.iter_mut().zip(vector.iter()) {
            *dst = M::coerce_from(v).expect("ray vectors must be convertible");
        }
        self.build_ray_from_shadow::<ZR, R>(cone, vector, shadow, relaxed, last_row, zero_set)
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
            .as_ref()
            .class
            .first_infeasible_row
            .filter(|_| force_infeasible && !ray_data.as_ref().class.feasible);
        let floor_pos = force_negative_row
            .and_then(|r| cone.row_to_pos.get(r).copied())
            .filter(|pos| *pos < cone.order_vector.len());

        let standard_vec = ray_data.as_ref().vector();
        let shadow_vec = ray_data.shadow();

        for (pos, &row_idx) in cone.order_vector.iter().enumerate() {
            let forced = floor_pos.is_some_and(|floor| pos >= floor);
            if forced {
                negative_out.insert(row_idx);
                continue;
            }

            let base_value = cone.row_value(row_idx, standard_vec);
            let base_sign = self.eps.sign(&base_value);
            let sign = self.adaptive_sign(cone, shadow_vec, row_idx, &base_value, base_sign);
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
        ray_data.ensure_shadow_matches_standard();

        if ray_data.standard.class.weakly_feasible {
            ray_data.standard.class.first_infeasible_row = None;
            return;
        }

        let mut first = None;
        let standard_vec = ray_data.standard.vector();
        let shadow_vec = ray_data.shadow.as_slice();
        for &row_idx in &cone.order_vector {
            let base_value = cone.row_value(row_idx, standard_vec);
            let base_sign = self.eps.sign(&base_value);
            let sign = self.adaptive_sign(cone, shadow_vec, row_idx, &base_value, base_sign);
            let kind = cone.equality_kinds[row_idx];
            if kind.weakly_violates_sign(sign, relaxed) {
                first = Some(row_idx);
                break;
            }
        }
        ray_data.standard.class.first_infeasible_row = first;
    }

    fn reclassify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        relaxed: bool,
    ) {
        let ray_data = &mut **ray_data;
        ray_data.ensure_shadow_matches_standard();
        ray_data.near_zero_rows.clear();
        ray_data.near_zero_truncated = false;

        let last_eval_row = ray_data.standard.class.last_eval_row;
        let mut last_eval = ray_data.standard.class.last_eval.clone();
        let mut last_sign = ray_data.standard.class.last_sign;

        let shadow_vec = ray_data.shadow.as_slice();

        ray_data.standard.class.zero_set.clear();
        let mut zero_set_count = 0usize;
        ray_data.standard.class.first_infeasible_row = None;
        ray_data.standard.class.feasible = true;
        ray_data.standard.class.weakly_feasible = true;

        for &row_idx in &cone.order_vector {
            let base_value = cone.row_value(row_idx, ray_data.standard.vector());
            let base_sign = self.eps.sign(&base_value);
            let consulted_shadow = self.should_consult_shadow(&base_value, base_sign);
            let sign = if consulted_shadow {
                let shadow_value = cone.matrix.shadow_row_value(row_idx, shadow_vec);
                let sign = self.shadow_sign(&shadow_value);
                if Some(row_idx) == last_eval_row {
                    ray_data.shadow_last_eval_row = Some(row_idx);
                    ray_data.shadow_last_eval = shadow_value;
                    ray_data.shadow_last_sign = sign;
                }
                sign
            } else {
                base_sign
            };
            if consulted_shadow {
                if ray_data.near_zero_rows.len() < MAX_NEAR_ZERO_ROWS {
                    ray_data.near_zero_rows.push(row_idx);
                } else {
                    ray_data.near_zero_truncated = true;
                }
            }

            if Some(row_idx) == last_eval_row {
                last_eval = base_value.clone();
                last_sign = sign;
            }

            if sign == Sign::Zero
                && let Some(id) = <ZR as ZeroRepr>::id_for_row(cone, row_idx)
            {
                ray_data.standard.class.zero_set.insert(id);
                zero_set_count += 1;
            }

            let kind = cone.equality_kinds[row_idx];
            if kind.weakly_violates_sign(sign, relaxed) {
                if ray_data.standard.class.first_infeasible_row.is_none() {
                    ray_data.standard.class.first_infeasible_row = Some(row_idx);
                }
                ray_data.standard.class.weakly_feasible = false;
            }
            if kind.violates_sign(sign, relaxed) {
                ray_data.standard.class.feasible = false;
            }
        }

        if relaxed {
            ray_data.standard.class.feasible = ray_data.standard.class.weakly_feasible;
        }

        ray_data.standard.class.last_eval_row = last_eval_row;
        ray_data.standard.class.last_eval = last_eval;
        ray_data.standard.class.last_sign = last_sign;
        ray_data.standard.class.zero_set_sig = ray_data.standard.class.zero_set.signature_u64();
        ray_data.standard.class.zero_set_count = zero_set_count;
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

        let val1 = if ray1.as_ref().class.last_eval_row == Some(row) {
            ray1.as_ref().class.last_eval.clone()
        } else {
            cone.row_value(row, ray1.as_ref().vector())
        };
        let val2 = if ray2.as_ref().class.last_eval_row == Some(row) {
            ray2.as_ref().class.last_eval.clone()
        } else {
            cone.row_value(row, ray2.as_ref().vector())
        };

        let a1 = val1.abs();
        let a2 = val2.abs();

        let mut new_vector = self.take_standard_vector(ray1.as_ref().vector().len());
        linalg::lin_comb2_into(&mut new_vector, ray1.as_ref().vector(), &a2, ray2.as_ref().vector(), &a1);
        let a1_shadow = M::coerce_from(&a1).expect("ray weights must be convertible");
        let a2_shadow = M::coerce_from(&a2).expect("ray weights must be convertible");
        let mut new_shadow = self.take_shadow_vector(ray1.shadow().len());
        linalg::lin_comb2_into(&mut new_shadow, ray1.shadow(), &a2_shadow, ray2.shadow(), &a1_shadow);
        if !self
            .normalizer
            .normalize_pair(&self.eps, &mut new_vector, &mut new_shadow)
        {
            return Err(zero_set);
        }

        Ok(self.build_ray_from_shadow::<ZR, R>(
            cone,
            new_vector,
            new_shadow,
            relaxed,
            Some(row),
            zero_set,
        ))
    }
}
