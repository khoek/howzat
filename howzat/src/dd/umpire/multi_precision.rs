use crate::HowzatError as Error;
use crate::dd::DefaultNormalizer;
use crate::dd::zero::{ZeroRepr, ZeroSet};
use crate::dd::{Ray, RayClass, RayId};
use crate::matrix::LpMatrix;
use calculo::linalg;
use calculo::num::{CoerceFrom, Epsilon, Normalizer, Num, Sign};
use hullabaloo::types::{Representation, Row, RowSet};

use super::policies::{HalfspacePolicy, LexMin};
use super::{ConeCtx, Umpire, UmpireMatrix};
use std::cmp::Ordering;

/// Matrix wrapper that maintains an exact/higher-precision shadow for sign decisions.
#[derive(Debug)]
pub struct ShadowedMatrix<N: Num, M: Num, R: Representation> {
    pub(crate) base: LpMatrix<N, R>,
    pub(crate) shadow: Vec<M>,
}

impl<N: Num, M: Num, R: Representation> Clone for ShadowedMatrix<N, M, R> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            shadow: self.shadow.clone(),
        }
    }
}

impl<N: Num, M: Num, R: Representation> ShadowedMatrix<N, M, R> {
    #[inline(always)]
    fn cols(&self) -> usize {
        self.base.col_count()
    }

    #[inline(always)]
    fn shadow_row(&self, row: Row) -> &[M] {
        let cols = self.cols();
        let start = row * cols;
        &self.shadow[start..start + cols]
    }

    #[inline(always)]
    pub(crate) fn shadow_row_value(&self, row: Row, shadow_vec: &[M]) -> M {
        linalg::dot(self.shadow_row(row), shadow_vec)
    }
}

impl<N: Num, M: Num, R: Representation> UmpireMatrix<N, R> for ShadowedMatrix<N, M, R> {
    #[inline(always)]
    fn base(&self) -> &LpMatrix<N, R> {
        &self.base
    }

    fn select_columns(&self, columns: &[usize]) -> Result<Self, Error> {
        let base = self.base.select_columns(columns)?;
        let old_cols = self.base.col_count();
        let new_cols = columns.len();
        let rows = self.base.row_count();
        let mut shadow = Vec::with_capacity(rows * new_cols);
        for row in 0..rows {
            let start = row * old_cols;
            for &col in columns {
                shadow.push(self.shadow[start + col].clone());
            }
        }
        Ok(Self { base, shadow })
    }
}

#[derive(Clone, Debug)]
pub struct CachedRay<N: Num, M: Num, ZS: ZeroSet = crate::dd::SatSet> {
    pub(crate) standard: Ray<N, ZS>,
    pub(crate) shadow: Vec<M>,
    pub(crate) shadow_last_eval_row: Option<Row>,
    pub(crate) shadow_last_eval: M,
    pub(crate) shadow_last_sign: Sign,
    pub(crate) near_zero_rows: Vec<Row>,
    pub(crate) near_zero_truncated: bool,
}

impl<N: Num, M: Num, ZS: ZeroSet> CachedRay<N, M, ZS> {
    #[inline(always)]
    pub(crate) fn shadow(&self) -> &[M] {
        &self.shadow
    }

    #[inline(always)]
    pub(crate) fn cached_shadow_sign(&self, row: Row) -> Option<Sign> {
        (self.shadow_last_eval_row == Some(row)).then_some(self.shadow_last_sign)
    }

    #[inline(always)]
    pub(crate) fn near_zero_rows(&self) -> &[Row] {
        &self.near_zero_rows
    }

    #[inline(always)]
    pub(crate) fn near_zero_truncated(&self) -> bool {
        self.near_zero_truncated
    }
}

impl<N: Num, M: Num, ZS: ZeroSet> AsRef<Ray<N, ZS>> for CachedRay<N, M, ZS> {
    #[inline(always)]
    fn as_ref(&self) -> &Ray<N, ZS> {
        &self.standard
    }
}

impl<N: Num, M: Num, ZS: ZeroSet> AsMut<Ray<N, ZS>> for CachedRay<N, M, ZS> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Ray<N, ZS> {
        &mut self.standard
    }
}

impl<N: Num, M: Num, ZS: ZeroSet> crate::dd::ray::RayData for CachedRay<N, M, ZS> {
    type ZeroSet = ZS;

    #[inline(always)]
    fn zero_set(&self) -> &ZS {
        &self.standard.class.zero_set
    }

    #[inline(always)]
    fn zero_set_mut(&mut self) -> &mut ZS {
        &mut self.standard.class.zero_set
    }

    #[inline(always)]
    fn zero_set_signature(&self) -> u64 {
        self.standard.class.zero_set_sig
    }

    #[inline(always)]
    fn set_zero_set_signature(&mut self, sig: u64) {
        self.standard.class.zero_set_sig = sig;
    }

    #[inline(always)]
    fn zero_set_count(&self) -> usize {
        self.standard.class.zero_set_count
    }

    #[inline(always)]
    fn set_zero_set_count(&mut self, count: usize) {
        self.standard.class.zero_set_count = count;
    }

    #[inline(always)]
    fn first_infeasible_row(&self) -> Option<Row> {
        self.standard.first_infeasible_row()
    }

    #[inline(always)]
    fn set_first_infeasible_row(&mut self, row: Option<Row>) {
        self.standard.class.first_infeasible_row = row;
    }

    #[inline(always)]
    fn is_feasible(&self) -> bool {
        self.standard.is_feasible()
    }

    #[inline(always)]
    fn is_weakly_feasible(&self) -> bool {
        self.standard.is_weakly_feasible()
    }

    #[inline(always)]
    fn last_sign(&self) -> Sign {
        self.standard.last_sign()
    }
}

impl<N: Num, M: Num + CoerceFrom<N>, ZS: ZeroSet> CachedRay<N, M, ZS> {
    pub(crate) fn ensure_shadow_matches_standard(&mut self) {
        if self.shadow.len() == self.standard.vector.len() {
            return;
        }
        self.shadow.resize(self.standard.vector.len(), M::zero());
        for (dst, v) in self.shadow.iter_mut().zip(self.standard.vector.iter()) {
            *dst = M::coerce_from(v).expect("ray vectors must be convertible");
        }
        self.shadow_last_eval_row = None;
        self.shadow_last_eval = M::zero();
        self.shadow_last_sign = Sign::Zero;
        self.near_zero_rows.clear();
        self.near_zero_truncated = false;
    }
}

/// Heavyweight umpire that computes signs in a higher-precision type `M`.
///
/// The DD core still stores rays/vectors in `N` (for output), but all sign decisions are made
/// on shadow values in `M` using `shadow_eps: Epsilon<M>`.
#[derive(Clone, Debug)]
pub struct MultiPrecisionUmpire<
    N: Num,
    M: Num,
    E: Epsilon<N> = calculo::num::DynamicEpsilon<N>,
    EM: Epsilon<M> = calculo::num::DynamicEpsilon<M>,
    NM: Normalizer<N> = <N as DefaultNormalizer>::Norm,
    H: HalfspacePolicy<N> = LexMin,
> {
    eps: E,
    shadow_eps: EM,
    normalizer: NM,
    halfspace: H,
    standard_vector_pool: Vec<Vec<N>>,
    shadow_vector_pool: Vec<Vec<M>>,
}

impl<N: DefaultNormalizer, M: Num + CoerceFrom<N>, E: Epsilon<N>, EM: Epsilon<M>>
    MultiPrecisionUmpire<N, M, E, EM, <N as DefaultNormalizer>::Norm, LexMin>
{
    pub fn new(eps: E, shadow_eps: EM) -> Self {
        Self {
            eps,
            shadow_eps,
            normalizer: <N as DefaultNormalizer>::Norm::default(),
            halfspace: LexMin,
            standard_vector_pool: Vec::new(),
            shadow_vector_pool: Vec::new(),
        }
    }
}

impl<N: Num, M: Num + CoerceFrom<N>, E: Epsilon<N>, EM: Epsilon<M>, NM: Normalizer<N>>
    MultiPrecisionUmpire<N, M, E, EM, NM, LexMin>
{
    pub fn with_normalizer(eps: E, shadow_eps: EM, normalizer: NM) -> Self {
        Self {
            eps,
            shadow_eps,
            normalizer,
            halfspace: LexMin,
            standard_vector_pool: Vec::new(),
            shadow_vector_pool: Vec::new(),
        }
    }
}

impl<N: DefaultNormalizer, M: Num + CoerceFrom<N>, E: Epsilon<N>, EM: Epsilon<M>, H: HalfspacePolicy<N>>
    MultiPrecisionUmpire<N, M, E, EM, <N as DefaultNormalizer>::Norm, H>
{
    pub fn with_halfspace_policy(eps: E, shadow_eps: EM, halfspace: H) -> Self {
        Self {
            eps,
            shadow_eps,
            normalizer: <N as DefaultNormalizer>::Norm::default(),
            halfspace,
            standard_vector_pool: Vec::new(),
            shadow_vector_pool: Vec::new(),
        }
    }
}

impl<
    N: Num,
    M: Num + CoerceFrom<N>,
    E: Epsilon<N>,
    EM: Epsilon<M>,
    NM: Normalizer<N>,
    H: HalfspacePolicy<N>,
> MultiPrecisionUmpire<N, M, E, EM, NM, H> {
    #[inline(always)]
    fn shadow_sign(&self, value: &M) -> Sign {
        self.shadow_eps.sign(value)
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

        let mut cached_last_eval: Option<(M, Sign)> = None;

        for &row_idx in &cone.order_vector {
            let value = cone.matrix.shadow_row_value(row_idx, &shadow);
            let sign = self.shadow_sign(&value);
            if track_negatives && sign == Sign::Negative {
                negative_rows.insert(row_idx);
            }

            if Some(row_idx) == last_row {
                cached_last_eval = Some((value.clone(), sign));
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

        let (shadow_last_eval, shadow_last_sign) = match last_row {
            Some(row_idx) => cached_last_eval.unwrap_or_else(|| {
                let value = cone.matrix.shadow_row_value(row_idx, &shadow);
                let sign = self.shadow_sign(&value);
                (value, sign)
            }),
            None => (M::zero(), Sign::Zero),
        };
        let standard_last_eval = match last_row {
            Some(row_idx) => cone.row_value(row_idx, &standard_vector),
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
                last_sign: shadow_last_sign,
            },
        };

        let ray_data = CachedRay {
            standard,
            shadow,
            shadow_last_eval_row: last_row,
            shadow_last_eval,
            shadow_last_sign,
            near_zero_rows: Vec::new(),
            near_zero_truncated: false,
        };

        self.halfspace.wrap_ray_data(ray_data, negative_rows)
    }
}

impl<
    N: Num,
    M: Num,
    E: Epsilon<N>,
    EM: Epsilon<M>,
    NM: Normalizer<N>,
    H: HalfspacePolicy<N>,
> MultiPrecisionUmpire<N, M, E, EM, NM, H> {
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
}

impl<
    N: Num,
    M: Num,
    E: Epsilon<N>,
    EM: Epsilon<M>,
    NM: Normalizer<N>,
    H: HalfspacePolicy<N>,
    ZR: crate::dd::mode::PreorderedBackend,
> Umpire<N, ZR> for MultiPrecisionUmpire<N, M, E, EM, NM, H>
where
    M: CoerceFrom<N>,
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
        ShadowedMatrix {
            base: matrix,
            shadow,
        }
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

    fn basis_column_vector(
        &mut self,
        basis: &hullabaloo::matrix::BasisMatrix<N>,
        col: usize,
    ) -> Vec<Self::Scalar> {
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

    fn rays_equivalent(&mut self, a: &Self::RayData, b: &Self::RayData) -> bool {
        let eps = &self.shadow_eps;
        let va = a.shadow();
        let vb = b.shadow();
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

        ray_data.standard.class.zero_set_sig = 0;
        ray_data.standard.class.zero_set_count = 0;
        ray_data.standard.class.first_infeasible_row = None;
        ray_data.standard.class.last_eval_row = None;
        ray_data.standard.class.last_eval = N::zero();

        ray_data.shadow_last_eval_row = None;
        ray_data.shadow_last_eval = M::zero();
        ray_data.shadow_last_sign = Sign::Zero;
        ray_data.near_zero_rows.clear();
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

    fn on_ray_removed<R: Representation>(
        &mut self,
        _cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        _relaxed: bool,
    ) {
        self.halfspace.on_ray_removed(ray_data);
    }

    fn on_ray_inserted<R: Representation>(
        &mut self,
        _cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &Self::RayData,
        _relaxed: bool,
    ) {
        self.halfspace.on_ray_inserted(ray_data);
    }

    #[inline(always)]
    fn sign_for_row_on_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray: &Self::RayData,
        row: Row,
    ) -> Sign {
        if ray.shadow_last_eval_row == Some(row) {
            return ray.shadow_last_sign;
        }
        self.shadow_sign(&cone.matrix.shadow_row_value(row, ray.shadow()))
    }

    fn classify_ray<R: Representation>(
        &mut self,
        cone: &ConeCtx<N, R, Self::MatrixData<R>>,
        ray_data: &mut Self::RayData,
        row: Row,
    ) -> Sign {
        ray_data.ensure_shadow_matches_standard();

        if ray_data.shadow_last_eval_row == Some(row) {
            return ray_data.shadow_last_sign;
        }
        let value = cone.matrix.shadow_row_value(row, &ray_data.shadow);
        let sign = self.shadow_sign(&value);

        ray_data.shadow_last_eval_row = Some(row);
        ray_data.shadow_last_eval = value.clone();
        ray_data.shadow_last_sign = sign;

        ray_data.standard.class.last_eval_row = Some(row);
        ray_data.standard.class.last_eval = cone.row_value(row, &ray_data.standard.vector);
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
            .standard
            .class
            .first_infeasible_row
            .filter(|_| force_infeasible && !ray_data.standard.class.feasible);
        let floor_pos = force_negative_row
            .and_then(|r| cone.row_to_pos.get(r).copied())
            .filter(|pos| *pos < cone.order_vector.len());

        for (pos, &row_idx) in cone.order_vector.iter().enumerate() {
            let forced = floor_pos.is_some_and(|floor| pos >= floor);
            if forced {
                negative_out.insert(row_idx);
                continue;
            }
            let value = cone.matrix.shadow_row_value(row_idx, ray_data.shadow());
            let sign = self.shadow_sign(&value);
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
        for &row_idx in &cone.order_vector {
            let value = cone.matrix.shadow_row_value(row_idx, &ray_data.shadow);
            let sign = self.shadow_sign(&value);
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
        ray_data.ensure_shadow_matches_standard();

        let last_eval_row = ray_data.standard.class.last_eval_row;
        let mut last_eval = ray_data.standard.class.last_eval.clone();
        let mut last_sign = ray_data.standard.class.last_sign;

        ray_data.standard.class.zero_set.clear();
        let mut zero_set_count = 0usize;
        ray_data.standard.class.first_infeasible_row = None;
        ray_data.standard.class.feasible = true;
        ray_data.standard.class.weakly_feasible = true;

        for &row_idx in &cone.order_vector {
            let value = cone.matrix.shadow_row_value(row_idx, &ray_data.shadow);
            let sign = self.shadow_sign(&value);
            if Some(row_idx) == last_eval_row {
                last_eval = cone.row_value(row_idx, &ray_data.standard.vector);
                last_sign = sign;
                ray_data.shadow_last_eval_row = Some(row_idx);
                ray_data.shadow_last_eval = value.clone();
                ray_data.shadow_last_sign = sign;
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

        let val1 = if ray1.standard.class.last_eval_row == Some(row) {
            ray1.standard.class.last_eval.clone()
        } else {
            cone.row_value(row, ray1.standard.vector())
        };
        let val2 = if ray2.standard.class.last_eval_row == Some(row) {
            ray2.standard.class.last_eval.clone()
        } else {
            cone.row_value(row, ray2.standard.vector())
        };

        let a1 = val1.abs();
        let a2 = val2.abs();

        let mut new_vector = self.take_standard_vector(ray1.standard.vector().len());
        linalg::lin_comb2_into(
            &mut new_vector,
            ray1.standard.vector(),
            &a2,
            ray2.standard.vector(),
            &a1,
        );
        let a1_shadow = M::coerce_from(&a1).expect("ray weights must be convertible");
        let a2_shadow = M::coerce_from(&a2).expect("ray weights must be convertible");
        let mut new_shadow = self.take_shadow_vector(ray1.shadow().len());
        linalg::lin_comb2_into(
            &mut new_shadow,
            ray1.shadow(),
            &a2_shadow,
            ray2.shadow(),
            &a1_shadow,
        );
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

#[cfg(all(test, feature = "rug"))]
mod tests {
    use super::MultiPrecisionUmpire;
    use crate::dd::{ConeCtx, SatSet, Umpire};
    use crate::matrix::LpMatrix;
    use calculo::num::{DynamicEpsilon, Num, RugRat, Sign};
    use hullabaloo::types::{Inequality, InequalityKind};

    #[test]
    fn multiprecision_classify_ray_uses_shadow_not_standard_vector() {
        let eps = f64::default_eps();
        let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0]]);
        let shadow_eps = DynamicEpsilon::<RugRat>::new(
            RugRat::try_from_f64(1.0e-12).expect("default eps is finite"),
        );
        let mut umpire = MultiPrecisionUmpire::<f64, RugRat, _, _>::new(eps, shadow_eps);
        let matrix = <_ as Umpire<f64>>::ingest(&mut umpire, matrix);
        let cone = ConeCtx {
            matrix,
            equality_kinds: vec![InequalityKind::Inequality],
            order_vector: vec![0],
            row_to_pos: vec![0],
            lex_order_cache: None,
            sat_row_to_id: vec![Some(0usize.into())],
            sat_id_to_row: vec![0],
            _phantom: std::marker::PhantomData,
        };

        let mut ray_data = <_ as Umpire<f64>>::classify_vector::<Inequality>(
            &mut umpire,
            &cone,
            vec![0.0, 1.0],
            false,
            Some(0),
            SatSet::default(),
        );

        // Introduce drift in the standard model only; keep shadow intact.
        ray_data.as_mut().vector[0] = 1e-3;
        ray_data.as_mut().class.last_eval_row = None;

        let sign = <_ as Umpire<f64>>::classify_ray::<Inequality>(&mut umpire, &cone, &mut ray_data, 0);
        assert_eq!(sign, Sign::Zero);
    }
}
