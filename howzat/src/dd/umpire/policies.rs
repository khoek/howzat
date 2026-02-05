use crate::dd::{ConeCtx, UmpireMatrix};
use crate::dd::ray::{RayData, RayNoNegatives, RayWithNegatives};
use calculo::num::{Epsilon, Num};
use hullabaloo::types::{Representation, Row, RowSet};

use super::LexOrderCache;

/// How the umpire maintains `ConeCtx.order_vector` and chooses the next halfspace row.
///
/// This is a *type-level* policy: the DD core never branches on an enum or a boolean flag.
pub trait HalfspacePolicy<N: Num>: Clone + std::fmt::Debug {
    /// Marker selecting which DD halfspace-add mode to use.
    ///
    /// See `crate::dd::mode::{Preordered, Dynamic}`.
    type Mode: crate::dd::mode::HalfspaceMode;

    /// Optional wrapper used to attach negative bookkeeping to the umpire's ray payload.
    type WrappedRayData<D: RayData>: RayData<ZeroSet = D::ZeroSet>
        + std::ops::Deref<Target = D>
        + std::ops::DerefMut<Target = D>;

    /// Wrap a newly-classified ray payload.
    fn wrap_ray_data<D: RayData>(
        &mut self,
        ray_data: D,
        negative_rows: RowSet,
    ) -> Self::WrappedRayData<D>;

    /// Recompute `cone.order_vector` and `cone.row_to_pos` under this policy.
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        strict_rows: &RowSet,
    );

    /// Choose the next halfspace row to add.
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        iteration: Row,
        active_rays: usize,
    ) -> Option<Row>;

    /// Reset any cached infeasible-count bookkeeping.
    #[inline(always)]
    fn reset_infeasible_counts(&mut self, _row_count: Row) {}

    /// Take a scratch `RowSet` to be filled with negative rows during classification.
    #[inline(always)]
    fn take_negative_rows(&mut self, _row_count: Row) -> RowSet {
        RowSet::new(0)
    }

    /// Recycle any bookkeeping stored in the wrapped ray payload.
    #[inline(always)]
    fn recycle_wrapped_ray_data<D: RayData>(&mut self, _ray_data: &mut Self::WrappedRayData<D>) {}

    /// Notify the policy that a ray has been inserted.
    #[inline(always)]
    fn on_ray_inserted<D: RayData>(&mut self, _ray_data: &Self::WrappedRayData<D>) {}

    /// Notify the policy that a ray is about to be removed.
    #[inline(always)]
    fn on_ray_removed<D: RayData>(&mut self, _ray_data: &Self::WrappedRayData<D>) {}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MaxIndex;

impl<N: Num> HalfspacePolicy<N> for MaxIndex {
    type Mode = crate::dd::mode::Preordered;
    type WrappedRayData<D: RayData> = RayNoNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(
        &mut self,
        ray_data: D,
        _negative_rows: RowSet,
    ) -> Self::WrappedRayData<D> {
        RayNoNegatives { inner: ray_data }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        _eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        _strict_rows: &RowSet,
    ) {
        let m = cone.matrix().row_count();
        if cone.order_vector.len() != m {
            cone.order_vector = vec![0; m];
        }
        for (i, dest) in (0..m).enumerate() {
            cone.order_vector[dest] = m - 1 - i;
        }
        cone.refresh_row_to_pos();
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        iteration: Row,
        _active_rays: usize,
    ) -> Option<Row> {
        let start = iteration.min(cone.matrix().row_count());
        for pos in start..cone.matrix().row_count() {
            let row = cone.order_vector[pos];
            if !excluded.contains(row) {
                return Some(row);
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MinIndex;

impl<N: Num> HalfspacePolicy<N> for MinIndex {
    type Mode = crate::dd::mode::Preordered;
    type WrappedRayData<D: RayData> = RayNoNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(
        &mut self,
        ray_data: D,
        _negative_rows: RowSet,
    ) -> Self::WrappedRayData<D> {
        RayNoNegatives { inner: ray_data }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        _eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        _strict_rows: &RowSet,
    ) {
        let m = cone.matrix().row_count();
        if cone.order_vector.len() != m {
            cone.order_vector = vec![0; m];
        }
        for i in 0..m {
            cone.order_vector[i] = i;
        }
        cone.refresh_row_to_pos();
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        iteration: Row,
        _active_rays: usize,
    ) -> Option<Row> {
        let start = iteration.min(cone.matrix().row_count());
        for pos in start..cone.matrix().row_count() {
            let row = cone.order_vector[pos];
            if !excluded.contains(row) {
                return Some(row);
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LexMin;

impl<N: Num> HalfspacePolicy<N> for LexMin {
    type Mode = crate::dd::mode::Preordered;
    type WrappedRayData<D: RayData> = RayNoNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(
        &mut self,
        ray_data: D,
        _negative_rows: RowSet,
    ) -> Self::WrappedRayData<D> {
        RayNoNegatives { inner: ray_data }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        strict_rows: &RowSet,
    ) {
        let m = cone.matrix().row_count();
        let d = cone.matrix().col_count();
        if cone.order_vector.len() != m {
            cone.order_vector = vec![0; m];
        }
        let matrix = cone.matrix();
        let strict_sig = strict_rows.hash_signature_u64();

        let cache_hit = cone.lex_order_cache.as_ref().is_some_and(|cache| {
            cache.row_count == m && cache.col_count == d && cache.strict_sig == strict_sig
        });
        if !cache_hit {
            let mut non_strict = Vec::new();
            let mut strict = Vec::new();
            for r in 0..m {
                if strict_rows.contains(r) {
                    strict.push(r);
                } else {
                    non_strict.push(r);
                }
            }
            non_strict.sort_by(|&ra, &rb| {
                let a = matrix.row(ra).expect("row index within bounds");
                let b = matrix.row(rb).expect("row index within bounds");
                crate::matrix::lex_cmp(a, b, eps)
            });
            strict.sort_by(|&ra, &rb| {
                let a = matrix.row(ra).expect("row index within bounds");
                let b = matrix.row(rb).expect("row index within bounds");
                crate::matrix::lex_cmp(a, b, eps)
            });
            cone.lex_order_cache = Some(LexOrderCache {
                row_count: m,
                col_count: d,
                strict_sig,
                non_strict,
                strict,
            });
        }
        let cache = cone
            .lex_order_cache
            .as_ref()
            .expect("LexMin cache populated");
        for (idx, row) in cache
            .non_strict
            .iter()
            .copied()
            .chain(cache.strict.iter().copied())
            .enumerate()
        {
            cone.order_vector[idx] = row;
        }
        cone.refresh_row_to_pos();
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        iteration: Row,
        _active_rays: usize,
    ) -> Option<Row> {
        let start = iteration.min(cone.matrix().row_count());
        for pos in start..cone.matrix().row_count() {
            let row = cone.order_vector[pos];
            if !excluded.contains(row) {
                return Some(row);
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LexMax;

impl<N: Num> HalfspacePolicy<N> for LexMax {
    type Mode = crate::dd::mode::Preordered;
    type WrappedRayData<D: RayData> = RayNoNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(
        &mut self,
        ray_data: D,
        _negative_rows: RowSet,
    ) -> Self::WrappedRayData<D> {
        RayNoNegatives { inner: ray_data }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        strict_rows: &RowSet,
    ) {
        let m = cone.matrix().row_count();
        let d = cone.matrix().col_count();
        if cone.order_vector.len() != m {
            cone.order_vector = vec![0; m];
        }
        let matrix = cone.matrix();
        let strict_sig = strict_rows.hash_signature_u64();

        let cache_hit = cone.lex_order_cache.as_ref().is_some_and(|cache| {
            cache.row_count == m && cache.col_count == d && cache.strict_sig == strict_sig
        });
        if !cache_hit {
            let mut non_strict = Vec::new();
            let mut strict = Vec::new();
            for r in 0..m {
                if strict_rows.contains(r) {
                    strict.push(r);
                } else {
                    non_strict.push(r);
                }
            }
            non_strict.sort_by(|&ra, &rb| {
                let a = matrix.row(ra).expect("row index within bounds");
                let b = matrix.row(rb).expect("row index within bounds");
                crate::matrix::lex_cmp(a, b, eps)
            });
            strict.sort_by(|&ra, &rb| {
                let a = matrix.row(ra).expect("row index within bounds");
                let b = matrix.row(rb).expect("row index within bounds");
                crate::matrix::lex_cmp(a, b, eps)
            });
            cone.lex_order_cache = Some(LexOrderCache {
                row_count: m,
                col_count: d,
                strict_sig,
                non_strict,
                strict,
            });
        }
        let cache = cone
            .lex_order_cache
            .as_ref()
            .expect("LexMax cache populated");
        for (idx, row) in cache
            .non_strict
            .iter()
            .rev()
            .copied()
            .chain(cache.strict.iter().rev().copied())
            .enumerate()
        {
            cone.order_vector[idx] = row;
        }
        cone.refresh_row_to_pos();
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        iteration: Row,
        _active_rays: usize,
    ) -> Option<Row> {
        // Same preordered scheduling as `LexMin`, but with a different `order_vector`.
        let start = iteration.min(cone.matrix().row_count());
        for pos in start..cone.matrix().row_count() {
            let row = cone.order_vector[pos];
            if !excluded.contains(row) {
                return Some(row);
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RandomRow {
    seed: u64,
}

impl RandomRow {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl Default for RandomRow {
    fn default() -> Self {
        Self { seed: 0xDEADBEEF }
    }
}

impl<N: Num> HalfspacePolicy<N> for RandomRow {
    type Mode = crate::dd::mode::Preordered;
    type WrappedRayData<D: RayData> = RayNoNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(
        &mut self,
        ray_data: D,
        _negative_rows: RowSet,
    ) -> Self::WrappedRayData<D> {
        RayNoNegatives { inner: ray_data }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        _eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        _strict_rows: &RowSet,
    ) {
        // Match the previous `RowOrder::RandomRow` permutation (no strict-row partitioning).
        let m = cone.matrix().row_count();
        if cone.order_vector.len() != m {
            cone.order_vector = vec![0; m];
        }
        let mut row_order: Vec<Row> = (0..m).collect();

        fn splitmix64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        let mut state = self.seed;
        let rand_max = u64::MAX as f64;
        for j in (1..row_order.len()).rev() {
            let r = splitmix64(&mut state) as f64;
            let u = r / rand_max; // in (0,1]
            let mut k = ((j as f64 + 1.0) * u).floor() as usize;
            if k > j {
                k = j;
            }
            row_order.swap(j, k);
        }

        for (idx, row) in row_order.into_iter().enumerate() {
            cone.order_vector[idx] = row;
        }
        cone.refresh_row_to_pos();
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        iteration: Row,
        _active_rays: usize,
    ) -> Option<Row> {
        let start = iteration.min(cone.matrix().row_count());
        for pos in start..cone.matrix().row_count() {
            let row = cone.order_vector[pos];
            if !excluded.contains(row) {
                return Some(row);
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
struct CutoffBookkeeping {
    counts: Vec<usize>,
    negative_pool: Vec<RowSet>,
}

impl Default for CutoffBookkeeping {
    fn default() -> Self {
        Self {
            counts: Vec::new(),
            negative_pool: Vec::new(),
        }
    }
}

impl CutoffBookkeeping {
    #[inline(always)]
    fn ensure_len(&mut self, row_count: Row) {
        if self.counts.len() != row_count {
            self.counts.resize(row_count, 0);
        }
    }

    #[inline]
    fn take_negative_rows(&mut self, row_count: Row) -> RowSet {
        self.ensure_len(row_count);
        let mut set = self
            .negative_pool
            .pop()
            .unwrap_or_else(|| RowSet::new(row_count));
        if set.len() != row_count {
            set.resize(row_count);
        }
        set.clear();
        set
    }

    #[inline]
    fn recycle_negative_rows(&mut self, mut set: RowSet) {
        set.clear();
        self.negative_pool.push(set);
    }

    #[inline]
    fn on_ray_inserted(&mut self, negative_rows: &RowSet) {
        for row_id in negative_rows.iter() {
            let row = row_id.as_index();
            self.counts[row] += 1;
        }
    }

    #[inline]
    fn on_ray_removed(&mut self, negative_rows: &RowSet) {
        for row_id in negative_rows.iter() {
            let row = row_id.as_index();
            let count = &mut self.counts[row];
            assert!(*count > 0, "row infeasible count underflow");
            *count -= 1;
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MinCutoff {
    cutoff: CutoffBookkeeping,
}

impl<N: Num> HalfspacePolicy<N> for MinCutoff {
    type Mode = crate::dd::mode::Dynamic;
    type WrappedRayData<D: RayData> = RayWithNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(&mut self, ray_data: D, negative_rows: RowSet) -> Self::WrappedRayData<D> {
        RayWithNegatives {
            inner: ray_data,
            negative_rows,
        }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        strict_rows: &RowSet,
    ) {
        // Matches the previous cutoff ordering: lex-min row scan order.
        LexMin.recompute_row_order_vector(eps, cone, strict_rows);
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        _cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        _iteration: Row,
        _active_rays: usize,
    ) -> Option<Row> {
        let mut chosen = None;
        let mut inf_min = usize::MAX;
        for idx in excluded.iter().complement() {
            let i = idx.as_index();
            let inf = self.cutoff.counts.get(i).copied().unwrap_or(0);
            if inf < inf_min {
                inf_min = inf;
                chosen = Some(i);
            }
        }
        chosen
    }

    #[inline(always)]
    fn reset_infeasible_counts(&mut self, row_count: Row) {
        self.cutoff.ensure_len(row_count);
        self.cutoff.counts.fill(0);
    }

    #[inline(always)]
    fn take_negative_rows(&mut self, row_count: Row) -> RowSet {
        self.cutoff.take_negative_rows(row_count)
    }

    #[inline(always)]
    fn recycle_wrapped_ray_data<D: RayData>(&mut self, ray_data: &mut Self::WrappedRayData<D>) {
        let mut set = RowSet::new(0);
        std::mem::swap(&mut set, &mut ray_data.negative_rows);
        self.cutoff.recycle_negative_rows(set);
    }

    #[inline(always)]
    fn on_ray_inserted<D: RayData>(&mut self, ray_data: &Self::WrappedRayData<D>) {
        self.cutoff.on_ray_inserted(ray_data.negative_rows());
    }

    #[inline(always)]
    fn on_ray_removed<D: RayData>(&mut self, ray_data: &Self::WrappedRayData<D>) {
        self.cutoff.on_ray_removed(ray_data.negative_rows());
    }
}

#[derive(Clone, Debug)]
pub struct MaxCutoff {
    cutoff: CutoffBookkeeping,
}

impl Default for MaxCutoff {
    fn default() -> Self {
        Self {
            cutoff: CutoffBookkeeping::default(),
        }
    }
}

impl<N: Num> HalfspacePolicy<N> for MaxCutoff {
    type Mode = crate::dd::mode::Dynamic;
    type WrappedRayData<D: RayData> = RayWithNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(&mut self, ray_data: D, negative_rows: RowSet) -> Self::WrappedRayData<D> {
        RayWithNegatives {
            inner: ray_data,
            negative_rows,
        }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        strict_rows: &RowSet,
    ) {
        LexMin.recompute_row_order_vector(eps, cone, strict_rows);
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        _cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        _iteration: Row,
        _active_rays: usize,
    ) -> Option<Row> {
        let mut chosen = None;
        let mut inf_max: isize = -1;
        for idx in excluded.iter().complement() {
            let i = idx.as_index();
            let inf = self.cutoff.counts.get(i).copied().unwrap_or(0) as isize;
            if inf > inf_max {
                inf_max = inf;
                chosen = Some(i);
            }
        }
        chosen
    }

    #[inline(always)]
    fn reset_infeasible_counts(&mut self, row_count: Row) {
        self.cutoff.ensure_len(row_count);
        self.cutoff.counts.fill(0);
    }

    #[inline(always)]
    fn take_negative_rows(&mut self, row_count: Row) -> RowSet {
        self.cutoff.take_negative_rows(row_count)
    }

    #[inline(always)]
    fn recycle_wrapped_ray_data<D: RayData>(&mut self, ray_data: &mut Self::WrappedRayData<D>) {
        let mut set = RowSet::new(0);
        std::mem::swap(&mut set, &mut ray_data.negative_rows);
        self.cutoff.recycle_negative_rows(set);
    }

    #[inline(always)]
    fn on_ray_inserted<D: RayData>(&mut self, ray_data: &Self::WrappedRayData<D>) {
        self.cutoff.on_ray_inserted(ray_data.negative_rows());
    }

    #[inline(always)]
    fn on_ray_removed<D: RayData>(&mut self, ray_data: &Self::WrappedRayData<D>) {
        self.cutoff.on_ray_removed(ray_data.negative_rows());
    }
}

#[derive(Clone, Debug)]
pub struct MixCutoff {
    cutoff: CutoffBookkeeping,
}

impl Default for MixCutoff {
    fn default() -> Self {
        Self {
            cutoff: CutoffBookkeeping::default(),
        }
    }
}

impl<N: Num> HalfspacePolicy<N> for MixCutoff {
    type Mode = crate::dd::mode::Dynamic;
    type WrappedRayData<D: RayData> = RayWithNegatives<D>;

    #[inline(always)]
    fn wrap_ray_data<D: RayData>(&mut self, ray_data: D, negative_rows: RowSet) -> Self::WrappedRayData<D> {
        RayWithNegatives {
            inner: ray_data,
            negative_rows,
        }
    }

    #[inline(always)]
    fn recompute_row_order_vector<R: Representation, M: UmpireMatrix<N, R>, E: Epsilon<N>>(
        &mut self,
        eps: &E,
        cone: &mut ConeCtx<N, R, M>,
        strict_rows: &RowSet,
    ) {
        LexMin.recompute_row_order_vector(eps, cone, strict_rows);
    }

    #[inline(always)]
    fn choose_next_halfspace<R: Representation, M: UmpireMatrix<N, R>>(
        &mut self,
        _cone: &ConeCtx<N, R, M>,
        excluded: &RowSet,
        _iteration: Row,
        active_rays: usize,
    ) -> Option<Row> {
        let mut chosen = None;
        let mut cut_max: isize = -1;
        for idx in excluded.iter().complement() {
            let i = idx.as_index();
            let inf = self.cutoff.counts.get(i).copied().unwrap_or(0);
            let fea = active_rays.saturating_sub(inf);
            let score = if fea <= inf { inf } else { fea } as isize;
            if score > cut_max {
                cut_max = score;
                chosen = Some(i);
            }
        }
        chosen
    }

    #[inline(always)]
    fn reset_infeasible_counts(&mut self, row_count: Row) {
        self.cutoff.ensure_len(row_count);
        self.cutoff.counts.fill(0);
    }

    #[inline(always)]
    fn take_negative_rows(&mut self, row_count: Row) -> RowSet {
        self.cutoff.take_negative_rows(row_count)
    }

    #[inline(always)]
    fn recycle_wrapped_ray_data<D: RayData>(&mut self, ray_data: &mut Self::WrappedRayData<D>) {
        let mut set = RowSet::new(0);
        std::mem::swap(&mut set, &mut ray_data.negative_rows);
        self.cutoff.recycle_negative_rows(set);
    }

    #[inline(always)]
    fn on_ray_inserted<D: RayData>(&mut self, ray_data: &Self::WrappedRayData<D>) {
        self.cutoff.on_ray_inserted(ray_data.negative_rows());
    }

    #[inline(always)]
    fn on_ray_removed<D: RayData>(&mut self, ray_data: &Self::WrappedRayData<D>) {
        self.cutoff.on_ray_removed(ray_data.negative_rows());
    }
}
