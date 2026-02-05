use calculo::num::{Num, Sign};
use hullabaloo::types::{Row, RowSet};

use crate::dd::sat::SatSet;
use crate::dd::zero::ZeroSet;

#[derive(Clone, Debug)]
pub(crate) struct Slot<T> {
    pub(crate) value: T,
    pub(crate) active: bool,
}

pub(crate) trait ResetPolicy<T> {
    fn reset(&mut self, value: &mut T);
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NoopReset;

impl<T> ResetPolicy<T> for NoopReset {
    fn reset(&mut self, _value: &mut T) {}
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ArenaIndex(usize);

impl ArenaIndex {
    pub(crate) fn as_index(self) -> usize {
        self.0
    }
}

impl From<usize> for ArenaIndex {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<ArenaIndex> for usize {
    fn from(value: ArenaIndex) -> Self {
        value.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Arena<T, P: ResetPolicy<T>> {
    pub(crate) slots: Vec<Slot<T>>,
    pub(crate) free: Vec<ArenaIndex>,
    pub(crate) reset: P,
}

impl<T, P> Arena<T, P>
where
    P: ResetPolicy<T>,
{
    pub(crate) fn new(reset: P) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            reset,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.slots.clear();
        self.free.clear();
    }

    pub(crate) fn insert(&mut self, value: T, active: bool) -> ArenaIndex {
        if let Some(idx) = self.free.pop() {
            let slot = self
                .slots
                .get_mut(idx.as_index())
                .expect("free list index out of bounds");
            slot.value = value;
            slot.active = active;
            return idx;
        }
        let idx = ArenaIndex(self.slots.len());
        self.slots.push(Slot { value, active });
        idx
    }

    pub(crate) fn deactivate(&mut self, id: impl Into<ArenaIndex>) {
        let id = id.into();
        let slot = self
            .slots
            .get_mut(id.as_index())
            .expect("Arena::deactivate: id out of bounds");
        if slot.active {
            slot.active = false;
            self.reset.reset(&mut slot.value);
            self.free.push(id);
        }
    }

    pub(crate) fn deactivate_all(&mut self) {
        self.free.clear();
        self.slots.iter_mut().enumerate().for_each(|(idx, slot)| {
            slot.active = false;
            self.reset.reset(&mut slot.value);
            self.free.push(ArenaIndex(idx));
        });
    }

    pub(crate) fn get(&self, id: impl Into<ArenaIndex>) -> Option<&T> {
        let id = id.into().as_index();
        self.slots
            .get(id)
            .and_then(|slot| slot.active.then_some(&slot.value))
    }

    pub(crate) fn get_mut(&mut self, id: impl Into<ArenaIndex>) -> Option<&mut T> {
        let id = id.into().as_index();
        self.slots
            .get_mut(id)
            .and_then(|slot| slot.active.then_some(&mut slot.value))
    }
}

impl<T, P> Default for Arena<T, P>
where
    P: ResetPolicy<T> + Default,
{
    fn default() -> Self {
        Self::new(P::default())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RayId(pub(crate) usize);

impl RayId {
    pub(crate) fn as_index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RayKey {
    pub(crate) slot: usize,
    pub(crate) generation: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RayOrigin {
    pub(crate) self_key: RayKey,
    pub(crate) parent_a: Option<RayKey>,
    pub(crate) parent_b: Option<RayKey>,
    pub(crate) creation_row: Option<Row>,
}

impl Default for RayOrigin {
    fn default() -> Self {
        Self {
            self_key: RayKey {
                slot: 0,
                generation: 0,
            },
            parent_a: None,
            parent_b: None,
            creation_row: None,
        }
    }
}

impl From<usize> for RayId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<RayId> for usize {
    fn from(value: RayId) -> Self {
        value.0
    }
}

#[derive(Clone, Debug)]
pub struct Ray<N: Num, ZS: ZeroSet = SatSet> {
    pub(crate) vector: Vec<N>,
    pub(crate) class: RayClass<N, ZS>,
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct RayNoNegatives<D: RayData> {
    pub(crate) inner: D,
}

impl<D: RayData> std::ops::Deref for RayNoNegatives<D> {
    type Target = D;

    #[inline(always)]
    fn deref(&self) -> &D {
        &self.inner
    }
}

impl<D: RayData> std::ops::DerefMut for RayNoNegatives<D> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut D {
        &mut self.inner
    }
}

impl<D: RayData> RayData for RayNoNegatives<D> {
    type ZeroSet = D::ZeroSet;

    #[inline(always)]
    fn zero_set(&self) -> &Self::ZeroSet {
        self.inner.zero_set()
    }

    #[inline(always)]
    fn zero_set_mut(&mut self) -> &mut Self::ZeroSet {
        self.inner.zero_set_mut()
    }

    #[inline(always)]
    fn zero_set_signature(&self) -> u64 {
        self.inner.zero_set_signature()
    }

    #[inline(always)]
    fn set_zero_set_signature(&mut self, sig: u64) {
        self.inner.set_zero_set_signature(sig);
    }

    #[inline(always)]
    fn zero_set_count(&self) -> usize {
        self.inner.zero_set_count()
    }

    #[inline(always)]
    fn set_zero_set_count(&mut self, count: usize) {
        self.inner.set_zero_set_count(count);
    }

    #[inline(always)]
    fn first_infeasible_row(&self) -> Option<Row> {
        self.inner.first_infeasible_row()
    }

    #[inline(always)]
    fn set_first_infeasible_row(&mut self, row: Option<Row>) {
        self.inner.set_first_infeasible_row(row);
    }

    #[inline(always)]
    fn is_feasible(&self) -> bool {
        self.inner.is_feasible()
    }

    #[inline(always)]
    fn is_weakly_feasible(&self) -> bool {
        self.inner.is_weakly_feasible()
    }

    #[inline(always)]
    fn last_sign(&self) -> Sign {
        self.inner.last_sign()
    }
}

#[derive(Clone, Debug)]
pub struct RayWithNegatives<D: RayData> {
    pub(crate) inner: D,
    pub(crate) negative_rows: RowSet,
}

impl<D: RayData> std::ops::Deref for RayWithNegatives<D> {
    type Target = D;

    #[inline(always)]
    fn deref(&self) -> &D {
        &self.inner
    }
}

impl<D: RayData> std::ops::DerefMut for RayWithNegatives<D> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut D {
        &mut self.inner
    }
}

impl<D: RayData> RayData for RayWithNegatives<D> {
    type ZeroSet = D::ZeroSet;

    #[inline(always)]
    fn zero_set(&self) -> &Self::ZeroSet {
        self.inner.zero_set()
    }

    #[inline(always)]
    fn zero_set_mut(&mut self) -> &mut Self::ZeroSet {
        self.inner.zero_set_mut()
    }

    #[inline(always)]
    fn zero_set_signature(&self) -> u64 {
        self.inner.zero_set_signature()
    }

    #[inline(always)]
    fn set_zero_set_signature(&mut self, sig: u64) {
        self.inner.set_zero_set_signature(sig);
    }

    #[inline(always)]
    fn zero_set_count(&self) -> usize {
        self.inner.zero_set_count()
    }

    #[inline(always)]
    fn set_zero_set_count(&mut self, count: usize) {
        self.inner.set_zero_set_count(count);
    }

    #[inline(always)]
    fn first_infeasible_row(&self) -> Option<Row> {
        self.inner.first_infeasible_row()
    }

    #[inline(always)]
    fn set_first_infeasible_row(&mut self, row: Option<Row>) {
        self.inner.set_first_infeasible_row(row);
    }

    #[inline(always)]
    fn is_feasible(&self) -> bool {
        self.inner.is_feasible()
    }

    #[inline(always)]
    fn is_weakly_feasible(&self) -> bool {
        self.inner.is_weakly_feasible()
    }

    #[inline(always)]
    fn last_sign(&self) -> Sign {
        self.inner.last_sign()
    }
}

impl<D: RayData> RayWithNegatives<D> {
    #[inline(always)]
    pub(crate) fn negative_rows(&self) -> &RowSet {
        &self.negative_rows
    }
}

pub trait RayData: Clone + std::fmt::Debug {
    type ZeroSet: ZeroSet;

    fn zero_set(&self) -> &Self::ZeroSet;
    fn zero_set_mut(&mut self) -> &mut Self::ZeroSet;
    fn zero_set_signature(&self) -> u64;
    fn set_zero_set_signature(&mut self, sig: u64);
    fn zero_set_count(&self) -> usize;
    fn set_zero_set_count(&mut self, count: usize);
    fn first_infeasible_row(&self) -> Option<Row>;
    fn set_first_infeasible_row(&mut self, row: Option<Row>);
    fn is_feasible(&self) -> bool;
    fn is_weakly_feasible(&self) -> bool;
    fn last_sign(&self) -> Sign;
}

#[derive(Clone, Debug)]
pub(crate) struct RayClass<N: Num, ZS: ZeroSet = SatSet> {
    pub(crate) zero_set: ZS,
    pub(crate) zero_set_sig: u64,
    pub(crate) zero_set_count: usize,
    pub(crate) first_infeasible_row: Option<Row>,
    pub(crate) feasible: bool,
    pub(crate) weakly_feasible: bool,
    pub(crate) last_eval_row: Option<Row>,
    pub(crate) last_eval: N,
    pub(crate) last_sign: Sign,
}

#[derive(Clone, Copy, Debug)]
pub struct RayPartition<'a> {
    pub negative: &'a [RayId],
    pub positive: &'a [RayId],
    pub zero: &'a [RayId],
}

#[derive(Clone, Debug, Default)]
pub struct RayPartitionOwned {
    pub negative: Vec<RayId>,
    pub positive: Vec<RayId>,
    pub zero: Vec<RayId>,
}

impl<N: Num, ZS: ZeroSet> Ray<N, ZS> {
    pub fn vector(&self) -> &[N] {
        &self.vector
    }

    pub fn zero_set(&self) -> &ZS {
        &self.class.zero_set
    }

    pub fn first_infeasible_row(&self) -> Option<Row> {
        self.class.first_infeasible_row
    }

    pub fn is_feasible(&self) -> bool {
        self.class.feasible
    }

    pub fn is_weakly_feasible(&self) -> bool {
        self.class.weakly_feasible
    }

    pub fn last_eval(&self) -> &N {
        &self.class.last_eval
    }

    pub fn last_sign(&self) -> Sign {
        self.class.last_sign
    }
}

impl<N: Num, ZS: ZeroSet> RayData for Ray<N, ZS> {
    type ZeroSet = ZS;

    #[inline(always)]
    fn zero_set(&self) -> &ZS {
        &self.class.zero_set
    }

    #[inline(always)]
    fn zero_set_mut(&mut self) -> &mut ZS {
        &mut self.class.zero_set
    }

    #[inline(always)]
    fn zero_set_signature(&self) -> u64 {
        self.class.zero_set_sig
    }

    #[inline(always)]
    fn set_zero_set_signature(&mut self, sig: u64) {
        self.class.zero_set_sig = sig;
    }

    #[inline(always)]
    fn zero_set_count(&self) -> usize {
        self.class.zero_set_count
    }

    #[inline(always)]
    fn set_zero_set_count(&mut self, count: usize) {
        self.class.zero_set_count = count;
    }

    #[inline(always)]
    fn first_infeasible_row(&self) -> Option<Row> {
        self.class.first_infeasible_row
    }

    #[inline(always)]
    fn set_first_infeasible_row(&mut self, row: Option<Row>) {
        self.class.first_infeasible_row = row;
    }

    #[inline(always)]
    fn is_feasible(&self) -> bool {
        self.class.feasible
    }

    #[inline(always)]
    fn is_weakly_feasible(&self) -> bool {
        self.class.weakly_feasible
    }

    #[inline(always)]
    fn last_sign(&self) -> Sign {
        self.class.last_sign
    }
}

impl<N: Num, ZS: ZeroSet> AsRef<Ray<N, ZS>> for Ray<N, ZS> {
    #[inline(always)]
    fn as_ref(&self) -> &Ray<N, ZS> {
        self
    }
}

impl<N: Num, ZS: ZeroSet> AsMut<Ray<N, ZS>> for Ray<N, ZS> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut Ray<N, ZS> {
        self
    }
}

#[derive(Clone, Debug)]
pub struct AdjacencyEdge {
    pub retained: RayId,
    pub removed: RayId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EdgeTarget {
    Scheduled(Row),
    Stale(Row),
    Discarded,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RayListHeads {
    pub(crate) pos_head: Option<RayId>,
    pub(crate) zero_head: Option<RayId>,
    pub(crate) neg_head: Option<RayId>,
    pub(crate) pos_tail: Option<RayId>,
    pub(crate) zero_tail: Option<RayId>,
    pub(crate) neg_tail: Option<RayId>,
}

type RayArena<D> = Arena<D, NoopReset>;

#[derive(Clone, Debug)]
pub(crate) struct RayGraph<D: RayData> {
    pub(crate) arena: RayArena<D>,
    pub(crate) active_order: Vec<RayId>,
    pub(crate) artificial_ray: Option<RayId>,
    pub(crate) total_ray_count: usize,
    pub(crate) feasible_ray_count: usize,
    pub(crate) weakly_feasible_ray_count: usize,
    pub(crate) zero_ray_count: usize,
    pub(crate) slot_generations: Vec<u32>,
    pub(crate) slot_origins: Vec<RayOrigin>,
    pub(crate) edges: Vec<Vec<AdjacencyEdge>>,
    pub(crate) non_empty_edge_buckets: Vec<Row>,
    pub(crate) edge_bucket_positions: Vec<Option<usize>>,
    pub(crate) edge_count: usize,
}

impl<D: RayData> RayGraph<D> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn deactivate_ids(&mut self, ids: &[RayId]) {
        for &id in ids {
            let Some(ray) = self.arena.get(id.as_index()) else {
                continue;
            };
            if ray.is_feasible() {
                assert!(self.feasible_ray_count > 0, "feasible ray count underflow");
                self.feasible_ray_count -= 1;
            }
            if ray.is_weakly_feasible() {
                assert!(
                    self.weakly_feasible_ray_count > 0,
                    "weakly feasible ray count underflow"
                );
                self.weakly_feasible_ray_count -= 1;
            }
            if ray.last_sign() == Sign::Zero {
                assert!(self.zero_ray_count > 0, "zero ray count underflow");
                self.zero_ray_count -= 1;
            }
            self.arena.deactivate(id.as_index());
        }
    }

    pub(crate) fn reset(&mut self, row_count: Row) {
        self.arena.reset();
        self.active_order.clear();
        self.artificial_ray = None;
        self.total_ray_count = 0;
        self.feasible_ray_count = 0;
        self.weakly_feasible_ray_count = 0;
        self.zero_ray_count = 0;
        self.slot_generations.clear();
        self.slot_origins.clear();
        self.edge_count = 0;
        assert!(
            row_count <= usize::MAX - 2,
            "row count overflow while allocating edge buckets"
        );
        self.edges = vec![Vec::new(); row_count + 2];
        self.non_empty_edge_buckets.clear();
        self.edge_bucket_positions = vec![None; self.edges.len()];
    }

    #[inline(always)]
    pub(crate) fn ray_key(&self, id: RayId) -> RayKey {
        let slot = id.as_index();
        let generation = self.slot_generations.get(slot).copied().unwrap_or(0);
        RayKey { slot, generation }
    }

    pub(crate) fn set_ray_origin(
        &mut self,
        id: RayId,
        parent_a: Option<RayKey>,
        parent_b: Option<RayKey>,
        creation_row: Option<Row>,
    ) {
        let slot = id.as_index();
        let generation = self.slot_generations.get(slot).copied().unwrap_or(0);
        if slot >= self.slot_origins.len() {
            self.slot_origins.resize_with(slot + 1, RayOrigin::default);
        }
        self.slot_origins[slot] = RayOrigin {
            self_key: RayKey { slot, generation },
            parent_a,
            parent_b,
            creation_row,
        };
    }

    pub(crate) fn ray_origin(&self, id: RayId) -> Option<&RayOrigin> {
        let key = self.ray_key(id);
        self.slot_origins
            .get(id.as_index())
            .filter(|origin| origin.self_key == key)
    }

    pub(crate) fn insert_active(&mut self, ray_data: D) -> RayId {
        let feasible = ray_data.is_feasible();
        let weakly_feasible = ray_data.is_weakly_feasible();
        let zero_eval = ray_data.last_sign() == Sign::Zero;
        let slot = self.arena.insert(ray_data, true).as_index();
        if slot >= self.slot_generations.len() {
            self.slot_generations.resize(slot + 1, 0);
        }
        let next = self.slot_generations[slot].wrapping_add(1);
        self.slot_generations[slot] = if next == 0 { 1 } else { next };

        if slot >= self.slot_origins.len() {
            self.slot_origins.resize_with(slot + 1, RayOrigin::default);
        }
        let generation = self.slot_generations[slot];
        self.slot_origins[slot] = RayOrigin {
            self_key: RayKey { slot, generation },
            parent_a: None,
            parent_b: None,
            creation_row: None,
        };

        let id = RayId(slot);
        self.active_order.push(id);
        self.total_ray_count += 1;
        if feasible {
            self.feasible_ray_count += 1;
        }
        if weakly_feasible {
            self.weakly_feasible_ray_count += 1;
        }
        if zero_eval {
            self.zero_ray_count += 1;
        }
        id
    }

    pub(crate) fn insert_inactive(&mut self, ray_data: D) -> RayId {
        let slot = self.arena.insert(ray_data, false).as_index();
        if slot >= self.slot_generations.len() {
            self.slot_generations.resize(slot + 1, 0);
        }
        let next = self.slot_generations[slot].wrapping_add(1);
        self.slot_generations[slot] = if next == 0 { 1 } else { next };

        if slot >= self.slot_origins.len() {
            self.slot_origins.resize_with(slot + 1, RayOrigin::default);
        }
        let generation = self.slot_generations[slot];
        self.slot_origins[slot] = RayOrigin {
            self_key: RayKey { slot, generation },
            parent_a: None,
            parent_b: None,
            creation_row: None,
        };

        RayId(slot)
    }

    pub(crate) fn ray_data(&self, id: RayId) -> Option<&D> {
        self.arena.get(id.as_index())
    }

    pub(crate) fn ray_data_mut(&mut self, id: RayId) -> Option<&mut D> {
        self.arena.get_mut(id.as_index())
    }

    #[cfg(test)]
    pub(crate) fn set_order(&mut self, mut order: Vec<RayId>) {
        order.retain(|id| self.arena.get(id.as_index()).is_some());
        let zero_count = order
            .iter()
            .filter(|id| {
                self.ray_data(**id)
                    .map(|r| r.last_sign() == Sign::Zero)
                    .unwrap_or(false)
            })
            .count();
        self.set_order_with_zero_count_unchecked(order, zero_count);
        self.recompute_counts();
    }

    pub(crate) fn active_rays(&self) -> impl Iterator<Item = &D> {
        self.active_order
            .iter()
            .filter_map(|rid| self.ray_data(*rid))
    }

    pub(crate) fn copy_active_ids(&self, dest: &mut Vec<RayId>) {
        dest.clear();
        dest.extend(self.active_order.iter().copied());
    }

    pub(crate) fn take_active_order(&mut self) -> Vec<RayId> {
        std::mem::take(&mut self.active_order)
    }

    pub(crate) fn set_order_with_zero_count_unchecked(
        &mut self,
        order: Vec<RayId>,
        zero_count: usize,
    ) {
        self.active_order = order;
        self.zero_ray_count = zero_count.min(self.active_len());
    }

    pub(crate) fn recompute_counts(&mut self) {
        let mut feasible_count = 0;
        let mut weakly_feasible_count = 0;
        let mut zero_count = 0;
        for rid in self.active_order.iter() {
            let Some(ray) = self.ray_data(*rid) else {
                continue;
            };
            if ray.is_feasible() {
                feasible_count += 1;
            }
            if ray.is_weakly_feasible() {
                weakly_feasible_count += 1;
            }
            if ray.last_sign() == Sign::Zero {
                zero_count += 1;
            }
        }
        self.feasible_ray_count = feasible_count;
        self.weakly_feasible_ray_count = weakly_feasible_count;
        self.zero_ray_count = zero_count.min(self.active_len());
    }

    pub(crate) fn remove_many_keep_order(&mut self, ids: &[RayId]) {
        self.deactivate_ids(ids);
    }

    pub(crate) fn deactivate_all(&mut self) {
        self.arena.deactivate_all();
        self.active_order.clear();
        self.feasible_ray_count = 0;
        self.weakly_feasible_ray_count = 0;
        self.zero_ray_count = 0;
    }

    pub(crate) fn active_len(&self) -> usize {
        self.active_order.len()
    }

    pub(crate) fn weakly_feasible_len(&self) -> usize {
        self.weakly_feasible_ray_count
    }

    pub(crate) fn zero_len(&self) -> usize {
        self.zero_ray_count
    }

    #[cfg(test)]
    pub(crate) fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub(crate) fn ensure_edge_capacity(&mut self, iteration: Row) {
        if iteration >= self.edges.len() {
            self.edges.resize_with(iteration + 1, Vec::new);
        }
        if iteration >= self.edge_bucket_positions.len() {
            self.edge_bucket_positions.resize(iteration + 1, None);
        }
    }

    pub(crate) fn mark_bucket_non_empty(&mut self, iteration: Row) {
        if iteration >= self.edge_bucket_positions.len() {
            self.edge_bucket_positions.resize(iteration + 1, None);
        }
        if self.edge_bucket_positions[iteration].is_none() {
            let pos = self.non_empty_edge_buckets.len();
            self.non_empty_edge_buckets.push(iteration);
            self.edge_bucket_positions[iteration] = Some(pos);
        }
    }

    pub(crate) fn mark_bucket_empty(&mut self, iteration: Row) {
        if iteration >= self.edge_bucket_positions.len() {
            return;
        }
        if let Some(pos) = self.edge_bucket_positions[iteration].take() {
            debug_assert!(
                pos < self.non_empty_edge_buckets.len(),
                "edge bucket position out of bounds while clearing bucket"
            );
            let removed = self.non_empty_edge_buckets.swap_remove(pos);
            debug_assert_eq!(removed, iteration);
            if let Some(&moved) = self.non_empty_edge_buckets.get(pos) {
                self.edge_bucket_positions[moved] = Some(pos);
            }
        }
    }

    pub(crate) fn queue_edge(&mut self, iteration: Row, edge: AdjacencyEdge) {
        self.schedule_edge(iteration, edge);
    }

    pub(crate) fn schedule_edge(&mut self, iteration: Row, edge: AdjacencyEdge) {
        self.ensure_edge_capacity(iteration);
        let was_empty = self.edges[iteration].is_empty();
        self.edges[iteration].push(edge);
        if was_empty {
            self.mark_bucket_non_empty(iteration);
        }
        self.edge_count += 1;
    }

    pub(crate) fn take_edges(&mut self, iteration: Row) -> Vec<AdjacencyEdge> {
        if iteration >= self.edges.len() {
            return Vec::new();
        }
        let mut bucket = Vec::new();
        std::mem::swap(&mut bucket, &mut self.edges[iteration]);
        if !bucket.is_empty() {
            self.mark_bucket_empty(iteration);
        }
        assert!(
            self.edge_count >= bucket.len(),
            "edge count underflow while draining bucket"
        );
        self.edge_count -= bucket.len();
        bucket
    }

    pub(crate) fn edge_buckets_len(&self) -> usize {
        self.edges.len()
    }

    #[cfg(test)]
    pub(crate) fn edges_at(&self, iteration: Row) -> &[AdjacencyEdge] {
        self.edges.get(iteration).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn artificial_ray(&self) -> Option<RayId> {
        self.artificial_ray
    }

    pub(crate) fn set_artificial(&mut self, id: RayId) {
        self.artificial_ray = Some(id);
    }
}

impl<D: RayData> Default for RayGraph<D> {
    fn default() -> Self {
        Self {
            arena: RayArena::default(),
            active_order: Vec::new(),
            artificial_ray: None,
            total_ray_count: 0,
            feasible_ray_count: 0,
            weakly_feasible_ray_count: 0,
            zero_ray_count: 0,
            slot_generations: Vec::new(),
            slot_origins: Vec::new(),
            edges: Vec::new(),
            non_empty_edge_buckets: Vec::new(),
            edge_bucket_positions: Vec::new(),
            edge_count: 0,
        }
    }
}
