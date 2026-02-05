use crate::HowzatError as Error;
use crate::dd::Umpire;
use crate::dd::index::RayIncidenceIndex as _;
use crate::dd::mode::{HalfspaceMode, UmpireHalfspaceMode};
use crate::dd::ray::{
    AdjacencyEdge, EdgeTarget, RayData as _, RayId, RayListHeads, RayPartition, RayPartitionOwned,
};
use crate::dd::state::{ConeDd, ConeEngine, ConeOutput};
use crate::dd::zero::{ZeroRepr, ZeroSet as _};
use calculo::num::{Num, Sign};
use hullabaloo::types::{ComputationStatus, Representation, RepresentationKind, Row, RowSet};

#[derive(Clone, Copy)]
enum DegeneracyCandidates<'a> {
    Slice(&'a [RayId]),
    BitSet(&'a RowSet),
}

impl<N: Num, R: Representation, U, ZR> ConeDd<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: crate::dd::mode::PreorderedBackend,
{
    pub fn run_to_completion(mut self) -> Result<ConeOutput<N, R, U, ZR>, Error> {
        loop {
            if self.state.core.recompute_row_order {
                let (ctx, umpire, strict) = (
                    &mut self.state.core.ctx,
                    &mut self.state.umpire,
                    &self.state.core._strict_inequality_set,
                );
                umpire.recompute_row_order_vector(ctx, strict);
                let priority = self.state.core.weakly_added_halfspaces.clone();
                self.state
                    .umpire
                    .bump_priority_rows(&mut self.state.core.ctx, &priority);
                <UmpireHalfspaceMode<N, U, ZR> as HalfspaceMode>::on_row_order_recomputed(
                    &mut self.state,
                );
                #[cfg(debug_assertions)]
                {
                    self.state.debug_assert_iteration_prefix();
                }
                self.state.core.recompute_row_order = false;
                self.state.core.edges_dirty = true;
            }
            if self.state.core.comp_status == ComputationStatus::RegionEmpty
                || self.state.core.comp_status == ComputationStatus::AllFound
            {
                break;
            }
            let active = self.state.core.ray_graph.active_len();
            let Some(hh) = self.state.umpire.choose_next_halfspace(
                &self.state.core.ctx,
                &self.state.core.weakly_added_halfspaces,
                self.state.core.iter_state.iteration,
                active,
            ) else {
                self.state.core.comp_status = ComputationStatus::AllFound;
                break;
            };
            <UmpireHalfspaceMode<N, U, ZR> as HalfspaceMode>::add_halfspace(
                &mut self.state,
                hh,
            )?;
        }
        Ok(ConeOutput { state: self.state })
    }
}

impl<N: Num, R: Representation, U, ZR> ConeEngine<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    pub(crate) fn update_ray_orders(&mut self) {
        let relaxed = self.core.options.relaxed_enumeration();
        let mut ids = std::mem::take(&mut self.core.active_id_scratch);
        self.core.ray_graph.copy_active_ids(&mut ids);

        let mut changed = false;
        for id in ids.iter().copied() {
            let Some(ray_data) = self.core.ray_graph.ray_data_mut(id) else {
                continue;
            };
            let old = ray_data.first_infeasible_row();
            self.umpire
                .update_first_infeasible_row(&self.core.ctx, ray_data, relaxed);
            changed |= old != ray_data.first_infeasible_row();
        }
        self.core.active_id_scratch = ids;
        if changed {
            self.core.edges_dirty = true;
        }
    }

    pub fn ray_count(&self) -> usize {
        self.core.ray_graph.active_len()
    }

    #[inline(always)]
    fn take_zero_set(&mut self) -> <ZR as ZeroRepr>::Set {
        self.core
            .zero_set_pool
            .pop()
            .unwrap_or_else(|| ZR::empty_set(self.row_count()))
    }

    #[inline(always)]
    fn recycle_zero_set(&mut self, zero_set: <ZR as ZeroRepr>::Set) {
        self.core.zero_set_pool.push(zero_set);
    }

    #[inline(always)]
    fn recycle_zero_set_from_ray_data(&mut self, ray_data: &mut U::RayData) {
        let mut set = ZR::empty_set(0);
        std::mem::swap(ray_data.zero_set_mut(), &mut set);
        self.recycle_zero_set(set);
    }

    fn position_of_row(&self, row: Row) -> Option<Row> {
        self.core
            .ctx
            .row_to_pos
            .get(row)
            .copied()
            .filter(|pos| *pos < self.row_count())
    }

    pub(crate) fn first_infeasible_position(&self, ray: &U::RayData) -> Option<Row> {
        ray.first_infeasible_row()
            .and_then(|row| self.position_of_row(row))
    }

    fn first_infeasible_position_or_m(&self, ray: &U::RayData) -> Row {
        self.first_infeasible_position(ray)
            .unwrap_or_else(|| self.row_count() + 1)
    }

    fn sync_iteration_with_added(&mut self) {
        let iter = self
            .core
            .weakly_added_halfspaces
            .cardinality()
            .min(self.row_count());
        self.core.iter_state.iteration = iter;
    }

    #[cfg(debug_assertions)]
    fn debug_assert_iteration_prefix(&self) {
        let iter = self.core.iter_state.iteration.min(self.row_count());
        debug_assert_eq!(
            iter,
            self.core.weakly_added_halfspaces.cardinality(),
            "iteration {} diverges from weakly-added count {}",
            iter,
            self.core.weakly_added_halfspaces.cardinality()
        );
        for pos in 0..iter {
            let row = self.core.ctx.order_vector[pos];
            debug_assert!(
                self.core.weakly_added_halfspaces.contains(row),
                "row {} at position {} not marked weakly added",
                row,
                pos
            );
        }
        for row in 0..self.row_count() {
            if self.core.weakly_added_halfspaces.contains(row) {
                let in_prefix = self
                    .position_of_row(row)
                    .map(|pos| pos < iter)
                    .unwrap_or(false);
                debug_assert!(
                    in_prefix,
                    "weakly-added row {} not placed in iteration prefix",
                    row
                );
            }
        }
    }

    pub(crate) fn add_new_halfspace_dynamic(&mut self, row: Row) -> Result<(), Error>
    where
        ZR: crate::dd::mode::PreorderedBackend,
    {
        self.tableau_entering_for_row(row);
        let partition = self.evaluate_row_partition(row);
        let partition_view = Self::partition_view(&partition);

        if matches!(R::KIND, RepresentationKind::Inequality)
            && partition_view.negative.is_empty()
            && self.core.ctx.equality_kinds[row] != hullabaloo::types::InequalityKind::Equality
        {
            self.core.redundant_halfspaces.insert(row);
            self.core.weakly_added_halfspaces.insert(row);
            self.sync_iteration_with_added();
            self.sync_tableau_flag_for_row(row);
            if self.core.ray_graph.active_len() == self.core.ray_graph.weakly_feasible_len() {
                self.core.comp_status = ComputationStatus::AllFound;
            }
            self.recycle_partition(partition);
            return Ok(());
        }
        if partition_view.positive.is_empty() && partition_view.zero.is_empty() {
            self.core.ray_graph.deactivate_all();
            self.umpire.reset_infeasible_counts(self.row_count());
            self.clear_ray_indices();
            self.core.pending_new_rays.clear();
            self.core.comp_status = ComputationStatus::RegionEmpty;
            self.recycle_partition(partition);
            return Ok(());
        }

        let (new_id, inserted) = ZR::ensure_id_for_row(&mut self.core.ctx, row);
        if inserted {
            for &ray_id in partition_view.zero {
                self.unregister_ray_id(ray_id);
                if let Some(ray_data) = self.ray_mut(ray_id) {
                    ray_data.zero_set_mut().insert(new_id);
                    ray_data.set_zero_set_signature(ray_data.zero_set().signature_u64());
                    ray_data.set_zero_set_count(ray_data.zero_set_count() + 1);
                }
                self.register_ray_id(ray_id);
            }
        }

        if partition_view.negative.is_empty() {
            self.core.weakly_added_halfspaces.insert(row);
            self.sync_iteration_with_added();
            self.sync_tableau_flag_for_row(row);
            if self.core.ray_graph.active_len() == self.core.ray_graph.weakly_feasible_len() {
                self.core.comp_status = ComputationStatus::AllFound;
            }
            self.recycle_partition(partition);
            return Ok(());
        }

        self.with_active_ray_ids(|state, adjacency_candidates| {
            for &pos in partition_view.positive {
                for &neg in partition_view.negative {
                    if !state.check_adjacency(neg, pos, adjacency_candidates) {
                        continue;
                    }
                    let _ = state.create_new_ray(neg, pos, row);
                }
            }
        });
        self.delete_negative_rays(partition_view);
        self.prune_recent_rays();
        self.core.added_halfspaces.insert(row);
        self.core.weakly_added_halfspaces.insert(row);
        self.sync_iteration_with_added();
        self.sync_tableau_flag_for_row(row);
        <UmpireHalfspaceMode<N, U, ZR> as HalfspaceMode>::on_halfspace_added(self, row);
        if self.core.ray_graph.active_len() == self.core.ray_graph.weakly_feasible_len() {
            self.core.comp_status = ComputationStatus::AllFound;
        }
        self.recycle_partition(partition);
        Ok(())
    }

    pub fn initialize_rays(&mut self)
    where
        ZR: crate::dd::mode::PreorderedBackend,
    {
        self.core.ray_graph.reset(self.row_count());
        self.umpire.reset_infeasible_counts(self.row_count());
        self.core.tableau.tableau_nonbasic = Self::init_nonbasic(self.col_count());
        self.core.tableau.tableau_basic_col_for_row = vec![-1; self.row_count()];
        if matches!(R::KIND, RepresentationKind::Generator) {
            for row in 0..self.row_count() {
                let _ = ZR::ensure_id_for_row(&mut self.core.ctx, row);
            }
        }
        self.add_artificial_ray();
        self.clear_ray_indices();

        self.core.initial_halfspaces.clear();
        for &row in &self.core.iter_state.initial_ray_index {
            if let Some(r) = row {
                self.core.initial_halfspaces.insert(r);
            }
        }

        self.core.added_halfspaces = self.core.initial_halfspaces.clone();
        self.core.weakly_added_halfspaces = self.core.initial_halfspaces.clone();
        for row_id in self.core.weakly_added_halfspaces.iter() {
            let row = row_id.as_index();
            let _ = ZR::ensure_id_for_row(&mut self.core.ctx, row);
        }
        self.sync_tableau_flags();
        let initial = self.core.initial_halfspaces.clone();
        self.umpire.bump_priority_rows(&mut self.core.ctx, &initial);
        self.core.edges_dirty = true;
        let last_row = self.core.ctx.order_vector.first().copied();
        let m = self.row_count();
        let wants_purify = self.umpire.wants_initial_purification();
        let mut expected_zero = RowSet::new(0);
        if wants_purify {
            expected_zero.resize(m);
        }
        for col in 0..self.col_count() {
            let mut vec = self
                .umpire
                .basis_column_vector(&self.core.tableau.basis, col);
            let pivot_row = self
                .core
                .iter_state
                .initial_ray_index
                .get(col)
                .copied()
                .flatten();
            if !self.umpire.normalize_vector(&mut vec) {
                continue;
            }
            if wants_purify {
                expected_zero.copy_from(&self.core.initial_halfspaces);
                expected_zero.union_inplace(&self.core.equality_set);
                if let Some(row) = pivot_row
                    && !self.core.equality_set.contains(row)
                {
                    expected_zero.remove(row);
                }

                if let Some(mut purified) =
                    self.umpire
                        .purify_vector_from_zero_set(&self.core.ctx, &expected_zero)
                    && self.umpire.normalize_vector(&mut purified)
                {
                    self.umpire.align_vector_in_place(&vec, &mut purified);
                    vec = purified;
                }
            }
            let has_initial = self
                .core
                .iter_state
                .initial_ray_index
                .get(col)
                .copied()
                .flatten()
                .is_some();
            let add_negative = !has_initial;
            let neg_vec = add_negative.then(|| {
                let mut neg = vec.clone();
                self.umpire.negate_vector_in_place(&mut neg);
                neg
            });
            // cddlib seeds use strict feasibility checks; do not apply relaxed handling here.
            let zero_set = self.take_zero_set();
            let mut ray_data = self
                .umpire
                .classify_vector(&self.core.ctx, vec, false, last_row, zero_set);
            let satisfies_equalities = self.core.equality_set.iter().all(|row_id| {
                let row = row_id.as_index();
                ZR::zero_set_contains_row(&self.core.ctx, ray_data.zero_set(), row)
            });
            if satisfies_equalities {
                self.umpire.on_ray_inserted(&self.core.ctx, &ray_data, false);
                let id = self.core.ray_graph.insert_active(ray_data);
                self.register_ray_id(id);
                if let Some(neg_vec) = neg_vec {
                    let zero_set = self.take_zero_set();
                    let neg_ray_data =
                        self.umpire
                            .classify_vector(&self.core.ctx, neg_vec, false, last_row, zero_set);
                    self.umpire
                        .on_ray_inserted(&self.core.ctx, &neg_ray_data, false);
                    let id = self.core.ray_graph.insert_active(neg_ray_data);
                    self.register_ray_id(id);
                }
            } else {
                self.recycle_zero_set_from_ray_data(&mut ray_data);
                self.umpire.recycle_ray_data(&mut ray_data);
            }
        }
        self.core.iter_state.iteration = self.col_count();
        self.with_active_ray_ids(|state, active_ids| {
            let mut floored = false;
            if state.core.iter_state.iteration < state.core.ctx.order_vector.len() {
                let floor_row = state.core.ctx.order_vector[state.core.iter_state.iteration];
                let floor_pos = state.core.iter_state.iteration;
                for &idx in active_ids {
                    if let Some(ray_pos) = state
                        .ray(idx)
                        .map(|ray| state.first_infeasible_position_or_m(ray))
                        && ray_pos < floor_pos
                        && let Some(ray_mut) = state.ray_mut(idx)
                    {
                        ray_mut.set_first_infeasible_row(Some(floor_row));
                        floored = true;
                    }
                }
            }
            if state.enforce_first_infeasible_floor(active_ids) {
                floored = true;
            }
            if floored {
                state.core.edges_dirty = true;
            }
        });
        <UmpireHalfspaceMode<N, U, ZR> as HalfspaceMode>::on_rays_initialized(self);
        self.core.iter_state.iteration = self.col_count().saturating_add(1);
        if self.core.iter_state.iteration > self.row_count() {
            self.core.iter_state.iteration = self.row_count();
        }
        if self.core.iter_state.iteration > self.row_count() {
            self.core.comp_status = ComputationStatus::AllFound;
        }

        // Start iteration counter aligned to rows that are actually marked added.
        // At initialization nothing has been added yet, so keep the iteration prefix empty.
        self.sync_iteration_with_added();
    }

    fn partition_view<'a>(partition: &'a RayPartitionOwned) -> RayPartition<'a> {
        RayPartition {
            negative: &partition.negative,
            positive: &partition.positive,
            zero: &partition.zero,
        }
    }

    pub(crate) fn recycle_partition(&mut self, mut partition: RayPartitionOwned) {
        partition.negative.clear();
        partition.positive.clear();
        partition.zero.clear();
        self.core.partitions = partition;
    }

    pub(crate) fn check_adjacency(&mut self, r1: RayId, r2: RayId, candidates: &[RayId]) -> bool {
        let (ray1_zero, ray1_zero_count, ray2_zero, ray2_zero_count) = {
            let ray_graph = &self.core.ray_graph;
            let (Some(ray1), Some(ray2)) = (ray_graph.ray_data(r1), ray_graph.ray_data(r2))
            else {
                return false;
            };
            (
                ray1.zero_set(),
                ray1.zero_set_count(),
                ray2.zero_set(),
                ray2.zero_set_count(),
            )
        };

        self.core.adj_face.copy_from(ray1_zero);
        let common = self.core.adj_face.intersection_inplace_and_count(ray2_zero);

        let required = self.adjacency_dimension().saturating_sub(2);
        if common < required {
            return false;
        }

        if self.core.options.assumes_nondegeneracy() {
            return true;
        }

        let min_parent_saturators = ray1_zero_count.min(ray2_zero_count);
        if common == required && min_parent_saturators == required + 1 {
            return true;
        }

        let domain = ZR::domain_size(&self.core.ctx);
        !self
            .core
            .ray_incidence
            .candidate_contains_face(
                &self.core.adj_face,
                domain,
                candidates,
                r1,
                r2,
            )
    }

    #[allow(dead_code)]
    fn reclassify_active_rays(&mut self) {
        let mut ids = std::mem::take(&mut self.core.active_id_scratch);
        self.core.ray_graph.copy_active_ids(&mut ids);
        let relaxed = self.core.options.relaxed_enumeration();
        for id in ids.iter().copied() {
            let Some(ray_data) = self.core.ray_graph.ray_data_mut(id) else {
                continue;
            };
            self.umpire.reclassify_ray(&self.core.ctx, ray_data, relaxed);
        }
        self.core.active_id_scratch = ids;
        self.rebuild_ray_index();
        self.core.ray_graph.recompute_counts();
    }

    fn prune_recent_rays(&mut self) {
        self.core.pending_new_rays.clear();
    }

    pub(crate) fn delete_negative_rays(&mut self, partition: RayPartition<'_>) {
        let relaxed = self.core.options.relaxed_enumeration();
        for &idx in partition.negative.iter() {
            let ray_data = self.core.ray_graph.ray_data(idx);
            if let Some(ray_data) = ray_data {
                self.umpire
                    .on_ray_removed(&self.core.ctx, ray_data, relaxed);
            }
            self.unregister_ray_id(idx);
            if let Some(ray_data) = self.core.ray_graph.ray_data_mut(idx) {
                let mut set = ZR::empty_set(0);
                std::mem::swap(ray_data.zero_set_mut(), &mut set);
                self.core.zero_set_pool.push(set);
                self.umpire.recycle_ray_data(ray_data);
            }
        }
        self.core
            .ray_graph
            .remove_many_keep_order(partition.negative);
        self.discard_pending_rays(partition.negative);

        let mut zeros = std::mem::take(&mut self.core.active_id_scratch);
        zeros.clear();
        zeros.reserve(partition.zero.len() + self.core.pending_new_rays.len());
        zeros.extend_from_slice(partition.zero);

        let mut order = self.core.ray_graph.take_active_order();
        order.clear();
        order.reserve(partition.positive.len() + zeros.len() + self.core.pending_new_rays.len());
        order.extend_from_slice(partition.positive);

        let mut max_id = 0usize;
        for &rid in partition.positive.iter().chain(zeros.iter()) {
            max_id = max_id.max(rid.as_index());
        }
        for &rid in &self.core.pending_new_rays {
            max_id = max_id.max(rid.as_index());
        }
        if max_id >= self.core.removed_marks.len() {
            self.core.removed_marks.resize(max_id + 1, 0);
        }
        self.core.removed_epoch = self.core.removed_epoch.wrapping_add(1);
        if self.core.removed_epoch == 0 {
            self.core.removed_epoch = 1;
            self.core.removed_marks.fill(0);
        }
        let epoch = self.core.removed_epoch;
        for &rid in partition.positive.iter().chain(zeros.iter()) {
            self.core.removed_marks[rid.as_index()] = epoch;
        }

        for &id in &self.core.pending_new_rays {
            if self.core.removed_marks[id.as_index()] == epoch {
                continue;
            }
            let Some(ray) = self.ray(id) else { continue };
            if ray.last_sign() == Sign::Positive {
                order.push(id);
            } else {
                zeros.push(id);
            }
            self.core.removed_marks[id.as_index()] = epoch;
        }

        zeros.sort_unstable_by(|a, b| {
            let fa = self
                .ray(*a)
                .map(|r| self.first_infeasible_position_or_m(r))
                .unwrap_or(self.row_count());
            let fb = self
                .ray(*b)
                .map(|r| self.first_infeasible_position_or_m(r))
                .unwrap_or(self.row_count());
            fa.cmp(&fb).then_with(|| a.as_index().cmp(&b.as_index()))
        });

        let pos_head = order.first().copied();
        let pos_tail = order.last().copied();
        let zero_head = zeros.first().copied();
        let zero_tail = zeros.last().copied();
        let zero_count = zeros.len();
        order.extend_from_slice(&zeros);

        self.core
            .ray_graph
            .set_order_with_zero_count_unchecked(order, zero_count);

        self.core.lists = RayListHeads::default();
        self.core.lists.pos_head = pos_head;
        self.core.lists.pos_tail = pos_tail;
        self.core.lists.zero_head = zero_head;
        self.core.lists.zero_tail = zero_tail;
        self.core.active_id_scratch = zeros;
        if self.core.ray_graph.active_len() == 0 {
            self.core.comp_status = ComputationStatus::RegionEmpty;
            self.umpire.reset_infeasible_counts(self.row_count());
        }
    }

    pub(crate) fn refresh_edge_buckets(&mut self) {
        let row_cap = self.row_count();
        assert!(
            row_cap < usize::MAX,
            "row count overflow while refreshing edge buckets"
        );
        let target_len = self.core.ray_graph.edge_buckets_len().max(row_cap + 1);
        if target_len > 0 {
            self.core.ray_graph.ensure_edge_capacity(target_len - 1);
        }
        self.core.ray_graph.edge_count = 0;
        let buckets_to_process = std::mem::take(&mut self.core.ray_graph.non_empty_edge_buckets);
        for &bucket_idx in &buckets_to_process {
            debug_assert!(
                bucket_idx < self.core.ray_graph.edge_bucket_positions.len(),
                "edge bucket index out of range while clearing bucket positions"
            );
            self.core.ray_graph.edge_bucket_positions[bucket_idx] = None;
        }
        for bucket_idx in buckets_to_process {
            debug_assert!(
                bucket_idx < self.core.ray_graph.edges.len(),
                "edge bucket index out of range while draining buckets"
            );
            let mut bucket = Vec::new();
            std::mem::swap(&mut bucket, &mut self.core.ray_graph.edges[bucket_idx]);
            for edge in bucket.into_iter() {
                let target = self.edge_target_iteration(&edge);
                match target {
                    EdgeTarget::Scheduled(row) => self.core.ray_graph.schedule_edge(row, edge),
                    EdgeTarget::Stale(_) => {}
                    EdgeTarget::Discarded => {}
                }
            }
        }
    }

    pub(crate) fn edge_target_iteration(&self, edge: &AdjacencyEdge) -> EdgeTarget {
        let Some(ray1) = self.ray(edge.retained) else {
            return EdgeTarget::Discarded;
        };
        let Some(ray2) = self.ray(edge.removed) else {
            return EdgeTarget::Discarded;
        };
        let m = self.row_count();
        let f1 = self.first_infeasible_position_or_m(ray1);
        let f2 = self.first_infeasible_position_or_m(ray2);
        if f1 == f2 {
            return EdgeTarget::Discarded;
        }
        if f1 >= m && f2 >= m {
            return EdgeTarget::Discarded;
        }
        let fmin = f1.min(f2);
        if fmin < self.core.iter_state.iteration {
            return EdgeTarget::Stale(fmin);
        }
        EdgeTarget::Scheduled(fmin)
    }

    #[cfg(test)]
    pub(crate) fn add_ray(&mut self, vector: Vec<U::Scalar>) -> RayId {
        let relaxed = self.core.options.relaxed_enumeration();
        let last_row = self.core.ctx.order_vector.first().copied();
        let zero_set = self.take_zero_set();
        let ray_data = self
            .umpire
            .classify_vector(&self.core.ctx, vector, relaxed, last_row, zero_set);
        debug_assert!(!ray_data.is_feasible() || ray_data.is_weakly_feasible());
        self.umpire
            .on_ray_inserted(&self.core.ctx, &ray_data, relaxed);
        let id = self.core.ray_graph.insert_active(ray_data);
        self.register_ray_id(id);
        self.record_new_ray(id);
        id
    }

    pub(crate) fn add_artificial_ray(&mut self) -> RayId {
        if let Some(idx) = self.core.ray_graph.artificial_ray() {
            return idx;
        }
        let dimension = self.col_count().max(1);
        let vector = self.umpire.zero_vector(dimension);
        let last_row = self.core.ctx.order_vector.first().copied();
        let zero_set = self.take_zero_set();
        let ray_data = self
            .umpire
            .classify_vector(&self.core.ctx, vector, false, last_row, zero_set);
        let idx = self.core.ray_graph.insert_inactive(ray_data);
        self.core.ray_graph.set_artificial(idx);
        idx
    }

    pub(crate) fn evaluate_row_partition(&mut self, row: Row) -> RayPartitionOwned {
        self.core.lists = RayListHeads::default();
        let mut partition = std::mem::take(&mut self.core.partitions);
        partition.negative.clear();
        partition.positive.clear();
        partition.zero.clear();
        let mut active_order = self.core.ray_graph.take_active_order();
        {
            let ctx = &self.core.ctx;
            let (umpire, ray_graph) = (&mut self.umpire, &mut self.core.ray_graph);
            for &idx in active_order.iter() {
                let Some(ray_data) = ray_graph.ray_data_mut(idx) else {
                    continue;
                };
                let sign = umpire.classify_ray(ctx, ray_data, row);
                match sign {
                    Sign::Negative => partition.negative.push(idx),
                    Sign::Positive => partition.positive.push(idx),
                    Sign::Zero => partition.zero.push(idx),
                }
            }
        }

        self.core.lists.neg_head = partition.negative.first().copied();
        self.core.lists.neg_tail = partition.negative.last().copied();
        self.core.lists.pos_head = partition.positive.first().copied();
        self.core.lists.pos_tail = partition.positive.last().copied();
        self.core.lists.zero_head = partition.zero.first().copied();
        self.core.lists.zero_tail = partition.zero.last().copied();

        let floored_pos = self.enforce_first_infeasible_floor(&partition.positive);
        let floored_zero = self.enforce_first_infeasible_floor(&partition.zero);
        if floored_pos || floored_zero {
            self.core.edges_dirty = true;
        }

        let mut cursor = 0usize;
        for &id in partition
            .negative
            .iter()
            .chain(partition.positive.iter())
            .chain(partition.zero.iter())
        {
            if cursor < active_order.len() {
                active_order[cursor] = id;
            } else {
                active_order.push(id);
            }
            cursor += 1;
        }
        active_order.truncate(cursor);

        self.core
            .ray_graph
            .set_order_with_zero_count_unchecked(active_order, partition.zero.len());
        partition
    }

    pub(crate) fn create_new_ray(&mut self, r1: RayId, r2: RayId, row: Row) -> Option<RayId> {
        if r1 == r2 {
            return None;
        }

        let relaxed = self.core.options.relaxed_enumeration();
        let parent_a = self.core.ray_graph.ray_key(r1);
        let parent_b = self.core.ray_graph.ray_key(r2);
        let ray_data_res = {
            let zero_set = self.take_zero_set();
            if self.core.ray_graph.ray_data(r1).is_none() || self.core.ray_graph.ray_data(r2).is_none() {
                self.recycle_zero_set(zero_set);
                return None;
            }
            let ray1 = self.core.ray_graph.ray_data(r1).expect("ray id checked");
            let ray2 = self.core.ray_graph.ray_data(r2).expect("ray id checked");
            self.umpire.generate_new_ray(
                &self.core.ctx,
                (r1, ray1, r2, ray2),
                row,
                relaxed,
                zero_set,
            )
        };
        let mut ray_data = match ray_data_res {
            Ok(ray_data) => ray_data,
            Err(zero_set) => {
                self.recycle_zero_set(zero_set);
                return None;
            }
        };
        if self.ray_exists(&ray_data) {
            self.core.dedup_drops = self.core.dedup_drops.wrapping_add(1);
            self.recycle_zero_set_from_ray_data(&mut ray_data);
            self.umpire.recycle_ray_data(&mut ray_data);
            return None;
        }

        self.umpire.on_ray_inserted(&self.core.ctx, &ray_data, relaxed);
        let id = self.core.ray_graph.insert_active(ray_data);
        self.core
            .ray_graph
            .set_ray_origin(id, Some(parent_a), Some(parent_b), Some(row));
        self.register_ray_id(id);
        self.record_new_ray(id);
        Some(id)
    }

    pub fn expand_ray_vector(&self, compact: &[N]) -> Vec<N> {
        assert_eq!(
            compact.len(),
            self.col_count(),
            "expand_ray_vector expects a vector of length col_count"
        );
        assert_eq!(
            self.core.iter_state.newcol.len(),
            self.core.iter_state.d_orig,
            "column remapping must match original dimension"
        );
        if !self.core.iter_state.col_reduced || self.core.iter_state.d_orig == self.col_count() {
            return compact.to_vec();
        }
        let mut full = vec![N::zero(); self.core.iter_state.d_orig];
        for (orig_idx, &map) in self.core.iter_state.newcol.iter().enumerate() {
            if let Some(m) = map {
                assert!(m < compact.len(), "column remapping out of range");
                // SAFETY: `orig_idx < d_orig == full.len()` by construction, and `m < compact.len()` asserted above.
                unsafe {
                    *full.get_unchecked_mut(orig_idx) = compact.get_unchecked(m).clone();
                }
            }
        }
        full
    }

    pub(crate) fn record_new_ray(&mut self, id: RayId) {
        self.core.pending_new_rays.push(id);
    }

    fn discard_pending_rays(&mut self, removed: &[RayId]) {
        if self.core.pending_new_rays.is_empty() || removed.is_empty() {
            return;
        }
        self.core.removed_epoch = self.core.removed_epoch.wrapping_add(1);
        if self.core.removed_epoch == 0 {
            self.core.removed_epoch = 1;
            self.core.removed_marks.fill(0);
        }

        let mut max_id = 0usize;
        for &rid in removed {
            max_id = max_id.max(rid.as_index());
        }
        for &rid in &self.core.pending_new_rays {
            max_id = max_id.max(rid.as_index());
        }
        if max_id >= self.core.removed_marks.len() {
            self.core.removed_marks.resize(max_id + 1, 0);
        }

        let epoch = self.core.removed_epoch;
        for &rid in removed {
            self.core.removed_marks[rid.as_index()] = epoch;
        }

        self.core
            .pending_new_rays
            .retain(|rid| self.core.removed_marks[rid.as_index()] != epoch);
    }

    pub(crate) fn ray(&self, index: RayId) -> Option<&U::RayData> {
        self.core.ray_graph.ray_data(index)
    }

    pub(crate) fn ray_mut(&mut self, index: RayId) -> Option<&mut U::RayData> {
        self.core.ray_graph.ray_data_mut(index)
    }

    pub(crate) fn with_active_ray_ids<T>(&mut self, f: impl FnOnce(&mut Self, &[RayId]) -> T) -> T {
        let mut ids = std::mem::take(&mut self.core.active_id_scratch);
        self.core.ray_graph.copy_active_ids(&mut ids);
        let result = f(self, &ids);
        self.core.active_id_scratch = ids;
        result
    }

    pub(crate) fn sync_tableau_flags(&mut self) {
        let m = self.row_count();
        if self.core.tableau.row_status.len() != m {
            self.core.tableau.row_status = vec![-1; m];
        } else {
            self.core.tableau.row_status.fill(-1);
        }
        for row in self.core.weakly_added_halfspaces.iter() {
            self.core.tableau.row_status[row.as_index()] = 0;
        }
        for row in self.core.added_halfspaces.iter() {
            self.core.tableau.row_status[row.as_index()] = 1;
        }
    }

    #[inline]
    pub(crate) fn sync_tableau_flag_for_row(&mut self, row: Row) {
        if row >= self.row_count() {
            return;
        }
        if self.core.tableau.row_status.len() != self.row_count() {
            self.sync_tableau_flags();
            return;
        }
        let flag = if self.core.added_halfspaces.contains(row) {
            1
        } else if self.core.weakly_added_halfspaces.contains(row) {
            0
        } else {
            -1
        };
        self.core.tableau.row_status[row] = flag;
    }

    pub(crate) fn enforce_first_infeasible_floor(&mut self, indices: &[RayId]) -> bool {
        let min_order = self.core.iter_state.iteration;
        if min_order >= self.core.ctx.order_vector.len() {
            return false;
        }
        let floor_row = self.core.ctx.order_vector[min_order];
        let mut changed = false;
        for &idx in indices {
            let should_floor = match self.ray(idx) {
                Some(ray) => {
                    if ray.last_sign() == Sign::Negative {
                        false
                    } else {
                        self.first_infeasible_position(ray)
                            .map(|pos| pos < min_order)
                            .unwrap_or(false)
                    }
                }
                None => false,
            };
            if should_floor && let Some(ray) = self.ray_mut(idx) {
                ray.set_first_infeasible_row(Some(floor_row));
                changed = true;
            }
        }
        changed
    }

    fn adjacency_dimension(&self) -> usize {
        self.col_count()
    }
}

impl<N: Num, R: Representation, U> ConeEngine<N, R, U, crate::dd::RowRepr>
where
    U: Umpire<N, crate::dd::RowRepr>,
{
    pub(crate) fn add_new_halfspace_preordered(&mut self, row: Row) -> Result<(), Error> {
        let iter_pos = self
            .order_position(row)
            .unwrap_or_else(|| panic!("row {row} not present in order vector"));
        self.core.iter_state.iteration = iter_pos;
        self.tableau_entering_for_row(row);

        let partition = self.evaluate_row_partition(row);
        let partition_view = Self::partition_view(&partition);

        if matches!(R::KIND, RepresentationKind::Inequality)
            && partition_view.negative.is_empty()
            && self.core.ctx.equality_kinds[row] != hullabaloo::types::InequalityKind::Equality
        {
            self.core.redundant_halfspaces.insert(row);
            self.core.weakly_added_halfspaces.insert(row);
            self.sync_iteration_with_added();
            self.sync_tableau_flag_for_row(row);
            if self.core.ray_graph.active_len() == self.core.ray_graph.weakly_feasible_len() {
                self.core.comp_status = ComputationStatus::AllFound;
            }
            self.recycle_partition(partition);
            return Ok(());
        }
        if partition_view.positive.is_empty() && partition_view.zero.is_empty() {
            self.core.ray_graph.deactivate_all();
            self.umpire.reset_infeasible_counts(self.row_count());
            self.clear_ray_indices();
            self.core.pending_new_rays.clear();
            self.core.comp_status = ComputationStatus::RegionEmpty;
            self.recycle_partition(partition);
            return Ok(());
        }

        let valid_first = Some(partition_view.positive);
        self.process_iteration_edges(iter_pos, self.core.ctx.order_vector[iter_pos], valid_first);
        self.delete_negative_rays(partition_view);
        self.prune_recent_rays();
        self.core.added_halfspaces.insert(row);
        self.core.weakly_added_halfspaces.insert(row);
        self.sync_iteration_with_added();
        self.sync_tableau_flag_for_row(row);

        if self.core.iter_state.iteration < self.row_count() && self.core.ray_graph.zero_len() > 1 {
            let zero_rays = std::mem::take(&mut self.core.active_id_scratch);
            if zero_rays.len() > 1 {
                self.update_edges(&zero_rays);
            }
            self.core.active_id_scratch = zero_rays;
        }

        <UmpireHalfspaceMode<N, U, crate::dd::RowRepr> as HalfspaceMode>::on_halfspace_added(
            self, row,
        );
        if self.core.ray_graph.active_len() == self.core.ray_graph.weakly_feasible_len() {
            self.core.comp_status = ComputationStatus::AllFound;
        }
        self.recycle_partition(partition);
        Ok(())
    }

    fn conditional_add_edge(&mut self, r1: RayId, r2: RayId, candidates: DegeneracyCandidates<'_>) {
        let f1 = match self.ray(r1) {
            Some(ray) => self.first_infeasible_position_or_m(ray),
            None => return,
        };
        let f2 = match self.ray(r2) {
            Some(ray) => self.first_infeasible_position_or_m(ray),
            None => return,
        };
        let (fmin, rmin, rmax) = if f1 <= f2 { (f1, r1, r2) } else { (f2, r2, r1) };
        if fmin >= self.core.ctx.order_vector.len() {
            return;
        }
        if self.ray(rmax).is_none() {
            return;
        }

        let max_zero = {
            let ray_graph = &self.core.ray_graph;
            let Some(ray) = ray_graph.ray_data(rmax) else {
                return;
            };
            ray.zero_set()
        };
        self.core.adj_face.copy_from(max_zero);

        let fmin_row = self
            .core
            .ctx
            .order_vector
            .get(fmin)
            .copied()
            .unwrap_or(self.row_count());
        let strong_sep = !self.core.adj_face.contains(fmin_row);
        if !strong_sep {
            return;
        }

        let min_zero = {
            let ray_graph = &self.core.ray_graph;
            let Some(ray) = ray_graph.ray_data(rmin) else {
                return;
            };
            ray.zero_set()
        };
        self.core.adj_face.intersection_inplace(min_zero);

        self.core.count_intersections += 1;
        let mut last_chance = true;
        for iteration in (self.core.iter_state.iteration + 1)..fmin {
            let row_idx = self.core.ctx.order_vector[iteration];
            let contains_row = self.core.adj_face.contains(row_idx);
            if contains_row && !self.core._strict_inequality_set.contains(row_idx) {
                last_chance = false;
                self.core.count_intersections_bad += 1;
                break;
            }
        }
        if last_chance {
            self.core.count_intersections_good += 1;
            let use_added = !self.core.added_halfspaces.is_empty();
            if use_added {
                self.core
                    .adj_face
                    .intersection_inplace(&self.core.added_halfspaces);
            } else {
                self.core.adj_face.intersection_inplace(&self.core.ground_set);
            }
            let required = self.adjacency_dimension().saturating_sub(2);
            let mut adjacent = self.core.adj_face.cardinality_at_least(required);
            if adjacent && !self.core.options.assumes_nondegeneracy() {
                let domain = crate::dd::RowRepr::domain_size(&self.core.ctx);
                let contains = match candidates {
                    DegeneracyCandidates::Slice(indices) => self.core.ray_incidence.candidate_contains_face(
                        &self.core.adj_face,
                        domain,
                        indices,
                        r1,
                        r2,
                    ),
                    DegeneracyCandidates::BitSet(set) => self.core.ray_incidence.candidate_set_contains_face(
                        &self.core.adj_face,
                        domain,
                        set,
                        r1,
                        r2,
                    ),
                };
                if contains {
                    adjacent = false;
                }
            }
            if adjacent {
                let edge = AdjacencyEdge {
                    retained: rmax,
                    removed: rmin,
                };
                self.core.ray_graph.queue_edge(fmin, edge);
            }
        }
    }

    pub(crate) fn create_initial_edges(&mut self) {
        self.core.iter_state.iteration = self.col_count();
        self.with_active_ray_ids(|state, rays| {
            if rays.len() < 2 {
                return;
            }
            for i in 0..rays.len() - 1 {
                let r1 = rays[i];
                let f1 = state
                    .ray(r1)
                    .map(|r| state.first_infeasible_position_or_m(r))
                    .unwrap_or(state.row_count());
                for &r2 in rays.iter().skip(i + 1) {
                    let f2 = state
                        .ray(r2)
                        .map(|r| state.first_infeasible_position_or_m(r))
                        .unwrap_or(state.row_count());
                    if f1 == f2 {
                        continue;
                    }
                    state.conditional_add_edge(r1, r2, DegeneracyCandidates::Slice(rays));
                }
            }
        });
    }

    pub(crate) fn update_edges(&mut self, zero_rays: &[RayId]) {
        if zero_rays.len() < 2 {
            return;
        }
        for i in 0..zero_rays.len() - 1 {
            let r1 = zero_rays[i];
            let fi = self
                .ray(r1)
                .map(|r| self.first_infeasible_position_or_m(r))
                .unwrap_or(self.row_count());
            for &r2 in zero_rays.iter().skip(i + 1) {
                let f2 = self
                    .ray(r2)
                    .map(|r| self.first_infeasible_position_or_m(r))
                    .unwrap_or(self.row_count());
                if f2 <= fi {
                    continue;
                }
                self.conditional_add_edge(r1, r2, DegeneracyCandidates::Slice(zero_rays));
            }
        }
    }

    fn process_iteration_edges(&mut self, iteration: Row, row: Row, valid_first: Option<&[RayId]>) {
        if iteration >= self.core.ray_graph.edge_buckets_len() {
            return;
        }
        if self.core.edges_dirty {
            self.refresh_edge_buckets();
            self.core.edges_dirty = false;
        }

        let valid_first = valid_first.unwrap_or(&[]);
        let edges = self.core.ray_graph.take_edges(iteration);

        if self.core.options.assumes_nondegeneracy() {
            for edge in edges {
                let f1 = match self.ray(edge.retained) {
                    Some(ray) => self.first_infeasible_position_or_m(ray),
                    None => continue,
                };

                let Some(new_idx) = self.create_new_ray(edge.removed, edge.retained, row) else {
                    continue;
                };

                let f2 = match self.ray(new_idx) {
                    Some(ray) => self.first_infeasible_position_or_m(ray),
                    None => continue,
                };

                if f1 != f2 {
                    self.conditional_add_edge(
                        edge.retained,
                        new_idx,
                        DegeneracyCandidates::Slice(valid_first),
                    );
                }
            }
            return;
        }

        let mut candidate_set = std::mem::replace(&mut self.core.candidate_ray_set, RowSet::new(0));
        candidate_set.resize(self.core.ray_incidence.ray_capacity());
        candidate_set.clear();
        for &rid in valid_first {
            candidate_set.insert(rid.as_index());
        }

        for edge in edges {
            let f1 = match self.ray(edge.retained) {
                Some(ray) => self.first_infeasible_position_or_m(ray),
                None => continue,
            };

            let Some(new_idx) = self.create_new_ray(edge.removed, edge.retained, row) else {
                continue;
            };

            let f2 = match self.ray(new_idx) {
                Some(ray) => self.first_infeasible_position_or_m(ray),
                None => continue,
            };

            if f1 != f2 {
                let need = self.core.ray_incidence.ray_capacity();
                if candidate_set.len() != need {
                    candidate_set.resize(need);
                }
                self.conditional_add_edge(
                    edge.retained,
                    new_idx,
                    DegeneracyCandidates::BitSet(&candidate_set),
                );
            }
        }
        self.core.candidate_ray_set = candidate_set;
    }
}

impl<N: Num, R: Representation, U, ZR> ConeEngine<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    pub fn feasibility_indices(&mut self, row: Row) -> (usize, usize) {
        let active = self.core.ray_graph.active_len();
        if row >= self.row_count() {
            return (active, 0);
        }

        let mut infeasible = 0usize;
        let mut ids = std::mem::take(&mut self.core.active_id_scratch);
        self.core.ray_graph.copy_active_ids(&mut ids);
        for id in ids.iter().copied() {
            let Some(ray_data) = self.core.ray_graph.ray_data(id) else {
                continue;
            };
            if self
                .umpire
                .sign_for_row_on_ray(&self.core.ctx, ray_data, row)
                == Sign::Negative
            {
                infeasible += 1;
            }
        }
        self.core.active_id_scratch = ids;

        assert!(active >= infeasible, "infeasible count exceeds active rays");
        (active - infeasible, infeasible)
    }
}
