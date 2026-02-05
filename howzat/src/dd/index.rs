use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use crate::dd::Umpire;
use crate::dd::ray::{RayData as _, RayId};
use crate::dd::sat::SatSet;
use crate::dd::state::ConeEngine;
use crate::dd::zero::ZeroSet;
use crate::dd::zero::ZeroRepr;
use calculo::num::Num;
use hullabaloo::types::{Representation, RowSet};

#[derive(Clone, Copy, Debug, Default)]
struct BuildIdentityHasher;

#[derive(Clone, Copy, Debug, Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), 8, "IdentityHasher expects u64 keys");
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        self.0 = u64::from_ne_bytes(arr);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

impl BuildHasher for BuildIdentityHasher {
    type Hasher = IdentityHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher(0)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RayZeroSetIndex {
    map: HashMap<u64, Vec<RayId>, BuildIdentityHasher>,
    bucket_pool: Vec<Vec<RayId>>,
}

impl RayZeroSetIndex {
    pub(crate) fn clear(&mut self) {
        for (_sig, mut bucket) in self.map.drain() {
            bucket.clear();
            self.bucket_pool.push(bucket);
        }
    }

    pub(crate) fn register(&mut self, id: RayId, sig: u64) {
        let bucket = self.map.entry(sig).or_insert_with(|| {
            if let Some(mut v) = self.bucket_pool.pop() {
                v.clear();
                v
            } else {
                Vec::new()
            }
        });
        bucket.push(id);
    }

    pub(crate) fn unregister(&mut self, id: RayId, sig: u64) {
        let Some(bucket) = self.map.get_mut(&sig) else {
            return;
        };
        if let Some(pos) = bucket.iter().position(|&x| x == id) {
            bucket.swap_remove(pos);
        }
        if bucket.is_empty()
            && let Some(mut empty) = self.map.remove(&sig)
        {
            empty.clear();
            self.bucket_pool.push(empty);
        }
    }

    #[inline]
    pub(crate) fn candidates<'a>(&'a self, sig: u64) -> &'a [RayId] {
        self.map.get(&sig).map_or(&[], Vec::as_slice)
    }
}

/// Dynamic sat-row→ray incidence index.
///
/// For each saturation-row `r` (in the prefix-sized sat index space), stores the set of active
/// rays whose zero-set contains `r`.
#[derive(Clone, Debug)]
pub struct SatRayIncidenceIndex {
    /// Active rays (bitset over ray slot indices).
    active: RowSet,
    /// For each sat-row index, the set of active rays incident to that row.
    by_row: Vec<RowSet>,
    /// Scratch set used for intersections.
    scratch: RowSet,
    ray_capacity: usize,
}

impl Default for SatRayIncidenceIndex {
    fn default() -> Self {
        Self {
            active: RowSet::new(0),
            by_row: Vec::new(),
            scratch: RowSet::new(0),
            ray_capacity: 0,
        }
    }
}

impl SatRayIncidenceIndex {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn ray_capacity(&self) -> usize {
        self.ray_capacity
    }

    #[inline]
    fn ensure_rows(&mut self, row_count: usize) {
        if self.by_row.len() >= row_count {
            return;
        }
        while self.by_row.len() < row_count {
            self.by_row.push(RowSet::new(self.ray_capacity));
        }
    }

    #[inline]
    fn ensure_ray_capacity(&mut self, needed: usize) {
        if needed <= self.ray_capacity {
            return;
        }
        let mut new_cap = self.ray_capacity.max(64);
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(2);
        }
        self.ray_capacity = new_cap;

        self.active.resize(new_cap);
        self.scratch.resize(new_cap);
        for set in &mut self.by_row {
            set.resize(new_cap);
        }
    }

    pub fn clear(&mut self) {
        self.active.clear();
        self.scratch.clear();
        for set in &mut self.by_row {
            set.clear();
        }
    }

    pub fn register(&mut self, id: RayId, zero_set: &SatSet, sat_row_count: usize) {
        let slot = id.as_index();
        self.ensure_rows(sat_row_count);
        self.ensure_ray_capacity(slot + 1);

        self.active.insert(slot);
        for row_id in zero_set.iter() {
            let row = row_id.as_index();
            let set = self
                .by_row
                .get_mut(row)
                .unwrap_or_else(|| panic!("sat row index {} out of bounds", row));
            set.insert(slot);
        }
    }

    pub fn unregister(&mut self, id: RayId, zero_set: &SatSet, sat_row_count: usize) {
        let slot = id.as_index();
        self.ensure_rows(sat_row_count);
        self.ensure_ray_capacity(slot + 1);

        self.active.remove(slot);
        for row_id in zero_set.iter() {
            let row = row_id.as_index();
            let set = self
                .by_row
                .get_mut(row)
                .unwrap_or_else(|| panic!("sat row index {} out of bounds", row));
            set.remove(slot);
        }
    }

    /// Returns `true` iff there exists a ray in `candidates` (excluding `exclude_a`/`exclude_b`)
    /// whose zero set contains all rows in `face`.
    pub fn candidate_contains_face(
        &mut self,
        face: &SatSet,
        sat_row_count: usize,
        candidates: &[RayId],
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        self.ensure_rows(sat_row_count);

        self.scratch.copy_from(&self.active);
        for row_id in face.iter() {
            let row = row_id.as_index();
            let Some(incidence) = self.by_row.get(row) else {
                self.scratch.clear();
                return false;
            };
            self.scratch.intersection_inplace(incidence);
            if self.scratch.is_empty() {
                return false;
            }
        }

        for &id in candidates {
            if id == exclude_a || id == exclude_b {
                continue;
            }
            if self.scratch.contains(id.as_index()) {
                return true;
            }
        }
        false
    }

    /// Returns `true` iff there exists a ray in `candidate_set` (excluding `exclude_a`/`exclude_b`)
    /// whose zero set contains all rows in `face`.
    ///
    /// `candidate_set` is a bitset over ray slot indices with the **same** dimension as this
    /// index's ray sets.
    pub fn candidate_set_contains_face(
        &mut self,
        face: &SatSet,
        sat_row_count: usize,
        candidate_set: &RowSet,
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        self.ensure_rows(sat_row_count);
        debug_assert_eq!(
            candidate_set.len(),
            self.ray_capacity,
            "candidate set dimension mismatch (expected {}, got {})",
            self.ray_capacity,
            candidate_set.len()
        );

        self.scratch.copy_from(candidate_set);
        if self.scratch.is_empty() {
            return false;
        }
        for row_id in face.iter() {
            let row = row_id.as_index();
            let Some(incidence) = self.by_row.get(row) else {
                self.scratch.clear();
                return false;
            };
            self.scratch.intersection_inplace(incidence);
            if self.scratch.is_empty() {
                return false;
            }
        }

        let a = exclude_a.as_index();
        if a < self.scratch.len() {
            self.scratch.remove(a);
        }
        let b = exclude_b.as_index();
        if b < self.scratch.len() {
            self.scratch.remove(b);
        }
        !self.scratch.is_empty()
    }
}

/// Dynamic row→ray incidence index for full-dimensional `RowSet` zero sets.
#[derive(Clone, Debug)]
pub struct RowRayIncidenceIndex {
    active: RowSet,
    by_row: Vec<RowSet>,
    scratch: RowSet,
    ray_capacity: usize,
}

impl Default for RowRayIncidenceIndex {
    fn default() -> Self {
        Self {
            active: RowSet::new(0),
            by_row: Vec::new(),
            scratch: RowSet::new(0),
            ray_capacity: 0,
        }
    }
}

impl RowRayIncidenceIndex {
    #[inline]
    pub(crate) fn ray_capacity(&self) -> usize {
        self.ray_capacity
    }

    #[inline]
    fn ensure_rows(&mut self, row_count: usize) {
        if self.by_row.len() == row_count {
            return;
        }
        self.by_row = (0..row_count)
            .map(|_| RowSet::new(self.ray_capacity))
            .collect();
    }

    #[inline]
    fn ensure_ray_capacity(&mut self, needed: usize) {
        if needed <= self.ray_capacity {
            return;
        }
        let mut new_cap = self.ray_capacity.max(64);
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(2);
        }
        self.ray_capacity = new_cap;

        self.active.resize(new_cap);
        self.scratch.resize(new_cap);
        for set in &mut self.by_row {
            set.resize(new_cap);
        }
    }

    pub fn clear(&mut self) {
        self.active.clear();
        self.scratch.clear();
        for set in &mut self.by_row {
            set.clear();
        }
    }

    pub fn register(&mut self, id: RayId, zero_set: &RowSet, row_count: usize) {
        debug_assert_eq!(
            row_count,
            zero_set.len(),
            "row count mismatch for row-incidence register"
        );
        let slot = id.as_index();
        self.ensure_rows(row_count);
        self.ensure_ray_capacity(slot + 1);

        self.active.insert(slot);
        for row_id in zero_set.iter() {
            let row = row_id.as_index();
            if let Some(set) = self.by_row.get_mut(row) {
                set.insert(slot);
            }
        }
    }

    pub fn unregister(&mut self, id: RayId, zero_set: &RowSet, row_count: usize) {
        debug_assert_eq!(
            row_count,
            zero_set.len(),
            "row count mismatch for row-incidence unregister"
        );
        let slot = id.as_index();
        self.ensure_rows(row_count);
        self.ensure_ray_capacity(slot + 1);

        self.active.remove(slot);
        for row_id in zero_set.iter() {
            let row = row_id.as_index();
            if let Some(set) = self.by_row.get_mut(row) {
                set.remove(slot);
            }
        }
    }

    pub fn candidate_contains_face(
        &mut self,
        face: &RowSet,
        row_count: usize,
        candidates: &[RayId],
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        self.ensure_rows(row_count);

        self.scratch.copy_from(&self.active);
        for row_id in face.iter() {
            let row = row_id.as_index();
            let Some(incidence) = self.by_row.get(row) else {
                self.scratch.clear();
                return false;
            };
            self.scratch.intersection_inplace(incidence);
            if self.scratch.is_empty() {
                return false;
            }
        }

        for &id in candidates {
            if id == exclude_a || id == exclude_b {
                continue;
            }
            if self.scratch.contains(id.as_index()) {
                return true;
            }
        }
        false
    }

    pub fn candidate_set_contains_face(
        &mut self,
        face: &RowSet,
        row_count: usize,
        candidate_set: &RowSet,
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        self.ensure_rows(row_count);
        debug_assert_eq!(
            candidate_set.len(),
            self.ray_capacity,
            "candidate set dimension mismatch (expected {}, got {})",
            self.ray_capacity,
            candidate_set.len()
        );

        self.scratch.copy_from(candidate_set);
        if self.scratch.is_empty() {
            return false;
        }
        for row_id in face.iter() {
            let row = row_id.as_index();
            let Some(incidence) = self.by_row.get(row) else {
                self.scratch.clear();
                return false;
            };
            self.scratch.intersection_inplace(incidence);
            if self.scratch.is_empty() {
                return false;
            }
        }

        let a = exclude_a.as_index();
        if a < self.scratch.len() {
            self.scratch.remove(a);
        }
        let b = exclude_b.as_index();
        if b < self.scratch.len() {
            self.scratch.remove(b);
        }
        !self.scratch.is_empty()
    }
}

#[doc(hidden)]
pub trait RayIncidenceIndex<ZS: ZeroSet>: Clone + std::fmt::Debug + Default {
    #[allow(dead_code)]
    fn ray_capacity(&self) -> usize;
    fn clear(&mut self);
    fn register(&mut self, id: RayId, zero_set: &ZS, domain_size: usize);
    fn unregister(&mut self, id: RayId, zero_set: &ZS, domain_size: usize);

    fn candidate_contains_face(
        &mut self,
        face: &ZS,
        domain_size: usize,
        candidates: &[RayId],
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool;

    #[allow(dead_code)]
    fn candidate_set_contains_face(
        &mut self,
        face: &ZS,
        domain_size: usize,
        candidate_set: &RowSet,
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool;
}

impl RayIncidenceIndex<RowSet> for RowRayIncidenceIndex {
    #[inline(always)]
    fn ray_capacity(&self) -> usize {
        RowRayIncidenceIndex::ray_capacity(self)
    }

    #[inline(always)]
    fn clear(&mut self) {
        RowRayIncidenceIndex::clear(self);
    }

    #[inline(always)]
    fn register(&mut self, id: RayId, zero_set: &RowSet, domain_size: usize) {
        RowRayIncidenceIndex::register(self, id, zero_set, domain_size);
    }

    #[inline(always)]
    fn unregister(&mut self, id: RayId, zero_set: &RowSet, domain_size: usize) {
        RowRayIncidenceIndex::unregister(self, id, zero_set, domain_size);
    }

    #[inline(always)]
    fn candidate_contains_face(
        &mut self,
        face: &RowSet,
        domain_size: usize,
        candidates: &[RayId],
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        RowRayIncidenceIndex::candidate_contains_face(
            self, face, domain_size, candidates, exclude_a, exclude_b,
        )
    }

    #[inline(always)]
    fn candidate_set_contains_face(
        &mut self,
        face: &RowSet,
        domain_size: usize,
        candidate_set: &RowSet,
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        RowRayIncidenceIndex::candidate_set_contains_face(
            self, face, domain_size, candidate_set, exclude_a, exclude_b,
        )
    }
}

impl RayIncidenceIndex<SatSet> for SatRayIncidenceIndex {
    #[inline(always)]
    fn ray_capacity(&self) -> usize {
        SatRayIncidenceIndex::ray_capacity(self)
    }

    #[inline(always)]
    fn clear(&mut self) {
        SatRayIncidenceIndex::clear(self);
    }

    #[inline(always)]
    fn register(&mut self, id: RayId, zero_set: &SatSet, domain_size: usize) {
        SatRayIncidenceIndex::register(self, id, zero_set, domain_size);
    }

    #[inline(always)]
    fn unregister(&mut self, id: RayId, zero_set: &SatSet, domain_size: usize) {
        SatRayIncidenceIndex::unregister(self, id, zero_set, domain_size);
    }

    #[inline(always)]
    fn candidate_contains_face(
        &mut self,
        face: &SatSet,
        domain_size: usize,
        candidates: &[RayId],
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        SatRayIncidenceIndex::candidate_contains_face(
            self, face, domain_size, candidates, exclude_a, exclude_b,
        )
    }

    #[inline(always)]
    fn candidate_set_contains_face(
        &mut self,
        face: &SatSet,
        domain_size: usize,
        candidate_set: &RowSet,
        exclude_a: RayId,
        exclude_b: RayId,
    ) -> bool {
        SatRayIncidenceIndex::candidate_set_contains_face(
            self, face, domain_size, candidate_set, exclude_a, exclude_b,
        )
    }
}

impl<N: Num, R: Representation, U, ZR> ConeEngine<N, R, U, ZR>
where
    U: Umpire<N, ZR>,
    ZR: ZeroRepr,
{
    pub(crate) fn clear_ray_indices(&mut self) {
        self.core.ray_index.clear();
        if !self.core.options.assumes_nondegeneracy() {
            self.core.ray_incidence.clear();
        }
    }

    pub(crate) fn register_ray_id(&mut self, id: RayId) {
        let use_incidence = !self.core.options.assumes_nondegeneracy();
        let (index, incidence, graph) = (
            &mut self.core.ray_index,
            &mut self.core.ray_incidence,
            &self.core.ray_graph,
        );
        if let Some(ray) = graph.ray_data(id) {
            index.register(id, ray.zero_set_signature());
            if use_incidence {
                let domain = ZR::domain_size(&self.core.ctx);
                incidence.register(id, ray.zero_set(), domain);
            }
        }
    }

    pub(crate) fn unregister_ray_id(&mut self, id: RayId) {
        let use_incidence = !self.core.options.assumes_nondegeneracy();
        let (index, incidence, graph) = (
            &mut self.core.ray_index,
            &mut self.core.ray_incidence,
            &self.core.ray_graph,
        );
        if let Some(ray) = graph.ray_data(id) {
            index.unregister(id, ray.zero_set_signature());
            if use_incidence {
                let domain = ZR::domain_size(&self.core.ctx);
                incidence.unregister(id, ray.zero_set(), domain);
            }
        }
    }

    pub(crate) fn rebuild_ray_index(&mut self) {
        self.clear_ray_indices();
        self.with_active_ray_ids(|state, ids| {
            for id in ids.iter().copied() {
                state.register_ray_id(id);
            }
        });
    }

    pub(crate) fn ray_exists(&mut self, ray_data: &U::RayData) -> bool {
        let candidates = self.core.ray_index.candidates(ray_data.zero_set_signature());
        if candidates.is_empty() {
            return false;
        }

        for &id in candidates {
            let Some(existing_ray_data) = self.core.ray_graph.ray_data(id) else {
                continue;
            };
            if existing_ray_data.zero_set() != ray_data.zero_set() {
                continue;
            }
            if self.core.options.assumes_nondegeneracy() {
                return true;
            }
            if self.umpire.rays_equivalent(existing_ray_data, ray_data) {
                return true;
            }
        }
        false
    }
}
