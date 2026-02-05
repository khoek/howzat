use crate::dd::sat::{SatRowId, SatSet};
use crate::dd::umpire::{ConeCtx, UmpireMatrix};
use calculo::num::Num;
use hullabaloo::types::{Representation, Row, RowId, RowSet, SetIter};

#[doc(hidden)]
pub trait ZeroSet: Clone + std::fmt::Debug + Eq {
    type Id: Copy + std::fmt::Debug;

    type Iter<'a>: Iterator<Item = Self::Id>
    where
        Self: 'a;

    fn ensure_domain(&mut self, domain: Row);

    fn is_empty(&self) -> bool;
    fn clear(&mut self);
    fn copy_from(&mut self, other: &Self);
    fn contains(&self, id: Self::Id) -> bool;
    fn insert(&mut self, id: Self::Id);

    fn intersection_inplace(&mut self, other: &Self);

    #[inline(always)]
    fn intersection_inplace_and_count(&mut self, other: &Self) -> usize {
        self.intersection_inplace(other);
        self.cardinality()
    }

    fn cardinality(&self) -> usize;
    fn cardinality_at_least(&self, target: usize) -> bool;
    fn signature_u64(&self) -> u64;

    fn iter(&self) -> Self::Iter<'_>;
}

#[inline(always)]
fn row_id_to_row(id: RowId) -> Row {
    id.as_index()
}

impl ZeroSet for RowSet {
    type Id = Row;

    type Iter<'a> = std::iter::Map<SetIter<'a, RowId>, fn(RowId) -> Row> where Self: 'a;

    #[inline(always)]
    fn ensure_domain(&mut self, domain: Row) {
        RowSet::resize(self, domain);
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        RowSet::is_empty(self)
    }

    #[inline(always)]
    fn clear(&mut self) {
        RowSet::clear(self);
    }

    #[inline(always)]
    fn copy_from(&mut self, other: &Self) {
        RowSet::copy_from(self, other);
    }

    #[inline(always)]
    fn contains(&self, id: Self::Id) -> bool {
        RowSet::contains(self, id)
    }

    #[inline(always)]
    fn insert(&mut self, id: Self::Id) {
        RowSet::insert(self, id);
    }

    #[inline(always)]
    fn intersection_inplace(&mut self, other: &Self) {
        RowSet::intersection_inplace(self, other);
    }

    #[inline(always)]
    fn cardinality(&self) -> usize {
        RowSet::cardinality(self)
    }

    #[inline(always)]
    fn cardinality_at_least(&self, target: usize) -> bool {
        RowSet::cardinality_at_least(self, target)
    }

    #[inline(always)]
    fn signature_u64(&self) -> u64 {
        RowSet::hash_signature_u64(self)
    }

    #[inline(always)]
    fn iter(&self) -> Self::Iter<'_> {
        RowSet::iter(self).map(row_id_to_row as fn(RowId) -> Row)
    }
}

impl ZeroSet for SatSet {
    type Id = SatRowId;

    type Iter<'a> = crate::dd::sat::SatSetIter<'a> where Self: 'a;

    #[inline(always)]
    fn ensure_domain(&mut self, _domain: Row) {}

    #[inline(always)]
    fn is_empty(&self) -> bool {
        SatSet::is_empty(self)
    }

    #[inline(always)]
    fn clear(&mut self) {
        SatSet::clear(self);
    }

    #[inline(always)]
    fn copy_from(&mut self, other: &Self) {
        SatSet::copy_from(self, other);
    }

    #[inline(always)]
    fn contains(&self, id: Self::Id) -> bool {
        SatSet::contains(self, id)
    }

    #[inline(always)]
    fn insert(&mut self, id: Self::Id) {
        SatSet::insert(self, id);
    }

    #[inline(always)]
    fn intersection_inplace(&mut self, other: &Self) {
        SatSet::intersection_inplace(self, other);
    }

    #[inline(always)]
    fn intersection_inplace_and_count(&mut self, other: &Self) -> usize {
        SatSet::intersection_inplace_and_count(self, other)
    }

    #[inline(always)]
    fn cardinality(&self) -> usize {
        SatSet::cardinality(self)
    }

    #[inline(always)]
    fn cardinality_at_least(&self, target: usize) -> bool {
        SatSet::cardinality_at_least(self, target)
    }

    #[inline(always)]
    fn signature_u64(&self) -> u64 {
        SatSet::signature_u64(self)
    }

    #[inline(always)]
    fn iter(&self) -> Self::Iter<'_> {
        SatSet::iter(self)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RowRepr;

#[derive(Clone, Copy, Debug, Default)]
pub struct SatRepr;

#[doc(hidden)]
pub trait ZeroRepr: Clone + std::fmt::Debug + Default + 'static {
    type Set: ZeroSet;
    type IncidenceIndex: RayIncidenceIndex<Self::Set>;

    fn empty_set(row_count: Row) -> Self::Set;

    fn fill_purify_expected_zero<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        expected_zero: &mut RowSet,
        row: Row,
        ray1_zero_set: &Self::Set,
        ray2_zero_set: &Self::Set,
    );

    fn domain_size<N: Num, R: Representation, M: UmpireMatrix<N, R>>(cone: &ConeCtx<N, R, M>)
        -> usize;

    fn id_for_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        row: Row,
    ) -> Option<<Self::Set as ZeroSet>::Id>;

    fn ensure_id_for_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &mut ConeCtx<N, R, M>,
        row: Row,
    ) -> (<Self::Set as ZeroSet>::Id, bool);

    fn row_for_id<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        id: <Self::Set as ZeroSet>::Id,
    ) -> Row;

    fn zero_set_contains_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        zero_set: &Self::Set,
        row: Row,
    ) -> bool;
}

impl ZeroRepr for RowRepr {
    type Set = RowSet;
    type IncidenceIndex = RowRayIncidenceIndex;

    #[inline(always)]
    fn empty_set(row_count: Row) -> Self::Set {
        RowSet::new(row_count)
    }

    #[inline(always)]
    fn fill_purify_expected_zero<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        expected_zero: &mut RowSet,
        row: Row,
        ray1_zero_set: &Self::Set,
        ray2_zero_set: &Self::Set,
    ) {
        expected_zero.resize(cone.matrix().row_count());
        expected_zero.copy_from(ray1_zero_set);
        expected_zero.intersection_inplace(ray2_zero_set);
        expected_zero.insert(row);
    }

    #[inline(always)]
    fn domain_size<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
    ) -> usize {
        cone.matrix().row_count()
    }

    #[inline(always)]
    fn id_for_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        row: Row,
    ) -> Option<<Self::Set as ZeroSet>::Id> {
        cone.sat_id_for_row(row).is_some().then_some(row)
    }

    #[inline(always)]
    fn ensure_id_for_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &mut ConeCtx<N, R, M>,
        row: Row,
    ) -> (<Self::Set as ZeroSet>::Id, bool) {
        if cone.sat_row_to_id.len() != cone.matrix().row_count() {
            cone.sat_row_to_id.resize(cone.matrix().row_count(), None);
        }
        let inserted = cone.sat_id_for_row(row).is_none();
        if inserted {
            cone.sat_row_to_id[row] = Some(SatRowId::from(row));
        }
        (row, inserted)
    }

    #[inline(always)]
    fn row_for_id<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        _cone: &ConeCtx<N, R, M>,
        id: <Self::Set as ZeroSet>::Id,
    ) -> Row {
        id
    }

    #[inline(always)]
    fn zero_set_contains_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        zero_set: &Self::Set,
        row: Row,
    ) -> bool {
        cone.sat_id_for_row(row).is_some() && zero_set.contains(row)
    }
}

impl ZeroRepr for SatRepr {
    type Set = SatSet;
    type IncidenceIndex = SatRayIncidenceIndex;

    #[inline(always)]
    fn empty_set(_row_count: Row) -> Self::Set {
        SatSet::default()
    }

    #[inline(always)]
    fn fill_purify_expected_zero<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        expected_zero: &mut RowSet,
        row: Row,
        ray1_zero_set: &Self::Set,
        ray2_zero_set: &Self::Set,
    ) {
        expected_zero.resize(cone.matrix().row_count());
        expected_zero.clear();
        expected_zero.insert(row);
        for id in ray1_zero_set.iter() {
            if ray2_zero_set.contains(id) {
                expected_zero.insert(cone.sat_row_for_id(id));
            }
        }
    }

    #[inline(always)]
    fn domain_size<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
    ) -> usize {
        cone.sat_id_to_row.len()
    }

    #[inline(always)]
    fn id_for_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        row: Row,
    ) -> Option<<Self::Set as ZeroSet>::Id> {
        cone.sat_id_for_row(row)
    }

    #[inline(always)]
    fn ensure_id_for_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &mut ConeCtx<N, R, M>,
        row: Row,
    ) -> (<Self::Set as ZeroSet>::Id, bool) {
        let had = cone.sat_id_for_row(row).is_some();
        let id = cone.assign_sat_id_for_row(row);
        (id, !had)
    }

    #[inline(always)]
    fn row_for_id<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        id: <Self::Set as ZeroSet>::Id,
    ) -> Row {
        cone.sat_row_for_id(id)
    }

    #[inline(always)]
    fn zero_set_contains_row<N: Num, R: Representation, M: UmpireMatrix<N, R>>(
        cone: &ConeCtx<N, R, M>,
        zero_set: &Self::Set,
        row: Row,
    ) -> bool {
        cone.sat_id_for_row(row).is_some_and(|id| zero_set.contains(id))
    }
}
use crate::dd::index::{RayIncidenceIndex, RowRayIncidenceIndex, SatRayIncidenceIndex};
