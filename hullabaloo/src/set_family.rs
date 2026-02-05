use crate::types::{Big, RowId, RowSet};

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct SetFamily {
    set_capacity: Big,
    sets: Vec<RowSet>,
}

#[derive(Clone, Debug)]
pub struct SetFamilyBuilder {
    set_capacity: Big,
    sets: Vec<RowSet>,
}

impl SetFamily {
    pub fn new(family_size: Big, set_capacity: Big) -> Self {
        SetFamilyBuilder::new(family_size, set_capacity).build()
    }

    pub fn from_sets(set_capacity: Big, mut sets: Vec<RowSet>) -> Self {
        for set in &mut sets {
            if set.len() != set_capacity {
                set.resize(set_capacity);
            }
        }
        Self { set_capacity, sets }
    }

    pub fn builder(family_size: Big, set_capacity: Big) -> SetFamilyBuilder {
        SetFamilyBuilder::new(family_size, set_capacity)
    }

    pub fn into_builder(self) -> SetFamilyBuilder {
        SetFamilyBuilder::from_family(self)
    }

    pub fn family_size(&self) -> Big {
        self.sets.len()
    }

    pub fn set_capacity(&self) -> Big {
        self.set_capacity
    }

    pub fn sets(&self) -> &[RowSet] {
        &self.sets
    }

    pub fn set(&self, index: Big) -> Option<&RowSet> {
        self.sets.get(index)
    }

    pub fn is_empty(&self) -> bool {
        self.sets.iter().all(|s| s.is_empty())
    }

    pub fn transpose(&self) -> Self {
        let mut fam = SetFamily::builder(self.set_capacity, self.family_size());
        for (row_idx, set) in self.sets.iter().enumerate() {
            for elem in set.iter() {
                fam.insert_into_set(elem.as_index(), RowId::new(row_idx));
            }
        }
        fam.build()
    }

    pub fn classify_input_incidence(
        &self,
        linearity: &RowSet,
        output_rows: usize,
    ) -> (RowSet, RowSet) {
        let mut dominant = RowSet::new(self.family_size());
        let mut redundant = RowSet::new(self.family_size());
        let cardinalities: Vec<usize> = self.sets.iter().map(|set| set.cardinality()).collect();

        for (i, &card) in cardinalities.iter().enumerate() {
            if card == output_rows {
                dominant.insert(i);
            }
        }

        for (i, current) in self.sets.iter().enumerate().rev() {
            let current_card = cardinalities[i];
            if current.is_empty() {
                redundant.insert(i);
                continue;
            }
            for (k, (candidate, &candidate_card)) in
                self.sets.iter().zip(&cardinalities).enumerate()
            {
                if k == i || redundant.contains(k) || dominant.contains(k) {
                    continue;
                }
                if candidate_card < current_card {
                    continue;
                }
                if current.subset_of(candidate) {
                    redundant.insert(i);
                    break;
                }
            }
        }

        for idx in linearity.iter() {
            debug_assert!(
                idx.as_index() < self.family_size(),
                "linearity row out of range for input incidence"
            );
            redundant.insert(idx);
        }

        (redundant, dominant)
    }
}

impl SetFamilyBuilder {
    pub fn new(family_size: Big, set_capacity: Big) -> Self {
        Self {
            set_capacity,
            sets: vec![RowSet::new(set_capacity); family_size],
        }
    }

    pub fn from_family(family: SetFamily) -> Self {
        Self {
            set_capacity: family.set_capacity,
            sets: family.sets,
        }
    }

    pub fn family_size(&self) -> Big {
        self.sets.len()
    }

    pub fn set_capacity(&self) -> Big {
        self.set_capacity
    }

    pub fn resize(&mut self, family_size: Big, set_capacity: Big) {
        self.set_capacity = set_capacity;
        self.sets.resize(family_size, RowSet::new(set_capacity));
        for set in &mut self.sets {
            set.resize(set_capacity);
        }
    }

    pub fn clear_set(&mut self, set_idx: Big) {
        let set = self.set_mut(set_idx);
        set.clear();
    }

    pub fn replace_set(&mut self, set_idx: Big, set: RowSet) {
        self.assert_index(set_idx);
        let mut normalized = set;
        if normalized.len() != self.set_capacity {
            normalized.resize(self.set_capacity);
        }
        self.sets[set_idx] = normalized;
    }

    pub fn insert_into_set(&mut self, set_idx: Big, value: RowId) {
        let set = self.set_mut(set_idx);
        set.insert(value);
    }

    pub fn build(self) -> SetFamily {
        SetFamily {
            set_capacity: self.set_capacity,
            sets: self.sets,
        }
    }

    fn set_mut(&mut self, idx: Big) -> &mut RowSet {
        self.assert_index(idx);
        let set = self
            .sets
            .get_mut(idx)
            .expect("set index must already be validated");
        set.resize(self.set_capacity);
        set
    }

    fn assert_index(&self, idx: Big) {
        debug_assert!(idx < self.sets.len(), "set index out of bounds");
    }
}

/// A family of sorted adjacency lists over a fixed universe.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ListFamily {
    sets: Vec<Vec<usize>>,
    universe: usize,
}

impl ListFamily {
    #[inline]
    pub fn new(mut sets: Vec<Vec<usize>>, universe: usize) -> Self {
        for s in &mut sets {
            s.sort_unstable();
            s.dedup();
        }
        Self { sets, universe }
    }

    #[inline]
    pub fn from_sorted_sets(sets: Vec<Vec<usize>>, universe: usize) -> Self {
        debug_assert!(sets.iter().all(|s| s.windows(2).all(|w| w[0] < w[1])));
        debug_assert!(sets.iter().all(|s| s.iter().all(|&x| x < universe)));
        Self { sets, universe }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.sets.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    #[inline]
    pub fn universe_size(&self) -> usize {
        self.universe
    }

    #[inline]
    pub fn set(&self, idx: usize) -> Option<&[usize]> {
        self.sets.get(idx).map(Vec::as_slice)
    }

    #[inline]
    pub fn to_adjacency_lists(&self) -> Vec<Vec<usize>> {
        self.sets.clone()
    }

    #[inline]
    pub fn into_adjacency_lists(self) -> Vec<Vec<usize>> {
        self.sets
    }

    #[inline]
    pub fn sets(&self) -> &[Vec<usize>] {
        &self.sets
    }

    pub fn transpose(&self) -> Self {
        let mut degrees = vec![0usize; self.universe];
        for set in &self.sets {
            debug_assert!(set.windows(2).all(|w| w[0] < w[1]));
            for &in_idx in set {
                debug_assert!(in_idx < self.universe);
                degrees[in_idx] += 1;
            }
        }

        let mut input_to_output: Vec<Vec<usize>> =
            degrees.into_iter().map(Vec::with_capacity).collect();

        for (out_idx, set) in self.sets.iter().enumerate() {
            for &in_idx in set {
                input_to_output[in_idx].push(out_idx);
            }
        }

        ListFamily::from_sorted_sets(input_to_output, self.sets.len())
    }
}

impl From<ListFamily> for SetFamily {
    fn from(family: ListFamily) -> Self {
        let ListFamily { sets, universe } = family;
        let mut out: Vec<RowSet> = Vec::with_capacity(sets.len());
        for set in sets {
            out.push(RowSet::from_indices(universe, &set));
        }
        SetFamily::from_sets(universe, out)
    }
}

impl From<SetFamily> for ListFamily {
    fn from(family: SetFamily) -> Self {
        let SetFamily { set_capacity, sets } = family;
        let mut out: Vec<Vec<usize>> = Vec::with_capacity(sets.len());
        for set in sets {
            out.push(set.to_indices());
        }
        ListFamily::from_sorted_sets(out, set_capacity)
    }
}

#[cfg(test)]
mod tests {
    use super::SetFamily;
    use crate::types::RowId;
    use std::fmt::Write;

    fn format_cdd_set_family(family: &SetFamily) -> String {
        let mut out = String::new();
        out.push_str("begin\n");
        let _ = writeln!(
            out,
            " {:4} {:4}",
            family.family_size(),
            family.set_capacity()
        );
        for (idx, set) in family.sets().iter().enumerate() {
            let card = set.cardinality();
            assert!(
                family.set_capacity() >= card,
                "set capacity underflow while formatting set family"
            );
            let complement = family.set_capacity() - card;
            if complement >= card {
                let _ = write!(out, " {:4} {:4} : ", idx, card as isize);
                for elem in set.iter() {
                    let _ = write!(out, "{} ", elem.as_index());
                }
            } else {
                let _ = write!(out, " {:4} {:4} : ", idx, -(card as isize));
                for elem in set.iter().complement() {
                    let _ = write!(out, "{} ", elem.as_index());
                }
            }
            out.push('\n');
        }
        out.push_str("end\n");
        out
    }

    #[test]
    fn compressed_string_prefers_complement_when_smaller() {
        let mut builder = SetFamily::builder(2, 5);
        builder.insert_into_set(0, RowId::new(0));
        builder.insert_into_set(0, RowId::new(4));
        for idx in [0, 1, 2, 4] {
            builder.insert_into_set(1, RowId::new(idx));
        }
        let family = builder.build();

        let expected = "\
begin
    2    5
    0    2 : 0 4 
    1   -4 : 3 
end
";
        assert_eq!(format_cdd_set_family(&family), expected);
    }
}
