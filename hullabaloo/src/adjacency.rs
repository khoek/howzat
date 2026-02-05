use ahash::AHashMap;
use smallvec::SmallVec;

use crate::adjacency_list::AdjacencyList;
use crate::set_family::{SetFamily, SetFamilyBuilder};
use crate::types::{RowId, RowSet};

type Key = SmallVec<[usize; 16]>;

/// Read-only view of an undirected adjacency structure.
pub trait Graph {
    fn node_count(&self) -> usize;
    fn degree(&self, node: usize) -> usize;
    fn contains(&self, node: usize, neighbor: usize) -> bool;
    fn neighbors(&self, node: usize) -> impl Iterator<Item = usize> + '_;
}

/// Marker trait for adjacency structures that can be built by the hullabaloo adjacency engine.
pub trait AdjacencyStore: Graph + Sized {
    type Builder: AdjacencyBuilder<Output = Self>;
}

/// A sink for building an undirected adjacency structure.
///
/// This is generic so the core FR/adjacency algorithm can write into either an edge-list style
/// adjacency (`AdjacencyList`) or a bitset-based adjacency (e.g. `howzat::SetFamily`) without paying for
/// conversions.
pub trait AdjacencyBuilder {
    type Output;

    fn new(node_count: usize) -> Self;
    fn add_undirected_edge(&mut self, a: usize, b: usize);
    fn finish(self) -> Self::Output;
}

#[derive(Clone, Debug)]
pub struct AdjacencyListBuilder {
    adjacency: Vec<Vec<usize>>,
}

impl AdjacencyBuilder for AdjacencyListBuilder {
    type Output = AdjacencyList;

    #[inline]
    fn new(node_count: usize) -> Self {
        Self {
            adjacency: vec![Vec::new(); node_count],
        }
    }

    #[inline]
    fn add_undirected_edge(&mut self, a: usize, b: usize) {
        self.adjacency[a].push(b);
        self.adjacency[b].push(a);
    }

    #[inline]
    fn finish(mut self) -> Self::Output {
        for neigh in &mut self.adjacency {
            neigh.sort_unstable();
            neigh.dedup();
        }
        AdjacencyList::from_sorted_adjacency_lists(self.adjacency)
    }
}

impl Graph for AdjacencyList {
    #[inline]
    fn node_count(&self) -> usize {
        self.num_vertices()
    }

    #[inline]
    fn degree(&self, node: usize) -> usize {
        self.adjacency_lists()[node].len()
    }

    #[inline]
    fn contains(&self, node: usize, neighbor: usize) -> bool {
        self.adjacency_lists()[node]
            .binary_search(&neighbor)
            .is_ok()
    }

    #[inline]
    fn neighbors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjacency_lists()[node].iter().copied()
    }
}

impl AdjacencyStore for AdjacencyList {
    type Builder = AdjacencyListBuilder;
}

impl AdjacencyBuilder for SetFamilyBuilder {
    type Output = SetFamily;

    #[inline]
    fn new(node_count: usize) -> Self {
        SetFamily::builder(node_count, node_count)
    }

    #[inline]
    fn add_undirected_edge(&mut self, a: usize, b: usize) {
        self.insert_into_set(a, RowId::new(b));
        self.insert_into_set(b, RowId::new(a));
    }

    #[inline]
    fn finish(self) -> Self::Output {
        self.build()
    }
}

impl Graph for SetFamily {
    #[inline]
    fn node_count(&self) -> usize {
        self.family_size()
    }

    #[inline]
    fn degree(&self, node: usize) -> usize {
        self.sets()[node].cardinality()
    }

    #[inline]
    fn contains(&self, node: usize, neighbor: usize) -> bool {
        self.sets()[node].contains(neighbor)
    }

    #[inline]
    fn neighbors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.sets()[node].iter().raw()
    }
}

impl AdjacencyStore for SetFamily {
    type Builder = SetFamilyBuilder;
}

#[derive(Clone, Debug)]
struct SparseRowMembershipIndex {
    row_offsets: Vec<usize>,
    members: Vec<usize>,
}

impl SparseRowMembershipIndex {
    fn new<R: AsRef<[usize]>>(rows_by_node: &[R], row_capacity: usize) -> Self {
        let mut degrees: Vec<usize> = vec![0; row_capacity];
        for rows in rows_by_node {
            for &row in rows.as_ref() {
                if row < row_capacity {
                    degrees[row] = degrees[row].saturating_add(1);
                }
            }
        }

        let mut row_offsets: Vec<usize> = vec![0; row_capacity.saturating_add(1)];
        for row in 0..row_capacity {
            row_offsets[row + 1] = row_offsets[row].saturating_add(degrees[row]);
        }

        let mut members: Vec<usize> = vec![0; row_offsets[row_capacity]];
        let mut cursor = row_offsets.clone();
        for (node, rows) in rows_by_node.iter().enumerate() {
            for &row in rows.as_ref() {
                if row >= row_capacity {
                    continue;
                }
                let pos = cursor[row];
                if pos < members.len() {
                    members[pos] = node;
                    cursor[row] = pos + 1;
                }
            }
        }

        Self {
            row_offsets,
            members,
        }
    }

    #[inline]
    fn members_of_row(&self, row: usize) -> &[usize] {
        let Some((&start, &end)) = self
            .row_offsets
            .get(row)
            .zip(self.row_offsets.get(row.saturating_add(1)))
        else {
            return &[];
        };
        self.members.get(start..end).unwrap_or(&[])
    }

    #[inline]
    fn row_contains_node(&self, row: usize, node: usize) -> bool {
        self.members_of_row(row).binary_search(&node).is_ok()
    }
}

#[inline]
fn intersect_sorted_rows(a: &[usize], b: &[usize], out: &mut Vec<usize>) {
    out.clear();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
}

fn has_third_node_containing_face(
    membership: &SparseRowMembershipIndex,
    face_rows: &[usize],
    node_a: usize,
    node_b: usize,
    non_excluded_nodes: usize,
) -> bool {
    if face_rows.is_empty() {
        // The empty face is contained in every node; only adjacent when no third node exists.
        return non_excluded_nodes > 2;
    }

    let mut base_row = face_rows[0];
    let mut base_len = membership.members_of_row(base_row).len();
    for &row in face_rows.iter().skip(1) {
        let len = membership.members_of_row(row).len();
        if len < base_len {
            base_row = row;
            base_len = len;
            if base_len <= 2 {
                break;
            }
        }
    }

    for &candidate in membership.members_of_row(base_row) {
        if candidate == node_a || candidate == node_b {
            continue;
        }
        let mut ok = true;
        for &row in face_rows {
            if row == base_row {
                continue;
            }
            if !membership.row_contains_node(row, candidate) {
                ok = false;
                break;
            }
        }
        if ok {
            return true;
        }
    }
    false
}

/// Options for building an adjacency graph from incidence sets.
#[derive(Clone, Copy, Debug, Default)]
pub struct IncidenceAdjacencyOptions<'a> {
    /// Nodes to exclude from adjacency computation (e.g. linearity generators).
    ///
    /// When `None`, all nodes are considered active.
    pub excluded_nodes: Option<&'a [bool]>,

    /// Active universe rows; rows with `false` are ignored.
    ///
    /// When `None`, all rows are active.
    pub active_rows: Option<&'a [bool]>,

    /// Candidate edges to test (pairs of node indices). When `None`, the builder enumerates
    /// candidate edges using the sparse membership index.
    pub candidate_edges: Option<&'a [(usize, usize)]>,

    /// If `true`, skip degeneracy containment checks.
    pub assume_nondegenerate: bool,
}

/// Options for building adjacency from pre-filtered per-node incidence rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowsByNodeAdjacencyOptions<'a> {
    /// Nodes to exclude from adjacency computation (e.g. linearity generators).
    ///
    /// When `None`, all nodes are considered active.
    pub excluded_nodes: Option<&'a [bool]>,

    /// Candidate edges to test (pairs of node indices). When `None`, the builder enumerates
    /// candidate edges using the sparse membership index.
    pub candidate_edges: Option<&'a [(usize, usize)]>,

    /// If `true`, skip degeneracy containment checks.
    pub assume_nondegenerate: bool,
}

pub fn input_adjacency_from_incidence_set_family(
    incidence: &SetFamily,
    redundant_nodes: &RowSet,
    dominant_nodes: &RowSet,
) -> SetFamily {
    let family_size = incidence.family_size();
    if redundant_nodes.len() != family_size {
        panic!(
            "redundant_nodes length mismatch (redundant_nodes={} family_size={})",
            redundant_nodes.len(),
            family_size
        );
    }
    if dominant_nodes.len() != family_size {
        panic!(
            "dominant_nodes length mismatch (dominant_nodes={} family_size={})",
            dominant_nodes.len(),
            family_size
        );
    }

    let row_capacity = incidence.set_capacity();
    let mut rows_by_node: Vec<Vec<usize>> = Vec::with_capacity(family_size);

    for idx in 0..family_size {
        if redundant_nodes.contains(idx) || dominant_nodes.contains(idx) {
            rows_by_node.push(Vec::new());
            continue;
        }
        let set = incidence
            .set(idx)
            .unwrap_or_else(|| panic!("incidence SetFamily missing set for index {idx}"));
        rows_by_node.push(set.iter().map(|id| id.as_index()).collect());
    }

    input_adjacency_from_rows_by_node_with::<SetFamilyBuilder>(
        &rows_by_node,
        row_capacity,
        redundant_nodes,
        dominant_nodes,
    )
}

pub fn input_adjacency_from_rows_by_node_with<S: AdjacencyBuilder>(
    rows_by_node: &[impl AsRef<[usize]>],
    row_capacity: usize,
    redundant_nodes: &RowSet,
    dominant_nodes: &RowSet,
) -> S::Output {
    let family_size = rows_by_node.len();
    if family_size == 0 {
        return S::new(0).finish();
    }
    if redundant_nodes.len() != family_size {
        panic!(
            "redundant_nodes length mismatch (redundant_nodes={} family_size={})",
            redundant_nodes.len(),
            family_size
        );
    }
    if dominant_nodes.len() != family_size {
        panic!(
            "dominant_nodes length mismatch (dominant_nodes={} family_size={})",
            dominant_nodes.len(),
            family_size
        );
    }

    let mut excluded_nodes = vec![false; family_size];

    let mut has_dominant = false;
    let mut normal_first = usize::MAX;
    let mut normal_second = usize::MAX;
    let mut normal_count = 0usize;

    for (idx, excluded_slot) in excluded_nodes.iter_mut().enumerate() {
        let is_dominant = dominant_nodes.contains(idx);
        has_dominant |= is_dominant;
        let excluded = redundant_nodes.contains(idx) || is_dominant;
        *excluded_slot = excluded;
        if !excluded {
            normal_count += 1;
            if normal_count == 1 {
                normal_first = idx;
            } else if normal_count == 2 {
                normal_second = idx;
            }
        }
    }

    let mut sink = S::new(family_size);
    adjacency_from_rows_by_node_into(
        &mut sink,
        rows_by_node,
        row_capacity,
        3,
        RowsByNodeAdjacencyOptions {
            excluded_nodes: Some(&excluded_nodes),
            candidate_edges: None,
            assume_nondegenerate: false,
        },
    );

    if normal_count == 2 {
        sink.add_undirected_edge(normal_first, normal_second);
    }

    if has_dominant {
        let mut nonredundant: Vec<usize> = Vec::new();
        for idx in 0..family_size {
            if !redundant_nodes.contains(idx) {
                nonredundant.push(idx);
            }
        }

        for dom in dominant_nodes.iter().map(|id| id.as_index()) {
            if redundant_nodes.contains(dom) {
                continue;
            }
            for &other in &nonredundant {
                if other == dom {
                    continue;
                }
                sink.add_undirected_edge(dom, other);
            }
        }
    }

    sink.finish()
}

/// Build an undirected adjacency graph from a family of incidence sets.
///
/// The criterion matches the "ridge adjacency" / "dual graph" logic:
/// - Let `face = S[i] ∩ S[j]` (restricted to `active_rows`).
/// - Require `|face| >= adj_dim - 2`.
/// - In degenerate mode, reject `(i,j)` if there exists any third node `k` whose set contains `face`.
///
/// `adj_dim` should be the dimension of the homogenized cone / system (typically `d+1` for polytopes).
///
/// # Panics
///
/// Panics if dimension mismatches occur in `excluded_nodes`, `active_rows`, or row indices.
pub fn adjacency_from_incidence(
    sets: &[Vec<usize>],
    universe_size: usize,
    adj_dim: usize,
    options: IncidenceAdjacencyOptions<'_>,
) -> AdjacencyList {
    adjacency_from_incidence_with::<AdjacencyListBuilder>(sets, universe_size, adj_dim, options)
}

/// Build adjacency from incidence sets, writing into an arbitrary sink.
///
/// # Panics
///
/// Panics if dimension mismatches occur in `excluded_nodes`, `active_rows`, or row indices.
pub fn adjacency_from_incidence_with<S: AdjacencyBuilder>(
    sets: &[Vec<usize>],
    universe_size: usize,
    adj_dim: usize,
    options: IncidenceAdjacencyOptions<'_>,
) -> S::Output {
    let family_size = sets.len();
    if family_size == 0 {
        return S::new(0).finish();
    }

    let excluded = options.excluded_nodes;
    if let Some(mask) = excluded {
        assert!(
            mask.len() == family_size,
            "excluded_nodes length mismatch (excluded_nodes={} family_size={})",
            mask.len(),
            family_size
        );
    }
    let is_excluded = |idx: usize| excluded.is_some_and(|m| m[idx]);

    let active = options.active_rows;
    if let Some(mask) = active {
        assert!(
            mask.len() == universe_size,
            "active_rows length mismatch (active_rows={} universe_size={})",
            mask.len(),
            universe_size
        );
    }

    let mut rows_by_node: Vec<Vec<usize>> = Vec::with_capacity(family_size);
    for (idx, rows) in sets.iter().enumerate() {
        if is_excluded(idx) {
            rows_by_node.push(Vec::new());
            continue;
        }

        let mut filtered: Vec<usize> = Vec::new();
        if let Some(active) = active {
            for &row in rows {
                debug_assert!(
                    row < universe_size,
                    "row index out of range (node={idx} row={row} universe_size={universe_size})"
                );
                if active[row] {
                    filtered.push(row);
                }
            }
        } else {
            for &row in rows {
                debug_assert!(
                    row < universe_size,
                    "row index out of range (node={idx} row={row} universe_size={universe_size})"
                );
                filtered.push(row);
            }
        }

        debug_assert!(
            filtered.windows(2).all(|w| w[0] < w[1]),
            "adjacency_from_incidence expects per-node sets to be sorted + deduped"
        );
        rows_by_node.push(filtered);
    }

    adjacency_from_rows_by_node_with::<S>(
        &rows_by_node,
        universe_size,
        adj_dim,
        RowsByNodeAdjacencyOptions {
            excluded_nodes: excluded,
            candidate_edges: options.candidate_edges,
            assume_nondegenerate: options.assume_nondegenerate,
        },
    )
}

/// Build adjacency from pre-filtered per-node incidence rows.
///
/// `rows_by_node[i]` must be sorted + deduped, and contain only row indices `< row_capacity`.
///
/// # Panics
///
/// Panics if `excluded_nodes` length doesn't match family size.
pub fn adjacency_from_rows_by_node_with<S: AdjacencyBuilder>(
    rows_by_node: &[impl AsRef<[usize]>],
    row_capacity: usize,
    adj_dim: usize,
    options: RowsByNodeAdjacencyOptions<'_>,
) -> S::Output {
    let family_size = rows_by_node.len();
    if family_size == 0 {
        return S::new(0).finish();
    }

    let mut sink = S::new(family_size);
    adjacency_from_rows_by_node_into(&mut sink, rows_by_node, row_capacity, adj_dim, options);
    sink.finish()
}

fn adjacency_from_rows_by_node_into<S, Rows>(
    sink: &mut S,
    rows_by_node: &[Rows],
    row_capacity: usize,
    adj_dim: usize,
    options: RowsByNodeAdjacencyOptions<'_>,
) where
    S: AdjacencyBuilder,
    Rows: AsRef<[usize]>,
{
    let family_size = rows_by_node.len();
    if family_size == 0 {
        return;
    }

    let excluded = options.excluded_nodes;
    if let Some(mask) = excluded {
        assert!(
            mask.len() == family_size,
            "excluded_nodes length mismatch (excluded_nodes={} family_size={})",
            mask.len(),
            family_size
        );
    }
    let is_excluded = |idx: usize| excluded.is_some_and(|m| m[idx]);

    let required = adj_dim.saturating_sub(2);
    let non_excluded = (0..family_size).filter(|&i| !is_excluded(i)).count();
    if non_excluded < 2 {
        return;
    }

    // Edge case: when `required == 0` (adj_dim <= 2), overlap-based candidate generation can't see
    // empty intersections. Handle directly.
    if options.candidate_edges.is_none() && required == 0 {
        let non_excluded_indices: Vec<usize> =
            (0..family_size).filter(|&i| !is_excluded(i)).collect();
        if options.assume_nondegenerate || non_excluded_indices.len() == 2 {
            for (pos, &i) in non_excluded_indices.iter().enumerate() {
                for &j in non_excluded_indices.iter().skip(pos + 1) {
                    sink.add_undirected_edge(i, j);
                }
            }
        }
        return;
    }

    // If nodes are simplicial (`|rows| == required + 1`), adjacency reduces to hashing their
    // (required)-subfaces ("ridges"). This avoids the overlap sweep whose cost can blow up with
    // high row-membership (e.g. vertex degrees in facet adjacency).
    //
    // In mixed cases, we still use ridge hashing to add all simplicial↔simplicial edges and then
    // run the generic test only for non-simplicial nodes.
    let mut simplicial_facet_dimension: Option<usize> = None;
    if options.candidate_edges.is_none() && required >= 1 {
        let facet_dimension = required + 1;
        let simplicial_count = simplicial_ridge_hash_into_sink(
            rows_by_node,
            facet_dimension,
            excluded,
            options.assume_nondegenerate,
            sink,
        );
        if simplicial_count == non_excluded {
            return;
        }
        simplicial_facet_dimension = Some(facet_dimension);
    }

    let membership = SparseRowMembershipIndex::new(rows_by_node, row_capacity);
    let mut face_rows: Vec<usize> = Vec::new();

    if let Some(edges) = options.candidate_edges {
        for &(i, j) in edges {
            if i == j || i >= family_size || j >= family_size {
                continue;
            }
            if is_excluded(i) || is_excluded(j) {
                continue;
            }
            intersect_sorted_rows(
                rows_by_node[i].as_ref(),
                rows_by_node[j].as_ref(),
                &mut face_rows,
            );
            if face_rows.len() < required {
                continue;
            }
            if !options.assume_nondegenerate
                && has_third_node_containing_face(&membership, &face_rows, i, j, non_excluded)
            {
                continue;
            }
            sink.add_undirected_edge(i, j);
        }
        return;
    }

    let mut counts: Vec<u32> = vec![0; family_size];
    let mut touched: Vec<usize> = Vec::new();

    for i in 0..family_size {
        if is_excluded(i) {
            continue;
        }
        if let Some(d) = simplicial_facet_dimension {
            if rows_by_node[i].as_ref().len() == d {
                continue;
            }
        }

        touched.clear();
        for &row in rows_by_node[i].as_ref() {
            debug_assert!(row < row_capacity, "row index out of range in rows_by_node");
            for &j in membership.members_of_row(row) {
                let c = counts[j];
                if c == 0 {
                    touched.push(j);
                }
                counts[j] = c + 1;
            }
        }

        for &j in &touched {
            if is_excluded(j) {
                continue;
            }
            if let Some(d) = simplicial_facet_dimension {
                if rows_by_node[j].as_ref().len() != d && j <= i {
                    continue;
                }
            } else if j <= i {
                continue;
            }
            if (counts[j] as usize) < required {
                continue;
            }
            if !options.assume_nondegenerate {
                intersect_sorted_rows(
                    rows_by_node[i].as_ref(),
                    rows_by_node[j].as_ref(),
                    &mut face_rows,
                );
                if face_rows.len() < required {
                    continue;
                }
                if has_third_node_containing_face(&membership, &face_rows, i, j, non_excluded) {
                    continue;
                }
            }
            sink.add_undirected_edge(i, j);
        }

        for &j in &touched {
            counts[j] = 0;
        }
    }
}

fn simplicial_ridge_hash_into_sink<S: AdjacencyBuilder>(
    facets: &[impl AsRef<[usize]>],
    facet_dimension: usize,
    excluded_nodes: Option<&[bool]>,
    assume_nondegenerate: bool,
    sink: &mut S,
) -> usize {
    if facet_dimension < 2 {
        return 0;
    }

    let is_excluded = |idx: usize| excluded_nodes.is_some_and(|m| m[idx]);

    let mut ridge_to_facets: AHashMap<Key, SmallVec<[usize; 2]>> = AHashMap::new();
    let mut simplicial_count: usize = 0;
    let ridge_len = facet_dimension - 1;
    for (facet_idx, facet) in facets.iter().enumerate() {
        if is_excluded(facet_idx) {
            continue;
        }
        let facet = facet.as_ref();
        if facet.len() != facet_dimension {
            continue;
        }
        simplicial_count += 1;
        for drop_pos in 0..facet_dimension {
            let mut ridge = Key::with_capacity(ridge_len);
            ridge.extend(facet[..drop_pos].iter().copied());
            ridge.extend(facet[(drop_pos + 1)..].iter().copied());
            ridge_to_facets.entry(ridge).or_default().push(facet_idx);
        }
    }

    if assume_nondegenerate {
        for incident in ridge_to_facets.values() {
            for (pos, &a) in incident.iter().enumerate() {
                for &b in incident.iter().skip(pos + 1) {
                    if a != b {
                        sink.add_undirected_edge(a, b);
                    }
                }
            }
        }
    } else {
        for incident in ridge_to_facets.values() {
            if let [a, b] = incident.as_slice() {
                if a != b {
                    sink.add_undirected_edge(*a, *b);
                }
            }
        }
    }

    simplicial_count
}

#[cfg(test)]
mod tests {
    use super::{IncidenceAdjacencyOptions, adjacency_from_incidence};

    #[test]
    fn square_facets_form_cycle() {
        // Unit square facets each contain two vertices (0..4).
        // Facet 0: {0,1}, facet 1: {1,3}, facet 2: {2,3}, facet 3: {0,2}
        let facets = vec![vec![0, 1], vec![1, 3], vec![2, 3], vec![0, 2]];
        let g = adjacency_from_incidence(&facets, 4, 3, IncidenceAdjacencyOptions::default());
        let degrees: Vec<usize> = g.adjacency_lists().iter().map(|n| n.len()).collect();
        assert_eq!(degrees, vec![2, 2, 2, 2]);
    }
}
