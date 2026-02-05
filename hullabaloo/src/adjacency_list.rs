use std::collections::VecDeque;

use crate::set_family::ListFamily;

/// Simple undirected graph represented by adjacency lists.
///
/// Vertices are 0-based indices.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct AdjacencyList {
    inner: ListFamily,
}

impl AdjacencyList {
    #[inline]
    pub fn empty() -> Self {
        Self {
            inner: ListFamily::from_sorted_sets(Vec::new(), 0),
        }
    }

    pub fn from_unsorted_adjacency_lists(adjacency: Vec<Vec<usize>>) -> Self {
        let universe = adjacency.len();
        Self {
            inner: ListFamily::new(adjacency, universe),
        }
    }

    pub fn from_sorted_adjacency_lists(adjacency: Vec<Vec<usize>>) -> Self {
        let universe = adjacency.len();
        Self {
            inner: ListFamily::from_sorted_sets(adjacency, universe),
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.inner.len()
    }

    pub fn degree(&self, v: usize) -> usize {
        self.inner.sets()[v].len()
    }

    pub fn neighbors(&self, v: usize) -> &[usize] {
        self.inner
            .set(v)
            .expect("AdjacencyList vertex index out of range")
    }

    pub fn contains(&self, v: usize, neighbor: usize) -> bool {
        self.neighbors(v).binary_search(&neighbor).is_ok()
    }

    pub fn adjacency_lists(&self) -> &[Vec<usize>] {
        self.inner.sets()
    }

    #[inline]
    pub fn into_adjacency_lists(self) -> Vec<Vec<usize>> {
        self.inner.into_adjacency_lists()
    }

    /// Returns the shortest path distance between two vertices, or None if disconnected.
    ///
    /// # Panics
    ///
    /// Panics if `start` or `goal` are out of bounds.
    pub fn distance(&self, start: usize, goal: usize) -> Option<usize> {
        assert!(
            start < self.num_vertices() && goal < self.num_vertices(),
            "graph indices out of range (start={start}, goal={goal}, size={})",
            self.num_vertices()
        );
        if start == goal {
            return Some(0);
        }

        let dist = self.bfs(start, Some(goal));
        match dist[goal] {
            usize::MAX => None,
            goal_dist => Some(goal_dist),
        }
    }

    /// Returns the diameter of the graph, or None if the graph is disconnected.
    pub fn diameter(&self) -> Option<usize> {
        let n = self.num_vertices();
        if n == 0 {
            return Some(0);
        }

        let mut diameter = 0usize;
        for start in 0..n {
            let dist = self.bfs_from(start)?;
            if let Some(max_for_start) = dist.into_iter().max() {
                diameter = diameter.max(max_for_start);
            }
        }
        Some(diameter)
    }

    fn bfs_from(&self, start: usize) -> Option<Vec<usize>> {
        let dist = self.bfs(start, None);
        if dist.contains(&usize::MAX) {
            return None;
        }
        Some(dist)
    }

    fn bfs(&self, start: usize, stop_at: Option<usize>) -> Vec<usize> {
        debug_assert!(
            start < self.num_vertices(),
            "graph start index out of range (start={start}, size={})",
            self.num_vertices()
        );

        let vertex_count = self.num_vertices();
        let mut dist = vec![usize::MAX; vertex_count];
        let mut queue = VecDeque::new();
        dist[start] = 0;
        queue.push_back(start);

        while let Some(v) = queue.pop_front() {
            let next_dist = dist[v] + 1;
            for &n in self.neighbors(v) {
                debug_assert!(
                    n < vertex_count,
                    "graph adjacency references {n} but size is {vertex_count}"
                );
                if dist[n] != usize::MAX {
                    continue;
                }
                dist[n] = next_dist;
                if Some(n) == stop_at {
                    return dist;
                }
                queue.push_back(n);
            }
        }

        dist
    }
}

#[cfg(test)]
mod tests {
    use super::AdjacencyList;

    #[test]
    fn distances_work_on_cycle() {
        let graph = AdjacencyList::from_sorted_adjacency_lists(vec![
            vec![1, 3],
            vec![0, 2],
            vec![1, 3],
            vec![0, 2],
        ]);

        assert_eq!(graph.distance(0, 0), Some(0));
        assert_eq!(graph.distance(0, 1), Some(1));
        assert_eq!(graph.distance(0, 2), Some(2));
        assert_eq!(graph.distance(0, 3), Some(1));
        assert_eq!(graph.diameter(), Some(2));
    }

    #[test]
    fn disconnected_graph_returns_none() {
        let graph =
            AdjacencyList::from_sorted_adjacency_lists(vec![vec![1], vec![0], vec![3], vec![2]]);

        assert_eq!(graph.distance(0, 2), None);
        assert_eq!(graph.diameter(), None);
    }
}
