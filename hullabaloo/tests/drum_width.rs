use std::f64::consts::PI;

use hullabaloo::{AdjacencyList, Drum, Geometrizable};
use rustworkx_core::petgraph::{algo::dijkstra, graph::UnGraph};

#[path = "support/backend.rs"]
mod backend;
mod fixtures;

#[test]
fn graph_distances_match_rustworkx_core() {
    let graph = AdjacencyList::from_sorted_adjacency_lists(vec![
        vec![1, 3],
        vec![0, 2],
        vec![1, 3],
        vec![0, 2],
    ]);

    let expected = petgraph_distance_table(&graph);
    assert_eq!(expected.len(), graph.num_vertices());
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(row.len(), graph.num_vertices());
        for (j, &expected_dist) in row.iter().enumerate() {
            assert_ne!(expected_dist, usize::MAX);
            assert_eq!(graph.distance(i, j).unwrap(), expected_dist);
        }
    }

    let diameter = expected
        .iter()
        .flat_map(|row| row.iter().copied())
        .filter(|d| *d != usize::MAX)
        .max()
        .unwrap_or(0);
    assert_eq!(graph.diameter().unwrap(), diameter);
}

#[test]
fn aligned_polygon_drums_have_width_two() {
    for sides in 3usize..=6 {
        let vertices = regular_polygon(1.0, sides, 0.0);
        let top_count = vertices.len();
        let bot_count = vertices.len();
        let drum = Drum::<f64>::new(vertices.clone(), vertices);
        let width = backend::drum_width(drum, top_count, bot_count).expect("drum width");
        assert_eq!(width, 2, "expected width two for {sides}-gon prism");
    }
}

#[test]
fn rotated_triangle_drum_has_width_three() {
    let top = regular_polygon(2.0, 3, 0.0);
    let bot = regular_polygon(1.0, 3, PI / 3.0);
    let drum = Drum::<f64>::new(top.clone(), bot.clone());

    let width = backend::drum_width(drum, top.len(), bot.len()).expect("drum width");
    assert_eq!(width, 3);
}

#[test]
fn santos_prismatoid_width_is_six() {
    let (top, bot) = fixtures::santos_bases_as_vecs();
    let drum = Drum::<f64>::new(top.clone(), bot.clone());

    let vertices = drum.clone().into_vertices();
    assert_eq!(vertices.len(), top.len() + bot.len());
    let width = backend::drum_width_from_vertices(&vertices, top.len(), bot.len()).expect("width");
    assert_eq!(width, 6);
}

#[test]
fn williamson_family_prismatoid_has_width_nine() {
    let (top, bot) = fixtures::williamson_k4_bases_as_vecs();
    let drum = Drum::<f64>::new(top.clone(), bot.clone());

    let width = backend::drum_width(drum, top.len(), bot.len()).expect("drum width");
    assert_eq!(width, 9);
}

#[test]
fn williamson_k1_prismatoid_has_width_six() {
    let (top, bot) = fixtures::williamson_k1_bases_as_vecs();
    let drum = Drum::<f64>::new(top.clone(), bot.clone());

    let width = backend::drum_width(drum, top.len(), bot.len()).expect("drum width");
    assert_eq!(width, 6);
}

fn regular_polygon(radius: f64, n: usize, phase: f64) -> Vec<Vec<f64>> {
    assert!(n >= 3);
    (0..n)
        .map(|i| {
            let angle = phase + 2.0 * PI * (i as f64) / (n as f64);
            vec![radius * angle.cos(), radius * angle.sin()]
        })
        .collect()
}

fn petgraph_distance_table(graph: &AdjacencyList) -> Vec<Vec<usize>> {
    let mut pg: UnGraph<(), ()> = UnGraph::default();
    let nodes: Vec<_> = (0..graph.num_vertices()).map(|_| pg.add_node(())).collect();

    for (u, neighbors) in graph.adjacency_lists().iter().enumerate() {
        for &v in neighbors {
            if u < v {
                pg.add_edge(nodes[u], nodes[v], ());
            }
        }
    }

    let n = graph.num_vertices();
    let mut dist = vec![vec![usize::MAX; n]; n];
    for start in 0..n {
        dist[start][start] = 0;
        let map = dijkstra(&pg, nodes[start], None, |_| 1usize);
        for (node, cost) in map {
            dist[start][node.index()] = cost;
        }
    }
    dist
}
