use calculo::num::{F64Em12Epsilon, InexactNum, Num, Rat, RugRat};
use howzat::dd::ConeOptions;
use howzat::dd::DefaultNormalizer;
use howzat::matrix::{LpMatrix, LpMatrixBuilder};
use howzat::polyhedron::{PolyhedronOptions, PolyhedronOutput};
use howzat::verify::{
    FacetGraphRepairOptions, PartialResolveIssue, ResolveError, ResolveOptions,
    SimplicialFrontierRepairOptions, simplicial_frontier_ridge_count, simplicial_frontier_ridges,
};
use hullabaloo::types::{AdjacencyOutput, Generator, IncidenceOutput, RowSet};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

#[derive(Clone, Debug)]
struct Stats {
    dimension: usize,
    vertices: usize,
    facets: usize,
    ridges: usize,
}

#[derive(Clone, Debug)]
struct FrGraph {
    stats: Stats,
    facet_keys: Vec<Vec<usize>>,
    ridge_keys: BTreeSet<(Vec<usize>, Vec<usize>)>,
}

fn build_generator_matrix<N: Num>(vertices: &[Vec<f64>]) -> LpMatrix<N, Generator> {
    let rows: Vec<Vec<N>> = vertices
        .iter()
        .map(|coords| {
            let mut row = Vec::with_capacity(coords.len() + 1);
            row.push(N::one());
            for &value in coords {
                let value = N::try_from_f64(value).expect("vertex coordinates must be finite");
                row.push(value);
            }
            row
        })
        .collect();
    LpMatrixBuilder::<N, Generator>::from_rows(rows).build()
}

fn build_poly_dd_inexact<N: InexactNum + DefaultNormalizer>(
    vertices: &[Vec<f64>],
    poly_options: PolyhedronOptions,
) -> PolyhedronOutput<N, Generator> {
    let eps = N::default_eps();
    let matrix = build_generator_matrix::<N>(vertices);
    PolyhedronOutput::<N, Generator>::builder(matrix)
        .cone_options(ConeOptions::default())
        .polyhedron_options(poly_options)
        .run_dd_with_eps(eps)
        .expect("DD conversion must succeed")
}

fn build_poly_dd_int<N: Rat>(
    vertices: &[Vec<f64>],
    poly_options: PolyhedronOptions,
) -> PolyhedronOutput<N, Generator> {
    let matrix = build_generator_matrix::<N>(vertices);
    PolyhedronOutput::<N, Generator>::builder(matrix)
        .cone_options(ConeOptions::default())
        .polyhedron_options(poly_options)
        .run_dd_int()
        .expect("DD conversion must succeed")
}

fn rowset_to_vec(set: &RowSet) -> Vec<usize> {
    set.iter().map(|v| v.as_index()).collect()
}

fn build_fr_graph<N: Num>(poly: &PolyhedronOutput<N, Generator>) -> FrGraph {
    assert!(
        !poly.is_empty(),
        "poly must be non-empty (status={:?} output_rows={})",
        poly.status(),
        poly.output().row_count()
    );
    let incidence = poly.incidence().expect("output incidence must be present");
    let adjacency = poly.adjacency().expect("output adjacency must be present");

    let output_rows = poly.output().row_count();
    assert_eq!(
        incidence.family_size(),
        output_rows,
        "incidence size mismatch"
    );

    let input_rows = poly.input().row_count();
    let output_linearity = poly.output().linearity();

    let mut facet_keys: Vec<Vec<usize>> = Vec::new();
    let mut key_to_new: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut old_to_new: Vec<Option<usize>> = vec![None; output_rows];

    for (old_idx, face) in incidence.sets().iter().enumerate() {
        if output_linearity.contains(old_idx) {
            continue;
        }
        let mut key = rowset_to_vec(face);
        key.retain(|v| *v < input_rows);

        if let Some(&existing) = key_to_new.get(&key) {
            old_to_new[old_idx] = Some(existing);
            continue;
        }
        let new_idx = facet_keys.len();
        facet_keys.push(key.clone());
        key_to_new.insert(key, new_idx);
        old_to_new[old_idx] = Some(new_idx);
    }

    let mut edges_idx: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (old_i, neighbors) in adjacency.sets().iter().enumerate() {
        let Some(new_i) = old_to_new.get(old_i).copied().flatten() else {
            continue;
        };
        for old_j in neighbors.iter().map(|j| j.as_index()) {
            let Some(new_j) = old_to_new.get(old_j).copied().flatten() else {
                continue;
            };
            if new_i == new_j {
                continue;
            }
            let (a, b) = if new_i < new_j {
                (new_i, new_j)
            } else {
                (new_j, new_i)
            };
            edges_idx.insert((a, b));
        }
    }

    let mut ridge_keys = BTreeSet::new();
    for (a, b) in edges_idx {
        ridge_keys.insert(ordered_edge(&facet_keys[a], &facet_keys[b]));
    }

    let vertices = match (poly.redundant_rows(), poly.dominant_rows()) {
        (Some(redundant), Some(dominant)) => {
            input_rows.saturating_sub(redundant.cardinality() + dominant.cardinality())
        }
        _ => input_rows,
    };

    FrGraph {
        stats: Stats {
            dimension: poly.dimension(),
            vertices,
            facets: facet_keys.len(),
            ridges: ridge_keys.len(),
        },
        facet_keys,
        ridge_keys,
    }
}

fn format_stats(stats: &Stats) -> String {
    format!(
        "dim={} vertices={} facets={} ridges={}",
        stats.dimension, stats.vertices, stats.facets, stats.ridges
    )
}

fn diff_counts<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> (usize, usize) {
    let missing = left.difference(right).count();
    let extra = right.difference(left).count();
    (missing, extra)
}

fn edge_key_set_from_simplicial_facets(
    facet_keys: &[Vec<usize>],
    facet_dimension: usize,
) -> BTreeSet<(Vec<usize>, Vec<usize>)> {
    let mut ridge_to_facet: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut edges = BTreeSet::new();

    for (facet_idx, facet) in facet_keys.iter().enumerate() {
        if facet.len() != facet_dimension {
            continue;
        }
        for drop_pos in 0..facet.len() {
            let mut ridge = facet.clone();
            ridge.remove(drop_pos);
            if let Some(other_idx) = ridge_to_facet.insert(ridge, facet_idx) {
                let Some(left) = facet_keys.get(other_idx) else {
                    continue;
                };
                edges.insert(ordered_edge(left, facet));
            }
        }
    }

    edges
}

fn ordered_edge(left: &[usize], right: &[usize]) -> (Vec<usize>, Vec<usize>) {
    if left <= right {
        (left.to_vec(), right.to_vec())
    } else {
        (right.to_vec(), left.to_vec())
    }
}

#[test]
fn frcheck_seed_16863925862282503163_resolution_f64_epsilon_m12_too_small() {
    let vertices: Vec<Vec<f64>> = vec![
        vec![
            0.562799995446813,
            1.7203446315845348,
            3.95635665210175,
            -9.924109773517035,
            -3.8693900181985397,
            8.468594452088066,
            1.0,
        ],
        vec![
            2.831944030767474,
            -6.418037462603343,
            1.8526630368432642,
            6.899517776210885,
            -5.603365327340457,
            -8.51727071787769,
            1.0,
        ],
        vec![
            -9.990787340655448,
            -9.45601705961888,
            -4.326741872839572,
            3.7861677364779833,
            7.943158016588484,
            0.9571469879308392,
            1.0,
        ],
        vec![
            -0.5991298506615941,
            -0.6233180965461713,
            -2.3710782844477096,
            3.7337972820952423,
            9.531136423377603,
            -2.3649761386190216,
            1.0,
        ],
        vec![
            -9.979706076445494,
            4.415828896595766,
            7.753945376164637,
            -7.312434616467902,
            -2.5498165795944328,
            -3.455582182265613,
            1.0,
        ],
        vec![
            -4.96498950270885,
            1.8284303903558108,
            1.1686623330241908,
            3.265493410740387,
            -5.843787718981264,
            -7.581825647428797,
            1.0,
        ],
        vec![
            0.7846863490719613,
            1.422817178074185,
            9.756256782941332,
            -1.7129780730095554,
            8.18292903505484,
            -6.781354476693777,
            1.0,
        ],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ];

    let poly_options = PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::List,
        input_incidence: IncidenceOutput::Set,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let exact_start = Instant::now();
    let exact = build_poly_dd_int::<RugRat>(&vertices, poly_options.clone());
    let exact_time = exact_start.elapsed();

    let non_exact_start = Instant::now();
    let non_exact = {
        let matrix = build_generator_matrix::<f64>(&vertices);
        PolyhedronOutput::<f64, Generator>::builder(matrix)
            .cone_options(ConeOptions::default())
            .polyhedron_options(poly_options.clone())
            .run_dd_with_eps(F64Em12Epsilon)
            .expect("DD conversion must succeed")
    };
    let non_exact_time = non_exact_start.elapsed();

    println!(
        "exact poly: status={:?} empty={} out_rows={} inc={} adj={}",
        exact.status(),
        exact.is_empty(),
        exact.output().row_count(),
        exact.incidence().is_some(),
        exact.adjacency().is_some()
    );
    println!(
        "non-exact poly: status={:?} empty={} out_rows={} inc={} adj={}",
        non_exact.status(),
        non_exact.is_empty(),
        non_exact.output().row_count(),
        non_exact.incidence().is_some(),
        non_exact.adjacency().is_some()
    );
    println!("timing: rugrat_dd={exact_time:?} f64_dd={non_exact_time:?}",);

    let exact_graph = build_fr_graph(&exact);
    let non_graph = build_fr_graph(&non_exact);

    let exact_facets: BTreeSet<Vec<usize>> = exact_graph.facet_keys.iter().cloned().collect();
    let non_facets: BTreeSet<Vec<usize>> = non_graph.facet_keys.iter().cloned().collect();
    let (missing_facets, extra_facets) = diff_counts(&exact_facets, &non_facets);
    let (missing_ridges, extra_ridges) =
        diff_counts(&exact_graph.ridge_keys, &non_graph.ridge_keys);

    assert!(
        missing_facets > 0 || extra_facets > 0 || missing_ridges > 0 || extra_ridges > 0,
        "expected FR graph mismatch between f64 and RugRat"
    );

    println!("exact: {}", format_stats(&exact_graph.stats));
    println!("non-exact: {}", format_stats(&non_graph.stats));
    println!(
        "exact vs non-exact: missing_facets={missing_facets} extra_facets={extra_facets} \
         missing_ridges={missing_ridges} extra_ridges={extra_ridges}"
    );

    let cert = non_exact
        .certificate()
        .expect("non-exact output must carry resolve certificate");
    let exact_eps = RugRat::default_eps();
    let resolved =
        cert.resolve_as::<RugRat>(poly_options.clone(), ResolveOptions::default(), &exact_eps);

    match resolved {
        Ok(poly) => {
            let resolved_graph = build_fr_graph(&poly);
            let resolved_facets: BTreeSet<Vec<usize>> =
                resolved_graph.facet_keys.iter().cloned().collect();
            let (missing_vs_exact, extra_vs_exact) = diff_counts(&exact_facets, &resolved_facets);
            let (missing_ridges_vs_exact, extra_ridges_vs_exact) =
                diff_counts(&exact_graph.ridge_keys, &resolved_graph.ridge_keys);

            println!("resolved: {}", format_stats(&resolved_graph.stats));
            println!(
                "exact vs resolved: missing_facets={missing_vs_exact} extra_facets={extra_vs_exact} \
                 missing_ridges={missing_ridges_vs_exact} extra_ridges={extra_ridges_vs_exact}"
            );

            let (missing_vs_non, extra_vs_non) = diff_counts(&non_facets, &resolved_facets);
            let (missing_ridges_vs_non, extra_ridges_vs_non) =
                diff_counts(&non_graph.ridge_keys, &resolved_graph.ridge_keys);
            println!(
                "non-exact vs resolved: missing_facets={missing_vs_non} extra_facets={extra_vs_non} \
                 missing_ridges={missing_ridges_vs_non} extra_ridges={extra_ridges_vs_non}"
            );
        }
        Err(err) => {
            println!("resolution failed: {err:?}");
            if let ResolveError::WitnessNotOneDim { output_row } = err {
                let incidence = non_exact
                    .incidence()
                    .expect("output incidence must be present");
                let Some(face) = incidence.set(output_row) else {
                    panic!("missing incidence set for output_row={output_row}");
                };
                let input_rows = non_exact.input().row_count();
                let mut key = rowset_to_vec(face);
                key.retain(|v| *v < input_rows);
                println!(
                    "witness not 1D: output_row={output_row} vertices={} key={:?}",
                    key.len(),
                    key
                );
            }
        }
    }

    let partial_start = Instant::now();
    let partial = cert
        .resolve_partial_as::<RugRat>(poly_options, ResolveOptions::default(), &exact_eps)
        .expect("partial resolution must not fail");
    let partial_time = partial_start.elapsed();
    let partial_poly = partial.polyhedron();
    let partial_graph = build_fr_graph(partial_poly);
    let partial_facets: BTreeSet<Vec<usize>> = partial_graph.facet_keys.iter().cloned().collect();

    let (missing_vs_exact, extra_vs_exact) = diff_counts(&exact_facets, &partial_facets);
    let (missing_ridges_vs_exact, extra_ridges_vs_exact) =
        diff_counts(&exact_graph.ridge_keys, &partial_graph.ridge_keys);

    println!("partial-resolved: {}", format_stats(&partial_graph.stats));
    println!(
        "exact vs partial-resolved: missing_facets={missing_vs_exact} extra_facets={extra_vs_exact} \
         missing_ridges={missing_ridges_vs_exact} extra_ridges={extra_ridges_vs_exact}"
    );

    let mut witness_not_one_dim = 0usize;
    let mut infeasible = 0usize;
    for issue in partial.issues() {
        match issue {
            PartialResolveIssue::WitnessNotOneDim { .. } => witness_not_one_dim += 1,
            PartialResolveIssue::InfeasibleResolvedRow { .. } => infeasible += 1,
        }
    }
    println!(
        "partial issues: total={} witness_not_1d={} infeasible={}",
        partial.issues().len(),
        witness_not_one_dim,
        infeasible
    );

    let mut frontier =
        simplicial_frontier_ridges(&partial_graph.facet_keys, partial_graph.stats.dimension);
    frontier.sort_unstable_by(|a, b| a.ridge().cmp(b.ridge()));
    println!("partial frontier ridges: {}", frontier.len());
    for (idx, ridge) in frontier.iter().take(8).enumerate() {
        let facet_idx = ridge.incident_facet();
        let dropped_vertex = ridge.dropped_vertex();
        println!(
            "  frontier[{idx}]: ridge={:?} incident_facet={:?} dropped_vertex={dropped_vertex}",
            ridge.ridge(),
            partial_graph.facet_keys[facet_idx],
        );
    }

    let repair_start = Instant::now();
    let repaired = partial
        .repair_simplicial_frontier(SimplicialFrontierRepairOptions::default(), &exact_eps)
        .expect("simplicial frontier repair must succeed");
    let repair_time = repair_start.elapsed();
    println!("timing: partial_resolve={partial_time:?} repair={repair_time:?}");
    let report = repaired.report();
    println!(
        "repair: steps={} new_facets={} remaining_frontier={} new_non_simplicial={}",
        report.steps_attempted,
        report.new_facets,
        report.remaining_frontier_ridges,
        report.new_non_simplicial_facets
    );
    let repaired_facets: BTreeSet<Vec<usize>> = repaired
        .facets()
        .iter()
        .map(|facet| facet.vertices().to_vec())
        .collect();
    let (missing_vs_exact, extra_vs_exact) = diff_counts(&exact_facets, &repaired_facets);
    println!("repaired: facets={}", repaired_facets.len(),);
    println!("exact vs repaired: missing_facets={missing_vs_exact} extra_facets={extra_vs_exact}");

    let repaired_vec: Vec<Vec<usize>> = repaired_facets.iter().cloned().collect();
    let repaired_frontier =
        simplicial_frontier_ridges(&repaired_vec, partial_graph.stats.dimension);
    println!("repaired frontier ridges: {}", repaired_frontier.len());

    let graph_repair_start = Instant::now();
    let graph_repaired = partial
        .repair_facet_graph(&non_exact, FacetGraphRepairOptions::default(), &exact_eps)
        .expect("facet graph repair must succeed");
    let graph_repair_time = graph_repair_start.elapsed();
    println!("timing: facet_graph_repair={graph_repair_time:?}");

    let graph_repaired_facets: BTreeSet<Vec<usize>> = graph_repaired
        .facets()
        .iter()
        .map(|facet| facet.vertices().to_vec())
        .collect();
    let (missing_vs_exact, extra_vs_exact) = diff_counts(&exact_facets, &graph_repaired_facets);
    let (missing_vs_partial, _) = diff_counts(&partial_facets, &graph_repaired_facets);
    println!(
        concat!(
            "facet-graph repaired: facets={} missing_vs_exact={} extra_vs_exact={} ",
            "missing_vs_partial={} unresolved_nodes={} steps={}",
        ),
        graph_repaired_facets.len(),
        missing_vs_exact,
        extra_vs_exact,
        missing_vs_partial,
        graph_repaired.report().unresolved_nodes,
        graph_repaired.report().steps_attempted
    );
    assert_eq!(
        missing_vs_partial, 0,
        "facet-graph repair should not drop partial facets"
    );
    assert_eq!(
        extra_vs_exact, 0,
        "facet-graph repair should not invent extra facets"
    );
}

#[test]
fn frcheck_seed_657708882109938962_resolution_default_f64_eps_repair_matches_exact() {
    let vertices: Vec<Vec<f64>> = vec![
        vec![
            2.794849156543302,
            4.1114551297784665,
            6.828132714634027,
            -0.6044237384515139,
            -3.854543532441248,
            6.699666817959251,
            1.0,
        ],
        vec![
            1.2396217802756162,
            2.660844346516562,
            -2.4443646692746013,
            4.859055581496108,
            7.457643307255996,
            2.0670664568935226,
            1.0,
        ],
        vec![
            -0.7395734954934774,
            -1.4274749235874005,
            3.7694845571521896,
            9.603830040255488,
            -2.6306131998579074,
            -7.212682556533219,
            1.0,
        ],
        vec![
            9.278708905827354,
            -9.231746988726538,
            -3.196835646609193,
            6.333561437836007,
            0.6965292275763844,
            6.809037859736229,
            1.0,
        ],
        vec![
            2.9886806323773527,
            -4.492087276334118,
            0.30443055108653994,
            -5.82607002646375,
            -5.615948167460063,
            -5.5946944779658025,
            1.0,
        ],
        vec![
            5.643451826810356,
            -7.9947924726529696,
            3.4973968477072184,
            -2.7572235735090223,
            3.474546836902146,
            -0.46340257906734905,
            1.0,
        ],
        vec![
            -2.8522675214720516,
            -4.269274506620273,
            5.757023908422587,
            1.9393358668184302,
            6.39625272335482,
            1.1296364546632987,
            1.0,
        ],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ];

    let poly_options = PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::List,
        input_incidence: IncidenceOutput::Set,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let exact = build_poly_dd_int::<RugRat>(&vertices, poly_options.clone());
    let non_exact = build_poly_dd_inexact::<f64>(&vertices, poly_options.clone());

    let exact_graph = build_fr_graph(&exact);
    let non_graph = build_fr_graph(&non_exact);

    let exact_facets: BTreeSet<Vec<usize>> = exact_graph.facet_keys.iter().cloned().collect();
    let non_facets: BTreeSet<Vec<usize>> = non_graph.facet_keys.iter().cloned().collect();
    let (missing_facets, extra_facets) = diff_counts(&exact_facets, &non_facets);
    let (missing_ridges, extra_ridges) =
        diff_counts(&exact_graph.ridge_keys, &non_graph.ridge_keys);

    assert!(
        missing_facets > 0 || extra_facets > 0 || missing_ridges > 0 || extra_ridges > 0,
        "expected FR graph mismatch between f64 and RugRat"
    );

    let cert = non_exact
        .certificate()
        .expect("non-exact output must carry resolve certificate");
    let exact_eps = RugRat::default_eps();

    let partial = cert
        .resolve_partial_as::<RugRat>(poly_options.clone(), ResolveOptions::default(), &exact_eps)
        .expect("partial resolution must not fail");

    let repaired = partial
        .repair_simplicial_frontier(SimplicialFrontierRepairOptions::default(), &exact_eps)
        .expect("simplicial frontier repair must succeed");

    let repaired_keys: Vec<Vec<usize>> = repaired
        .facets()
        .iter()
        .map(|facet| facet.vertices().to_vec())
        .collect();
    let repaired_facets: BTreeSet<Vec<usize>> = repaired_keys.iter().cloned().collect();

    let (missing_vs_exact, extra_vs_exact) = diff_counts(&exact_facets, &repaired_facets);
    assert_eq!(missing_vs_exact, 0, "repaired must not miss exact facets");
    assert_eq!(extra_vs_exact, 0, "repaired must not add extra facets");

    let repaired_ridges =
        edge_key_set_from_simplicial_facets(&repaired_keys, exact_graph.stats.dimension);
    let (missing_ridges_vs_exact, extra_ridges_vs_exact) =
        diff_counts(&exact_graph.ridge_keys, &repaired_ridges);
    assert_eq!(
        missing_ridges_vs_exact, 0,
        "repaired must not miss exact ridges"
    );
    assert_eq!(
        extra_ridges_vs_exact, 0,
        "repaired must not add extra ridges"
    );
}

#[test]
fn frcheck_seed_15655891048604234947_resolution_default_f64_eps_repair_matches_exact() {
    let vertices: Vec<Vec<f64>> = vec![
        vec![
            -2.1528641565787643,
            3.4778094085960944,
            9.558351455253554,
            5.2799530324246025,
            1.0,
        ],
        vec![
            -9.57192793546437,
            2.2382384912202227,
            3.5721919261794,
            8.314539695615998,
            1.0,
        ],
        vec![
            -7.364130129440354,
            2.6071013222028796,
            -6.449399399626388,
            9.293399765749303,
            1.0,
        ],
        vec![
            -0.287673641337971,
            -2.8535285795716048,
            -9.905098811032019,
            -2.6468182094702453,
            1.0,
        ],
        vec![
            -1.7044305000023066,
            -8.382424265575903,
            5.975855682114904,
            -2.6096420724456193,
            1.0,
        ],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
    ];

    let poly_options = PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::List,
        input_incidence: IncidenceOutput::Set,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let exact = build_poly_dd_int::<RugRat>(&vertices, poly_options.clone());
    let non_exact = build_poly_dd_inexact::<f64>(&vertices, poly_options.clone());

    let exact_graph = build_fr_graph(&exact);
    let non_graph = build_fr_graph(&non_exact);

    let exact_facets: BTreeSet<Vec<usize>> = exact_graph.facet_keys.iter().cloned().collect();
    let non_facets: BTreeSet<Vec<usize>> = non_graph.facet_keys.iter().cloned().collect();
    let (missing_facets, extra_facets) = diff_counts(&exact_facets, &non_facets);
    let (missing_ridges, extra_ridges) =
        diff_counts(&exact_graph.ridge_keys, &non_graph.ridge_keys);

    assert!(
        missing_facets > 0 || extra_facets > 0 || missing_ridges > 0 || extra_ridges > 0,
        "expected FR graph mismatch between f64 and RugRat"
    );

    let cert = non_exact
        .certificate()
        .expect("non-exact output must carry resolve certificate");
    let exact_eps = RugRat::default_eps();

    let partial = cert
        .resolve_partial_as::<RugRat>(poly_options.clone(), ResolveOptions::default(), &exact_eps)
        .expect("partial resolution must not fail");
    assert!(
        partial.issues().is_empty(),
        "expected no partial resolution issues (missing facets only)"
    );

    let partial_graph = build_fr_graph(partial.polyhedron());
    assert_eq!(
        partial_graph.facet_keys, non_graph.facet_keys,
        "expected partial resolution to match non-exact facet set"
    );

    let frontier_ridges =
        simplicial_frontier_ridge_count(&partial_graph.facet_keys, exact_graph.stats.dimension);
    assert_eq!(
        frontier_ridges, 5,
        "expected simplicial frontier ridges to detect missing facet boundary"
    );

    let repaired = partial
        .repair_simplicial_frontier(SimplicialFrontierRepairOptions::default(), &exact_eps)
        .expect("simplicial frontier repair must succeed");
    assert_eq!(
        repaired.report().remaining_frontier_ridges,
        0,
        "expected frontier to be empty after repair"
    );

    let repaired_keys: Vec<Vec<usize>> = repaired
        .facets()
        .iter()
        .map(|facet| facet.vertices().to_vec())
        .collect();
    let repaired_frontier =
        simplicial_frontier_ridge_count(&repaired_keys, exact_graph.stats.dimension);
    assert_eq!(
        repaired_frontier, 0,
        "expected no simplicial frontier ridges after repair"
    );
    let repaired_facets: BTreeSet<Vec<usize>> = repaired_keys.iter().cloned().collect();

    let (missing_vs_exact, extra_vs_exact) = diff_counts(&exact_facets, &repaired_facets);
    assert_eq!(missing_vs_exact, 0, "repaired must not miss exact facets");
    assert_eq!(extra_vs_exact, 0, "repaired must not add extra facets");

    let repaired_ridges =
        edge_key_set_from_simplicial_facets(&repaired_keys, exact_graph.stats.dimension);
    let (missing_ridges_vs_exact, extra_ridges_vs_exact) =
        diff_counts(&exact_graph.ridge_keys, &repaired_ridges);
    assert_eq!(
        missing_ridges_vs_exact, 0,
        "repaired must not miss exact ridges"
    );
    assert_eq!(
        extra_ridges_vs_exact, 0,
        "repaired must not add extra ridges"
    );
}
