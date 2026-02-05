use calculo::num::Num;
use howzat::matrix::{IncidenceConfig, LpMatrix};
use hullabaloo::types::{Generator, Inequality, RowId};

#[test]
fn input_incidence_remaps_compact_outputs() {
    let eps = f64::default_eps();
    let input =
        LpMatrix::<f64, Inequality>::from_rows(vec![vec![0.0, 1.0, 0.0], vec![0.0, -1.0, 0.0]]);

    let compact = LpMatrix::<f64, Generator>::from_rows(vec![vec![1.0]]);
    let mapping = vec![Some(0usize), None, None];

    let (ainc_mapped, red_mapped, dom_mapped) = input
        .input_incidence_against(&compact, &eps, IncidenceConfig::with_column_mapping(&mapping))
        .expect("mapped incidence");
    assert_eq!(ainc_mapped.family_size(), 2);
    assert_eq!(ainc_mapped.sets()[0].cardinality(), 1);
    assert_eq!(ainc_mapped.sets()[1].cardinality(), 1);
    assert!(red_mapped.is_empty());
    assert!(dom_mapped.contains(0) && dom_mapped.contains(1));

    let adj = input
        .input_adjacency_against(&compact, &eps, IncidenceConfig::with_column_mapping(&mapping))
        .expect("adjacency");
    assert!(adj.sets()[0].contains(1));
    assert!(adj.sets()[1].contains(0));

    let full = LpMatrix::<f64, Generator>::from_rows(vec![vec![1.0, 0.0, 0.0]]);
    let (ainc_full, red_full, dom_full) = input
        .input_incidence_against(&full, &eps, IncidenceConfig::default())
        .expect("full incidence");

    assert_eq!(ainc_full.sets()[0], ainc_mapped.sets()[0]);
    assert_eq!(ainc_full.sets()[1], ainc_mapped.sets()[1]);
    assert_eq!(red_full, red_mapped);
    assert_eq!(dom_full, dom_mapped);

    let out_inc = input
        .output_incidence_against(&compact, &eps, IncidenceConfig::with_column_mapping(&mapping))
        .expect("output incidence");
    assert_eq!(out_inc.family_size(), compact.row_count());
    assert_eq!(out_inc.sets()[0].cardinality(), 2);
}

#[test]
fn input_incidence_adds_slack_row_for_non_homogeneous_hrep() {
    let eps = f64::default_eps();
    let input = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, -1.0]]);
    let outputs = LpMatrix::<f64, Generator>::from_rows(vec![vec![0.0, 1.0]]);

    let (ainc, redundant, dominant) = input
        .input_incidence_against(&outputs, &eps, IncidenceConfig::default())
        .expect("incidence with slack row");

    assert_eq!(ainc.family_size(), 2);
    assert!(ainc.set(0).expect("original constraint set").is_empty());

    let slack_idx = 1;
    assert!(
        ainc.set(slack_idx)
            .expect("slack constraint set")
            .contains(RowId::new(0))
    );
    assert!(redundant.contains(RowId::new(0)));
    assert!(dominant.contains(RowId::new(slack_idx)));
}
