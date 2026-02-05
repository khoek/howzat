use super::*;
use crate::dd::Umpire;
use crate::dd::ray::{EdgeTarget, RayNoNegatives};
use crate::dd::SatSet;
use crate::matrix::LpMatrix;
use calculo::num::{Num, Sign};
use hullabaloo::types::{ComputationStatus, Inequality, InequalityKind};
use std::cmp::Ordering;

#[test]
fn column_reduce_keeps_cost_vector_alignment() {
    let matrix =
        LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0]])
            .with_row_vec(vec![5.0, 11.0, 13.0]);

    let equality_kinds = vec![InequalityKind::Inequality; 2];
    let mut cone = Cone::<_, _>::new(matrix.clone(), equality_kinds, ConeOptions::default())
        .expect("build cone")
        .into_basis_prep(f64::default_eps());

    assert!(cone.find_initial_rays());
    assert!(cone.is_col_reduced());
    assert_eq!(cone.col_count(), 2);
    assert_eq!(cone.column_mapping(), &[Some(0), None, Some(1)]);
    assert_eq!(
        cone.matrix().rows().map(|r| r.to_vec()).collect::<Vec<_>>(),
        vec![vec![1.0, 1.0], vec![0.0, 1.0]]
    );
    assert_eq!(cone.matrix().row_vec().to_vec(), vec![5.0, 13.0]);
    assert_eq!(
        cone.input_matrix()
            .rows()
            .map(|r| r.to_vec())
            .collect::<Vec<_>>(),
        matrix.rows().map(|r| r.to_vec()).collect::<Vec<_>>()
    );
    assert_eq!(
        cone.input_matrix().row_vec().to_vec(),
        matrix.row_vec().to_vec()
    );
}

#[test]
fn evaluate_partitions_floor_first_infeasible_for_nonnegative_rays() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![0.0], vec![1.0], vec![0.0]]);
    let kinds = vec![InequalityKind::Inequality; 3];
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());
    cone.core.iter_state.iteration = 2;

    for row in 0..cone.row_count() {
        cone.core.ctx.assign_sat_id_for_row(row);
    }
    let mut zero_zs = SatSet::default();
    zero_zs.insert(cone.core.ctx.sat_id_for_row(0).unwrap());
    zero_zs.insert(cone.core.ctx.sat_id_for_row(1).unwrap());
    zero_zs.insert(cone.core.ctx.sat_id_for_row(2).unwrap());
    let zero_zs_sig = zero_zs.signature_u64();
    let zero_ray = Ray {
        vector: vec![0.0],
        class: RayClass {
            zero_set: zero_zs,
            zero_set_sig: zero_zs_sig,
            zero_set_count: 3,
            first_infeasible_row: Some(1),
            feasible: true,
            weakly_feasible: true,
            last_eval_row: None,
            last_eval: 0.0,
            last_sign: Sign::Zero,
        },
    };
    let mut pos_zs = SatSet::default();
    pos_zs.insert(cone.core.ctx.sat_id_for_row(0).unwrap());
    pos_zs.insert(cone.core.ctx.sat_id_for_row(2).unwrap());
    let pos_zs_sig = pos_zs.signature_u64();
    let pos_ray = Ray {
        vector: vec![1.0],
        class: RayClass {
            zero_set: pos_zs,
            zero_set_sig: pos_zs_sig,
            zero_set_count: 2,
            first_infeasible_row: Some(1),
            feasible: true,
            weakly_feasible: true,
            last_eval_row: None,
            last_eval: 0.0,
            last_sign: Sign::Zero,
        },
    };

    let row_count = cone.row_count();
    cone.core.ray_graph.reset(row_count);
    Umpire::<f64>::reset_infeasible_counts(&mut cone.umpire, row_count);
    let zero_id = cone
        .core
        .ray_graph
        .insert_active(RayNoNegatives { inner: zero_ray });
    let pos_id = cone
        .core
        .ray_graph
        .insert_active(RayNoNegatives { inner: pos_ray });
    cone.core.ray_graph.set_order(vec![zero_id, pos_id]);
    let partition = cone.evaluate_row_partition(2);
    assert!(partition.negative.is_empty());

    let zero_infeasible = cone
        .first_infeasible_position(cone.core.ray_graph.ray_data(zero_id).unwrap())
        .unwrap_or(cone.row_count());
    let pos_infeasible = cone
        .first_infeasible_position(cone.core.ray_graph.ray_data(pos_id).unwrap())
        .unwrap_or(cone.row_count());
    assert_eq!(zero_infeasible, cone.iteration());
    assert_eq!(pos_infeasible, cone.iteration());
}

#[test]
fn refresh_preserves_edges_after_flooring_first_infeasible() {
    let matrix = LpMatrix::<f64, Inequality>::new(5, 1);
    let kinds = vec![InequalityKind::Inequality; 5];
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());
    cone.core.iter_state.iteration = 3;

    for row in 0..cone.row_count() {
        cone.core.ctx.assign_sat_id_for_row(row);
    }
    let mut ray_one_zs = SatSet::default();
    for i in 0..cone.row_count() {
        ray_one_zs.insert(cone.core.ctx.sat_id_for_row(i).unwrap());
    }
    let ray_one_zs_sig = ray_one_zs.signature_u64();
    let ray_one = Ray {
        vector: vec![1.0],
        class: RayClass {
            zero_set: ray_one_zs,
            zero_set_sig: ray_one_zs_sig,
            zero_set_count: cone.row_count(),
            first_infeasible_row: Some(2),
            feasible: true,
            weakly_feasible: true,
            last_eval_row: None,
            last_eval: 0.0,
            last_sign: Sign::Zero,
        },
    };
    let mut ray_two_zs = SatSet::default();
    for i in 0..cone.row_count() {
        ray_two_zs.insert(cone.core.ctx.sat_id_for_row(i).unwrap());
    }
    let ray_two_zs_sig = ray_two_zs.signature_u64();
    let ray_two = Ray {
        vector: vec![1.0],
        class: RayClass {
            zero_set: ray_two_zs,
            zero_set_sig: ray_two_zs_sig,
            zero_set_count: cone.row_count(),
            first_infeasible_row: Some(4),
            feasible: true,
            weakly_feasible: true,
            last_eval_row: None,
            last_eval: 0.0,
            last_sign: Sign::Zero,
        },
    };

    let row_count = cone.row_count();
    cone.core.ray_graph.reset(row_count);
    Umpire::<f64>::reset_infeasible_counts(&mut cone.umpire, row_count);
    let ray_one_id = cone
        .core
        .ray_graph
        .insert_active(RayNoNegatives { inner: ray_one });
    let ray_two_id = cone
        .core
        .ray_graph
        .insert_active(RayNoNegatives { inner: ray_two });
    cone.core.ray_graph.set_order(vec![ray_one_id, ray_two_id]);
    let iteration = cone.iteration();
    cone.core.ray_graph.queue_edge(
        iteration,
        AdjacencyEdge {
            retained: ray_two_id,
            removed: ray_one_id,
        },
    );

    cone.refresh_edge_buckets();

    let updated = cone.core.ray_graph.ray_data(ray_one_id).unwrap();
    let _updated_pos = cone
        .first_infeasible_position(updated)
        .unwrap_or(cone.row_count());
    assert!(
        cone.core.ray_graph.edge_count() <= 1,
        "edge count should not grow when refreshing buckets"
    );
    assert!(
        cone.core.ray_graph.edges_at(cone.iteration()).len() <= 1,
        "edge bucket should not gain edges during refresh"
    );
}

#[test]
fn assume_nondegenerate_mode_does_not_panic_when_processing_edges() {
    // This is a regression test for a panic caused by `process_iteration_edges` building a
    // zero-length scratch `RowSet` when the incidence index is disabled under
    // `AssumeNondegenerate`.
    //
    // Use a small, nondegenerate polytope (unit square) in H-representation.
    let rows = vec![
        vec![0.0, 1.0, 0.0],  // x >= 0
        vec![0.0, 0.0, 1.0],  // y >= 0
        vec![1.0, -1.0, 0.0], // 1 - x >= 0
        vec![1.0, 0.0, -1.0], // 1 - y >= 0
    ];
    let matrix = LpMatrix::<f64, Inequality>::from_rows(rows);
    let kinds = vec![InequalityKind::Inequality; matrix.row_count()];
    let options = ConeOptions {
        enumeration_mode: EnumerationMode::AssumeNondegenerate,
        basis_initialization: BasisInitialization::Top,
    };

    let output = Cone::<f64, Inequality>::new(matrix, kinds, options)
        .expect("cone")
        .run_dd(f64::default_eps())
        .expect("dd");

    assert!(output.nondegenerate_assumed());
    assert_eq!(output.status(), ComputationStatus::AllFound);
    assert!(output.ray_count() > 0, "expected some rays in output");
}

#[test]
fn first_infeasible_at_zero_is_not_overwritten() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0], vec![1.0]]);
    let kinds = vec![InequalityKind::Inequality; 2];
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());

    let ray = Ray {
        vector: vec![1.0],
        class: {
            let zero_set = SatSet::default();
            let zero_set_sig = zero_set.signature_u64();
            RayClass {
                zero_set,
                zero_set_sig,
                zero_set_count: 0,
                first_infeasible_row: Some(0),
                feasible: false,
                weakly_feasible: false,
                last_eval_row: None,
                last_eval: -1.0,
                last_sign: Sign::Negative,
            }
        },
    };

    let row_count = cone.row_count();
    cone.core.ray_graph.reset(row_count);
    Umpire::<f64>::reset_infeasible_counts(&mut cone.umpire, row_count);
    let ray_id = cone
        .core
        .ray_graph
        .insert_active(RayNoNegatives { inner: ray });
    cone.core.ray_graph.set_order(vec![ray_id]);

    let partition = cone.evaluate_row_partition(1);

    drop(partition);

    let updated = cone.core.ray_graph.ray_data(ray_id).unwrap();
    assert_eq!(updated.first_infeasible_row(), Some(0));
    let relaxed = cone.relaxed_enumeration();
    let negative_set = cone
        .sign_sets_for_ray_id(ray_id, relaxed, true)
        .expect("ray must exist");
    assert!(negative_set.contains(0));
}

#[test]
fn edge_target_iteration_accepts_zero_first_infeasible() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0], vec![1.0]]);
    let kinds = vec![InequalityKind::Inequality; 2];
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());

    let ray_a = Ray {
        vector: vec![1.0],
        class: {
            let zero_set = SatSet::default();
            let zero_set_sig = zero_set.signature_u64();
            RayClass {
                zero_set,
                zero_set_sig,
                zero_set_count: 0,
                first_infeasible_row: Some(0),
                feasible: false,
                weakly_feasible: false,
                last_eval_row: None,
                last_eval: 1.0,
                last_sign: Sign::Positive,
            }
        },
    };
    let ray_b = Ray {
        vector: vec![1.0],
        class: {
            let zero_set = SatSet::default();
            let zero_set_sig = zero_set.signature_u64();
            RayClass {
                zero_set,
                zero_set_sig,
                zero_set_count: 0,
                first_infeasible_row: Some(1),
                feasible: false,
                weakly_feasible: false,
                last_eval_row: None,
                last_eval: -1.0,
                last_sign: Sign::Negative,
            }
        },
    };

    let row_count = cone.row_count();
    cone.core.ray_graph.reset(row_count);
    let retained = cone
        .core
        .ray_graph
        .insert_active(RayNoNegatives { inner: ray_a });
    let removed = cone
        .core
        .ray_graph
        .insert_active(RayNoNegatives { inner: ray_b });
    cone.core.ray_graph.set_order(vec![retained, removed]);

    let target = cone.edge_target_iteration(&AdjacencyEdge { retained, removed });
    assert_eq!(target, EdgeTarget::Scheduled(0));
}

#[test]
fn strict_zero_values_violate_in_exact_mode() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0]]);
    let kinds = vec![InequalityKind::StrictInequality];
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());

    cone.core.ctx.assign_sat_id_for_row(0);
    let ray_id = cone.add_ray(vec![0.0]);
    let partition = cone.evaluate_row_partition(0);

    assert!(partition.negative.is_empty());
    assert_eq!(partition.zero, &[ray_id]);
    {
        let ray = cone.core.ray_graph.ray_data(ray_id).unwrap();
        assert!(ray.is_feasible());
        assert!(ray.is_weakly_feasible());
        assert_eq!(ray.first_infeasible_row(), None);
    }
    let relaxed = cone.relaxed_enumeration();
    let negative_set = cone
        .sign_sets_for_ray_id(ray_id, relaxed, true)
        .expect("ray must exist");
    assert!(!negative_set.contains(0));
    assert!(
        cone.core
            .ray_graph
            .ray_data(ray_id)
            .unwrap()
            .zero_set()
            .contains(cone.core.ctx.sat_id_for_row(0).unwrap())
    );
}

#[test]
fn relaxed_mode_marks_strict_zero_only_weakly_feasible() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0]]);
    let kinds = vec![InequalityKind::StrictInequality];
    let options = ConeOptions {
        enumeration_mode: EnumerationMode::Relaxed,
        ..ConeOptions::default()
    };
    let mut cone = Cone::<_, _>::new(matrix, kinds, options)
        .unwrap()
        .into_basis_prep(f64::default_eps());

    cone.core.ctx.assign_sat_id_for_row(0);
    let ray_id = cone.add_ray(vec![0.0]);
    let partition = cone.evaluate_row_partition(0);

    assert!(partition.negative.is_empty());
    assert_eq!(partition.zero, &[ray_id]);
    {
        let ray = cone.core.ray_graph.ray_data(ray_id).unwrap();
        assert!(ray.is_feasible());
        assert!(ray.is_weakly_feasible());
        assert_eq!(ray.first_infeasible_row(), None);
    }
    let relaxed = cone.relaxed_enumeration();
    let negative_set = cone
        .sign_sets_for_ray_id(ray_id, relaxed, true)
        .expect("ray must exist");
    assert!(!negative_set.contains(0));
    assert!(
        cone.core
            .ray_graph
            .ray_data(ray_id)
            .unwrap()
            .zero_set()
            .contains(cone.core.ctx.sat_id_for_row(0).unwrap())
    );
}

#[test]
fn cached_feasibility_indices_follow_ray_lifecycle() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0], vec![-1.0]]);

    let kinds = vec![InequalityKind::Inequality; 2];
    let eps = f64::default_eps();
    let umpire = crate::dd::SinglePrecisionUmpire::with_halfspace_policy(
        eps,
        crate::dd::umpire::policies::MinCutoff::default(),
    );
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep_with_umpire(umpire);

    cone.add_ray(vec![1.0]);
    cone.add_ray(vec![-1.0]);
    {
        let state = &mut cone.state;
        let (core, umpire) = state.split_core_umpire();
        let active = core.ray_graph.active_len();
        let choice = <_ as Umpire<f64>>::choose_next_halfspace(
            umpire,
            &core.ctx,
            &core.weakly_added_halfspaces,
            core.iter_state.iteration,
            active,
        );
        assert_eq!(choice, Some(0));
    }

    let partition = cone.evaluate_row_partition(1);
    let partition_view = RayPartition {
        negative: &partition.negative,
        positive: &partition.positive,
        zero: &partition.zero,
    };
    cone.delete_negative_rays(partition_view);
    cone.recycle_partition(partition);

    assert_eq!(cone.core.ray_graph.active_len(), 1);
    {
        let state = &mut cone.state;
        let (core, umpire) = state.split_core_umpire();
        let active = core.ray_graph.active_len();
        let choice = <_ as Umpire<f64>>::choose_next_halfspace(
            umpire,
            &core.ctx,
            &core.weakly_added_halfspaces,
            core.iter_state.iteration,
            active,
        );
        assert_eq!(choice, Some(1));
    }
}

#[test]
fn deleting_all_rays_marks_region_empty() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0]]);
    let kinds = vec![InequalityKind::Inequality];
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());

    let ray_id = cone.add_ray(vec![-1.0]);
    let partition = cone.evaluate_row_partition(0);
    assert_eq!(partition.negative, &[ray_id]);

    let partition_view = RayPartition {
        negative: &partition.negative,
        positive: &partition.positive,
        zero: &partition.zero,
    };
    cone.delete_negative_rays(partition_view);
    cone.recycle_partition(partition);

    assert_eq!(cone.status(), ComputationStatus::RegionEmpty);
    assert_eq!(cone.core.ray_graph.active_len(), 0);
}

#[test]
fn strict_inequalities_participate_in_basis_selection() {
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

    let equality_kinds = vec![InequalityKind::StrictInequality, InequalityKind::Inequality];
    let mut cone = Cone::<_, _>::new(matrix, equality_kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());
    let mut rank = 0usize;

    cone.find_basis(&mut rank);

    assert_eq!(rank, 1);
    assert!(!cone.initial_halfspaces().contains(0));
    assert!(cone.initial_halfspaces().contains(1));
}

#[test]
fn lex_compare_columns_respects_order_vector() {
    let eps = f64::default_eps();
    let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
    let kinds = vec![InequalityKind::Inequality; 2];
    let mut cone = Cone::<_, _>::new(matrix, kinds, ConeOptions::default())
        .unwrap()
        .into_basis_prep(f64::default_eps());

    // Identity basis => tableau == matrix.
    cone.rebuild_tableau();

    // Swap the row order: now the first row inspected in lex-compare is row 1.
    cone.core.ctx.order_vector = vec![1, 0];
    cone.core.ctx.refresh_row_to_pos();

    assert_eq!(cone.lex_compare_columns(0, 1, &eps), Ordering::Greater);
    assert_eq!(cone.lex_compare_columns(1, 0, &eps), Ordering::Less);
}
