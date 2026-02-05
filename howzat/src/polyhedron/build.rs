//! Polyhedron construction from matrix input via DD or LRS algorithms.
//!
//! This module provides the main entry points for converting H-representation to V-representation
//! (and vice versa) using the double-description (DD) method or the LRS reverse-search algorithm.

use super::{
    AdjacencyBuildProfile, DdConfig, DdEdgeHint, DdRepairHints, DdTrace, IncidenceArtifacts,
    IncidenceRequests, LrsConfig, OutputRayData, PolyhedronOptions, PolyhedronOutput,
    build_adjacency_with,
};
use crate::HowzatError as Error;
use crate::dd::{ConeBuilder, ConeEngine, IntUmpire, RayData as _, RayKey, RayOrigin, Umpire};
use crate::matrix::{LpMatrix, LpMatrixBuilder, Matrix as DenseMatrix};
use calculo::num::{CoerceFrom, Epsilon, Num, Rat};
use hullabaloo::adjacency::{AdjacencyStore, Graph};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{
    ColSet, ComputationStatus, DualRepresentation, InequalityKind, Representation,
    RepresentationKind, Row, RowSet,
};
use smallvec::SmallVec;
use std::ops::Deref;

type DualOf<R> = <R as DualRepresentation>::Dual;

impl<N: Num, R: DualRepresentation, Adj: AdjacencyStore> PolyhedronOutput<N, R, Adj> {
    fn from_matrix_dd_with_umpire<U: Umpire<N>>(
        matrix: LpMatrix<N, R>,
        dd: DdConfig,
        umpire: U,
    ) -> Result<Self, Error> {
        let DdConfig {
            cone: cone_options,
            poly: poly_options,
        } = dd;
        let original_row_count = matrix.row_count();
        let original_matrix = matrix;

        let (working_matrix, working_equality_kinds) =
            prepare_working_matrix(&original_matrix, &umpire)?;

        assert!(
            working_equality_kinds.len() >= original_row_count,
            "working_equality_kinds shorter than input (kinds={} input_rows={})",
            working_equality_kinds.len(),
            original_row_count
        );

        let row_positions =
            Self::expanded_row_positions(&working_equality_kinds, original_row_count);
        let stored_equality_kinds = working_equality_kinds
            .get(..original_row_count)
            .map(|slice| slice.to_vec())
            .unwrap_or_else(|| working_equality_kinds.clone());

        let homogeneous = umpire.is_homogeneous(&working_matrix);
        let cost_vector = original_matrix.row_vec().to_vec();

        let mut cone_builder = ConeBuilder::new(working_matrix, working_equality_kinds);
        cone_builder.options(cone_options);
        let cone = cone_builder.finish()?;
        let cone = cone.into_basis_prep_with_umpire(umpire).run_dd()?;

        let status = cone.status();
        let is_empty = status == ComputationStatus::RegionEmpty;
        let column_mapping = cone.column_mapping().to_vec();

        let trace = build_dd_trace(&cone, &poly_options);
        let mut adjacency_profile = poly_options
            .profile_adjacency
            .then(|| Box::new(AdjacencyBuildProfile::default()));

        let linearity_dimension = adjusted_linearity_dimension(&cone);
        let dimension = compute_dimension(&cone, homogeneous);

        if is_empty {
            return Ok(build_empty_polyhedron(
                original_matrix,
                stored_equality_kinds,
                homogeneous,
                dimension,
                linearity_dimension,
                cost_vector,
                row_positions,
                column_mapping,
                trace,
                adjacency_profile,
            ));
        }

        let mut output_rays = collect_output_rays(&cone);
        let output_size = output_rays.len();

        let requests = IncidenceRequests::from_options(&poly_options);
        let output = build_output_matrix::<DualOf<R>, R, N>(&mut output_rays, &original_matrix)
            .unwrap_or_else(|| Self::empty_output_matrix(original_matrix.col_count()));

        let incidence_artifacts = build_incidence_artifacts(
            &cone,
            &output_rays,
            &original_matrix,
            original_row_count,
            output_size,
            requests,
        );
        let IncidenceArtifacts {
            incidence,
            input_incidence,
            redundant_rows,
            dominant_rows,
            input_adjacency,
        } = incidence_artifacts;

        let adjacency = if requests.build_output_adjacency {
            // Get active rows from cone state
            let active_rows = if R::KIND == RepresentationKind::Generator {
                cone.ground_set().clone()
            } else {
                cone.added_halfspaces().clone()
            };

            // Build output linearity mask
            let family_size = output_rays.len();
            let mut output_linearity = RowSet::new(family_size);
            for (idx, ray) in output_rays.iter().enumerate() {
                if ray.is_linearity {
                    output_linearity.insert(idx);
                }
            }

            // Build incidence SetFamily from rays
            let set_capacity = cone.row_count();
            let mut incidence_builder = SetFamily::builder(family_size, set_capacity);
            for (idx, ray) in output_rays.iter().enumerate() {
                for id in ray.zero_set.iter() {
                    incidence_builder.insert_into_set(idx, id);
                }
            }
            let ray_incidence = incidence_builder.build();

            build_adjacency_with::<Adj>(
                &ray_incidence,
                &output_linearity,
                &active_rows,
                cone.col_count(),
                None,
                cone.nondegenerate_assumed(),
                adjacency_profile.as_deref_mut(),
            )
        } else {
            None
        };

        let repair_hints = if poly_options.save_repair_hints {
            build_dd_repair_hints(
                &cone,
                &original_matrix,
                &output_rays,
                incidence.as_ref(),
                adjacency.as_ref(),
                &column_mapping,
            )
        } else {
            None
        };

        Ok(Self {
            representation: R::KIND,
            homogeneous,
            dimension,
            input: original_matrix,
            output,
            equality_kinds: stored_equality_kinds,
            linearity_dimension,
            output_size,
            incidence,
            adjacency,
            input_incidence,
            input_adjacency,
            redundant_rows,
            dominant_rows,
            status,
            is_empty: false,
            cost_vector: Some(cost_vector),
            row_positions,
            column_mapping,
            trace,
            repair_hints,
            adjacency_profile,
        })
    }

    pub fn from_matrix_dd<U: Umpire<N>>(
        matrix: LpMatrix<N, R>,
        dd: DdConfig,
        umpire: U,
    ) -> Result<Self, Error>
    {
        Self::from_matrix_dd_with_umpire(matrix, dd, umpire)
    }

    /// Exact DD via integer-only internal arithmetic.
    ///
    /// This entrypoint is only available for rational types and runs the DD core with
    /// `IntUmpire`, ensuring intermediate rays are stored as primitive integer vectors.
    pub fn from_matrix_dd_int(
        matrix: LpMatrix<N, R>,
        dd: DdConfig,
    ) -> Result<Self, Error>
    where
        N: Rat,
    {
        let umpire = IntUmpire::new(N::default_eps());
        Self::from_matrix_dd_with_umpire(matrix, dd, umpire)
    }

    pub fn from_matrix_dd_int_with_umpire<E, H>(
        matrix: LpMatrix<N, R>,
        dd: DdConfig,
        umpire: IntUmpire<N, E, H>,
    ) -> Result<Self, Error>
    where
        N: Rat,
        E: Epsilon<N>,
        H: crate::dd::umpire::policies::HalfspacePolicy<N>,
    {
        Self::from_matrix_dd_with_umpire(matrix, dd, umpire)
    }

    /// Compute a polyhedron conversion using the LRS (reverse search) engine, converting the
    /// input into exact rationals internally and returning results in `M`.
    pub fn from_matrix_lrs_as_exact<Q, M>(
        matrix: LpMatrix<N, R>,
        config: LrsConfig,
        eps: &impl Epsilon<M>,
    ) -> Result<PolyhedronOutput<M, R, Adj>, Error>
    where
        Q: Rat + CoerceFrom<N>,
        M: Num + CoerceFrom<Q>,
    {
        let LrsConfig {
            poly: poly_options,
            lrs: lrs_options,
        } = config;
        let original_matrix = matrix;
        let original_row_count = original_matrix.row_count();

        let original_matrix_exact: LpMatrix<Q, R> = original_matrix.coerce_as::<Q>()?;

        let (working_matrix, working_equality_kinds) =
            prepare_working_matrix_lrs(&original_matrix_exact)?;

        assert!(
            working_equality_kinds.len() >= original_row_count,
            "working_equality_kinds shorter than input (kinds={} input_rows={})",
            working_equality_kinds.len(),
            original_row_count
        );

        let row_positions: Vec<isize> = (0..original_row_count).map(|i| i as isize).collect();
        let stored_equality_kinds = working_equality_kinds
            .get(..original_row_count)
            .map(|slice| slice.to_vec())
            .unwrap_or_else(|| working_equality_kinds.clone());

        let requests = IncidenceRequests::from_options(&poly_options);
        let lrs_matrix: LpMatrix<Q, R> = working_matrix;

        let enumeration = if requests.wants_any_incidence() {
            lrs_matrix
                .lrs_enumerate_rows_with_incidence(lrs_options.clone())
                .map(|(rows, cols, data, inc)| (rows, cols, data, Some(inc)))
        } else {
            lrs_matrix
                .lrs_enumerate_rows(lrs_options.clone())
                .map(|(rows, cols, data)| (rows, cols, data, None))
        };

        let (status, output_size, output_cols, output_data, lrs_incidence) = match enumeration {
            Ok((rows, cols, data, incidence)) => {
                (ComputationStatus::AllFound, rows, cols, data, incidence)
            }
            Err(crate::lrs::Error::Infeasible) => (
                ComputationStatus::RegionEmpty,
                0usize,
                original_matrix_exact.col_count(),
                Vec::new(),
                None,
            ),
            Err(_) => return Err(Error::NumericallyInconsistent),
        };
        let is_empty = status == ComputationStatus::RegionEmpty;

        assert_eq!(
            output_cols,
            original_matrix_exact.col_count(),
            "LRS output column count mismatch (output_cols={} input_cols={})",
            output_cols,
            original_matrix_exact.col_count()
        );

        let output_exact: LpMatrix<Q, DualOf<R>> = if output_size == 0 {
            LpMatrixBuilder::<Q, DualOf<R>>::with_columns(output_cols).build()
        } else {
            LpMatrixBuilder::<Q, DualOf<R>>::from_flat(output_size, output_cols, output_data)
                .with_linearity(RowSet::new(output_size))
                .with_row_vec(original_matrix_exact.row_vec().to_vec())
                .with_objective(original_matrix_exact.objective())
                .build()
        };

        let mut adjacency_profile = poly_options
            .profile_adjacency
            .then(|| Box::new(AdjacencyBuildProfile::default()));

        let incidence_exact = if is_empty || !requests.wants_any_incidence() {
            None
        } else {
            let lrs_incidence = lrs_incidence
                .as_ref()
                .expect("LRS incidence must be present when incidence is requested");
            let mut builder = SetFamily::builder(output_size, original_row_count);
            for (idx, set) in lrs_incidence.sets().iter().enumerate() {
                let mut zs = RowSet::new(original_row_count);
                for row in set.iter().map(|r| r.as_index()) {
                    if row < original_row_count {
                        zs.insert(row);
                    }
                }
                builder.replace_set(idx, zs);
            }
            Some(builder.build())
        };

        let input: LpMatrix<M, R> = original_matrix_exact.coerce_as::<M>()?;
        let output: LpMatrix<M, DualOf<R>> = output_exact.coerce_as::<M>()?;
        let cost_vector = input.row_vec().to_vec();

        let homogeneous = input.is_homogeneous(eps);
        let dimension = if homogeneous {
            input.col_count()
        } else {
            input.col_count().saturating_sub(1)
        };

        let ignored_rows = RowSet::new(input.row_count());
        let ignored_cols = ColSet::new(input.col_count());
        let rank_info = input.storage().rank(&ignored_rows, &ignored_cols, eps);
        let linearity_dimension = input.col_count().saturating_sub(rank_info.rank);

        let column_mapping: Vec<Option<usize>> = (0..input.col_count()).map(Some).collect();
        let incidence = incidence_exact;

        let (input_incidence, redundant_rows, dominant_rows, input_adjacency) =
            build_input_incidence_artifacts::<R>(
                &incidence,
                output_size,
                input.row_count(),
                input.linearity(),
                requests,
                is_empty,
            );

        let adjacency = if !is_empty && requests.build_output_adjacency {
            let lrs_incidence = lrs_incidence
                .as_ref()
                .expect("LRS incidence must be present when output adjacency is requested");
            let active_rows = RowSet::all(lrs_incidence.set_capacity());
            build_adjacency_with::<Adj>(
                lrs_incidence,
                output.linearity(),
                &active_rows,
                rank_info.rank,
                None,
                false,
                adjacency_profile.as_deref_mut(),
            )
        } else {
            None
        };

        Ok(PolyhedronOutput::<M, R, Adj> {
            representation: R::KIND,
            homogeneous,
            dimension,
            input,
            output,
            equality_kinds: stored_equality_kinds,
            linearity_dimension,
            output_size,
            incidence,
            adjacency,
            input_incidence,
            input_adjacency,
            redundant_rows,
            dominant_rows,
            status,
            is_empty,
            cost_vector: Some(cost_vector),
            row_positions,
            column_mapping,
            trace: None,
            repair_hints: None,
            adjacency_profile,
        })
    }
}

fn prepare_working_matrix<N: Num, R: DualRepresentation, U: Umpire<N>>(
    original_matrix: &LpMatrix<N, R>,
    umpire: &U,
) -> Result<(LpMatrix<N, R>, Vec<InequalityKind>), Error> {
    let mut working_matrix = original_matrix.clone();
    let mut working_equality_kinds =
        PolyhedronOutput::<N, R>::derive_equality_kinds(&working_matrix);

    // For non-homogeneous H-reps, run DD in homogeneous coordinates and add the x0 >= 0 constraint.
    if R::KIND == RepresentationKind::Inequality && !umpire.is_homogeneous(&working_matrix) {
        let mut slack = vec![N::zero(); working_matrix.col_count()];
        if let Some(first) = slack.first_mut() {
            *first = N::one();
        }
        let cols = working_matrix.col_count();
        let mut storage = DenseMatrix::with_capacity(working_matrix.row_count() + 1, cols);
        storage.extend_from_matrix(working_matrix.storage());
        storage.push(&slack);

        let rows_len = storage.row_count();
        let mut lin = working_matrix.linearity().clone();
        lin.resize(rows_len);
        let mut builder = LpMatrixBuilder::with_columns(cols);
        builder = builder.with_storage(storage).with_linearity(lin);
        working_matrix = builder.build();
        working_equality_kinds.push(InequalityKind::Inequality);
    }

    Ok((working_matrix, working_equality_kinds))
}

fn prepare_working_matrix_lrs<Q: Rat, R: DualRepresentation>(
    original_matrix_exact: &LpMatrix<Q, R>,
) -> Result<(LpMatrix<Q, R>, Vec<InequalityKind>), Error> {
    let mut working_matrix = original_matrix_exact.clone();
    let mut working_equality_kinds =
        PolyhedronOutput::<Q, R>::derive_equality_kinds(&working_matrix);
    let exact_eps = Q::default_eps();

    if R::KIND == RepresentationKind::Inequality && !working_matrix.is_homogeneous(&exact_eps) {
        let mut slack = vec![Q::zero(); working_matrix.col_count()];
        if let Some(first) = slack.first_mut() {
            *first = Q::one();
        }
        let cols = working_matrix.col_count();
        let mut storage = DenseMatrix::with_capacity(working_matrix.row_count() + 1, cols);
        storage.extend_from_matrix(working_matrix.storage());
        storage.push(&slack);

        let rows_len = storage.row_count();
        let mut lin = working_matrix.linearity().clone();
        lin.resize(rows_len);
        let mut builder = LpMatrixBuilder::with_columns(cols);
        builder = builder.with_storage(storage).with_linearity(lin);
        working_matrix = builder.build();
        working_equality_kinds.push(InequalityKind::Inequality);
    }

    Ok((working_matrix, working_equality_kinds))
}

fn build_dd_trace<N: Num, R: DualRepresentation, U: Umpire<N>, C>(
    cone: &C,
    poly_options: &PolyhedronOptions,
) -> Option<Box<DdTrace<N>>>
where
    C: Deref<Target = ConeEngine<N, R, U>>,
{
    if !poly_options.save_basis_and_tableau {
        return None;
    }

    let tableau_snapshot = cone.tableau_snapshot();
    let tableau_rows = tableau_snapshot.as_ref().map_or(0, |t| t.len());
    let tableau_cols = tableau_snapshot
        .as_ref()
        .and_then(|t| t.first())
        .map_or(0, |r| r.len());

    Some(Box::new(DdTrace {
        saved_basis: Some(cone.basis_saved().clone()),
        tableau_snapshot,
        tableau_nonbasic: Some(cone.tableau_nonbasic().to_vec()),
        tableau_basic_col_for_row: Some(cone.tableau_basic_cols().to_vec()),
        tableau_rows,
        tableau_cols,
    }))
}

fn adjusted_linearity_dimension<N: Num, R: DualRepresentation, U: Umpire<N>, C>(cone: &C) -> Row
where
    C: Deref<Target = ConeEngine<N, R, U>>,
{
    let skip_affine_col0 =
        R::KIND == RepresentationKind::Inequality && !cone.umpire.is_homogeneous(cone.matrix());
    let removed = cone
        .column_mapping()
        .iter()
        .enumerate()
        .filter(|(idx, map)| map.is_none() && (!skip_affine_col0 || *idx != 0))
        .count();
    if removed > 0 {
        removed
    } else {
        cone.linearity_dimension()
    }
}

fn compute_dimension<N: Num, R: DualRepresentation, U: Umpire<N>, C>(
    cone: &C,
    homogeneous: bool,
) -> Row
where
    C: Deref<Target = ConeEngine<N, R, U>>,
{
    if homogeneous {
        cone.original_dimension()
    } else {
        let dim = cone.original_dimension();
        assert!(dim > 0, "non-homogeneous input requires leading column");
        dim - 1
    }
}

#[allow(clippy::too_many_arguments)]
fn build_empty_polyhedron<N: Num, R: DualRepresentation, Adj>(
    original_matrix: LpMatrix<N, R>,
    stored_equality_kinds: Vec<InequalityKind>,
    homogeneous: bool,
    dimension: Row,
    linearity_dimension: Row,
    cost_vector: Vec<N>,
    row_positions: Vec<isize>,
    column_mapping: Vec<Option<usize>>,
    trace: Option<Box<DdTrace<N>>>,
    adjacency_profile: Option<Box<AdjacencyBuildProfile>>,
) -> PolyhedronOutput<N, R, Adj> {
    let output = PolyhedronOutput::<N, R, Adj>::empty_output_matrix(original_matrix.col_count());

    PolyhedronOutput {
        representation: R::KIND,
        homogeneous,
        dimension,
        input: original_matrix,
        output,
        equality_kinds: stored_equality_kinds,
        linearity_dimension,
        output_size: 0,
        incidence: None,
        adjacency: None,
        input_incidence: None,
        input_adjacency: None,
        redundant_rows: None,
        dominant_rows: None,
        status: ComputationStatus::RegionEmpty,
        is_empty: true,
        cost_vector: Some(cost_vector),
        row_positions,
        column_mapping,
        trace,
        repair_hints: None,
        adjacency_profile,
    }
}

pub(crate) fn collect_output_rays<N, InRep, U, ZR, C>(cone: &C) -> Vec<OutputRayData<N>>
where
    N: Num,
    ZR: crate::dd::ZeroRepr,
    U: Umpire<N, ZR>,
    C: Deref<Target = ConeEngine<N, InRep, U, ZR>>,
    InRep: DualRepresentation,
{
    let ground = cone.ground_set();

    cone.core
        .ray_graph
        .active_order
        .iter()
        .copied()
        .filter_map(|ray_id| {
            let ray_data = cone.core.ray_graph.ray_data(ray_id)?;
            if !ray_data.is_feasible() {
                return None;
            }

            let mut zero_set = RowSet::new(cone.row_count());
            for id in <ZR::Set as crate::dd::ZeroSet>::iter(ray_data.zero_set()) {
                let row = <ZR as crate::dd::ZeroRepr>::row_for_id(&cone.core.ctx, id);
                zero_set.insert(row);
            }
            zero_set.intersection_inplace(ground);

            let compact = cone.umpire.ray_vector_for_output(ray_data);
            let vector = cone.expand_ray_vector(&compact);

            let mut near_zero_rows: Vec<Row> = Vec::new();
            if let Some(rows) = cone.umpire.near_zero_rows_on_ray(ray_data) {
                near_zero_rows.extend(rows.iter().copied().filter(|&row| ground.contains(row)));
                near_zero_rows.sort_unstable();
                near_zero_rows.dedup();
            }

            Some(OutputRayData {
                ray_id,
                vector,
                zero_set,
                is_linearity: false,
                near_zero_rows,
            })
        })
        .collect()
}

pub(crate) fn build_output_matrix<
    TargetRep: Representation,
    TemplateRep: Representation,
    N: Num,
>(
    rays: &mut [OutputRayData<N>],
    template: &LpMatrix<N, TemplateRep>,
) -> Option<LpMatrix<N, TargetRep>> {
    if rays.is_empty() {
        return None;
    }

    let row_count = rays.len();
    let cols = template.col_count();

    let mut linearity = RowSet::new(row_count);
    let mut data = Vec::with_capacity(row_count * cols);

    for (idx, ray) in rays.iter_mut().enumerate() {
        if ray.is_linearity {
            linearity.insert(idx);
        }
        let row = std::mem::take(&mut ray.vector);
        debug_assert_eq!(row.len(), cols);
        data.extend(row);
    }

    let storage = DenseMatrix::from_flat(row_count, cols, data);

    let mut builder = LpMatrixBuilder::<N, TargetRep>::with_columns(cols);
    builder = builder.with_storage(storage);
    builder = builder.with_linearity(linearity);
    builder = builder.with_row_vec(template.row_vec().to_vec());
    builder = builder.with_objective(template.objective());
    Some(builder.build())
}

pub(crate) fn build_incidence_artifacts<N: Num, R: DualRepresentation, U: Umpire<N>, C>(
    cone: &C,
    rays: &[OutputRayData<N>],
    original_matrix: &LpMatrix<N, R>,
    original_row_count: usize,
    output_size: usize,
    requests: IncidenceRequests,
) -> IncidenceArtifacts
where
    C: Deref<Target = ConeEngine<N, R, U>>,
{
    if !requests.build_output_incidence {
        return IncidenceArtifacts::default();
    }

    let lifting = {
        let mut lifting = vec![Vec::new(); cone.matrix().row_count()];
        for (row, rows) in lifting[..original_row_count].iter_mut().enumerate() {
            rows.push(row);
        }
        lifting
    };

    let mut artifacts = IncidenceArtifacts {
        incidence: Some(
            build_incidence(cone, rays, original_row_count, &lifting)
                .unwrap_or_else(|| SetFamily::new(0, 0)),
        ),
        ..IncidenceArtifacts::default()
    };

    let (input_incidence, redundant_rows, dominant_rows, input_adjacency) =
        build_input_incidence_artifacts::<R>(
            &artifacts.incidence,
            output_size,
            original_row_count,
            original_matrix.linearity(),
            requests,
            false,
        );
    artifacts.input_incidence = input_incidence;
    artifacts.redundant_rows = redundant_rows;
    artifacts.dominant_rows = dominant_rows;
    artifacts.input_adjacency = input_adjacency;

    artifacts
}

pub(crate) fn build_incidence<N: Num, InRep: Representation, U: Umpire<N>, C>(
    cone: &C,
    rays: &[OutputRayData<N>],
    input_rows: usize,
    lifting: &[Vec<usize>],
) -> Option<SetFamily>
where
    C: Deref<Target = ConeEngine<N, InRep, U>>,
{
    if rays.is_empty() {
        return None;
    }
    let mut builder = SetFamily::builder(rays.len(), input_rows);
    let mut lifted = RowSet::new(input_rows);

    for (pos, ray) in rays.iter().enumerate() {
        builder.clear_set(pos);
        debug_assert!(ray.zero_set.subset_of(cone.ground_set()));
        lift_zero_set_into(&ray.zero_set, input_rows, lifting, &mut lifted);
        for row in lifted.iter() {
            builder.insert_into_set(pos, row);
        }
    }
    Some(builder.build())
}

pub(crate) fn build_input_incidence_artifacts<R: DualRepresentation>(
    incidence: &Option<SetFamily>,
    output_size: usize,
    row_count: usize,
    linearity: &RowSet,
    requests: IncidenceRequests,
    skip: bool,
) -> (
    Option<SetFamily>,
    Option<RowSet>,
    Option<RowSet>,
    Option<SetFamily>,
) {
    if !requests.build_input_incidence || skip {
        return (None, None, None, None);
    }

    let incidence_ref = incidence
        .as_ref()
        .expect("input incidence requested without incidence");
    let input_incidence = incidence_ref.transpose();

    let (redundant_rows, dominant_rows) = if R::KIND == RepresentationKind::Generator {
        (RowSet::new(row_count), RowSet::new(row_count))
    } else {
        input_incidence.classify_input_incidence(linearity, output_size)
    };
    let input_adjacency = requests.build_input_adjacency.then(|| {
        hullabaloo::adjacency::input_adjacency_from_incidence_set_family(
            &input_incidence,
            &redundant_rows,
            &dominant_rows,
        )
    });

    (
        Some(input_incidence),
        Some(redundant_rows),
        Some(dominant_rows),
        input_adjacency,
    )
}

fn lift_zero_set_into(
    source: &RowSet,
    target_capacity: usize,
    lifting: &[Vec<usize>],
    out: &mut RowSet,
) {
    out.resize(target_capacity);
    out.clear();
    for row in source.iter() {
        let orig_rows = lifting
            .get(row.as_index())
            .expect("lifting index out of range");
        for &orig in orig_rows {
            out.insert(orig);
        }
    }
}

fn build_dd_repair_hints<N: Num, R: DualRepresentation, U: Umpire<N>, C, A: Graph>(
    cone: &C,
    input: &LpMatrix<N, R>,
    rays: &[OutputRayData<N>],
    incidence: Option<&SetFamily>,
    adjacency: Option<&A>,
    column_mapping: &[Option<usize>],
) -> Option<Box<DdRepairHints>>
where
    C: Deref<Target = ConeEngine<N, R, U>>,
{
    let incidence = incidence?;
    if rays.len() != incidence.family_size() {
        return None;
    }

    let cols = input.col_count();
    if cols == 0 {
        return None;
    }

    let redund_cols = PolyhedronOutput::<N, R>::redundant_cols_from_column_mapping(column_mapping);
    let mut redund_mask = vec![false; cols];
    for &col in &redund_cols {
        if col < cols {
            redund_mask[col] = true;
        }
    }
    let reduced_cols = cols.saturating_sub(redund_cols.len());
    let witness_target_rank = reduced_cols.checked_sub(1)?;
    let ridge_target_rank = witness_target_rank.saturating_sub(1);

    let eps = N::default_eps();

    let ray_keys: Vec<RayKey> = rays
        .iter()
        .map(|ray| cone.core.ray_graph.ray_key(ray.ray_id))
        .collect();
    let ray_origins: Vec<Option<RayOrigin>> = rays
        .iter()
        .map(|ray| cone.core.ray_graph.ray_origin(ray.ray_id).copied())
        .collect();

    let facet_witness_basis =
        build_facet_witness_bases(input, incidence, witness_target_rank, &redund_mask, &eps);

    let incoming_edge_hints = build_incoming_edge_hints(
        input,
        incidence,
        adjacency,
        &facet_witness_basis,
        &ray_keys,
        &ray_origins,
        ridge_target_rank,
        &redund_mask,
        &eps,
    );

    let (facet_near_zero_offsets, facet_near_zero_rows) = build_near_zero_rows(rays);

    Some(Box::new(DdRepairHints {
        facet_witness_basis,
        incoming_edge_hints,
        facet_near_zero_offsets,
        facet_near_zero_rows,
        dedup_drops: cone.core.dedup_drops,
    }))
}

fn build_facet_witness_bases<N: Num, R: DualRepresentation>(
    input: &LpMatrix<N, R>,
    incidence: &SetFamily,
    witness_target_rank: usize,
    redund_mask: &[bool],
    eps: &impl Epsilon<N>,
) -> Vec<SmallVec<[Row; 16]>> {
    let input_rows = input.row_count();
    let mut facet_witness_basis: Vec<SmallVec<[Row; 16]>> =
        Vec::with_capacity(incidence.family_size());
    let mut scratch_candidates: Vec<Row> = Vec::new();

    for out_idx in 0..incidence.family_size() {
        let Some(face) = incidence.set(out_idx) else {
            facet_witness_basis.push(SmallVec::new());
            continue;
        };

        let card = face.cardinality();
        if card < witness_target_rank {
            facet_witness_basis.push(SmallVec::new());
            continue;
        }

        if card == witness_target_rank {
            let mut basis: SmallVec<[Row; 16]> = SmallVec::with_capacity(card);
            for v in face.iter().map(|v| v.as_index()) {
                if v < input_rows {
                    basis.push(v);
                }
            }
            basis.sort_unstable();
            basis.dedup();
            if basis.len() != witness_target_rank {
                basis.clear();
            }
            facet_witness_basis.push(basis);
            continue;
        }

        scratch_candidates.clear();
        scratch_candidates.extend(
            face.iter()
                .map(|v| v.as_index())
                .filter(|&v| v < input_rows),
        );
        let Some(selected) = input.storage().select_row_basis_rows(
            &scratch_candidates,
            witness_target_rank,
            redund_mask,
            eps,
        ) else {
            facet_witness_basis.push(SmallVec::new());
            continue;
        };

        let mut basis: SmallVec<[Row; 16]> = SmallVec::new();
        basis.extend(selected);
        basis.sort_unstable();
        basis.dedup();
        if basis.len() != witness_target_rank {
            basis.clear();
        }
        facet_witness_basis.push(basis);
    }

    facet_witness_basis
}

#[allow(clippy::too_many_arguments)]
fn build_incoming_edge_hints<N: Num, R: DualRepresentation, A: Graph>(
    input: &LpMatrix<N, R>,
    incidence: &SetFamily,
    adjacency: Option<&A>,
    facet_witness_basis: &[SmallVec<[Row; 16]>],
    ray_keys: &[RayKey],
    ray_origins: &[Option<RayOrigin>],
    ridge_target_rank: usize,
    redund_mask: &[bool],
    eps: &impl Epsilon<N>,
) -> Option<Vec<Vec<DdEdgeHint>>> {
    let adjacency = adjacency?;
    if adjacency.node_count() != incidence.family_size() {
        return None;
    }

    let mut out: Vec<Vec<DdEdgeHint>> = Vec::with_capacity(adjacency.node_count());
    let mut ridge_candidates: Vec<Row> = Vec::new();

    for to in 0..adjacency.node_count() {
        let Some(face_to) = incidence.set(to) else {
            out.push(Vec::new());
            continue;
        };
        let origin_to = ray_origins.get(to).copied().flatten();

        let mut hints: Vec<DdEdgeHint> = Vec::with_capacity(adjacency.degree(to));
        for from in adjacency.neighbors(to) {
            let hint = build_single_edge_hint(
                input,
                incidence,
                facet_witness_basis,
                ray_keys,
                origin_to,
                face_to,
                from,
                to,
                ridge_target_rank,
                redund_mask,
                eps,
                &mut ridge_candidates,
            );
            hints.push(hint);
        }
        out.push(hints);
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn build_single_edge_hint<N: Num, R: DualRepresentation>(
    input: &LpMatrix<N, R>,
    incidence: &SetFamily,
    facet_witness_basis: &[SmallVec<[Row; 16]>],
    ray_keys: &[RayKey],
    origin_to: Option<RayOrigin>,
    face_to: &RowSet,
    from: usize,
    to: usize,
    ridge_target_rank: usize,
    redund_mask: &[bool],
    eps: &impl Epsilon<N>,
    ridge_candidates: &mut Vec<Row>,
) -> DdEdgeHint {
    let input_rows = input.row_count();
    let mut ridge_basis: SmallVec<[Row; 16]> = SmallVec::new();
    let mut drop_candidates: SmallVec<[Row; 8]> = SmallVec::new();
    let mut from_witness: Option<Row> = None;
    let mut minimizers: SmallVec<[Row; 8]> = SmallVec::new();
    let mut minimizers_complete = false;

    let entered_row = origin_to.and_then(|origin| {
        let from_key = ray_keys.get(from).copied()?;
        if origin.parent_a == Some(from_key) || origin.parent_b == Some(from_key) {
            origin.creation_row
        } else {
            None
        }
    });

    if let Some(face_from) = incidence.set(from) {
        // Find witness from 'from' facet not in 'to' facet
        for v in face_from.iter().map(|id| id.as_index()) {
            if v < input_rows && !face_to.contains(v) {
                from_witness = Some(v);
                break;
            }
        }

        // Find minimizers (vertices in 'to' not in 'from')
        let mut minimizer_count = 0usize;
        for v in face_to.iter().map(|id| id.as_index()) {
            if v < input_rows && !face_from.contains(v) {
                minimizer_count += 1;
                if minimizers.len() < 8 {
                    minimizers.push(v);
                }
            }
        }
        minimizers_complete = minimizer_count <= 8;
        minimizers.sort_unstable();
        minimizers.dedup();
        if minimizers.len() > 8 {
            minimizers.truncate(8);
            minimizers_complete = false;
        }

        // Build ridge basis if needed
        if ridge_target_rank > 0 {
            ridge_candidates.clear();
            let (small, large) = if face_to.cardinality() <= face_from.cardinality() {
                (face_to, face_from)
            } else {
                (face_from, face_to)
            };
            ridge_candidates.extend(
                small
                    .iter()
                    .map(|id| id.as_index())
                    .filter(|&v| v < input_rows && large.contains(v)),
            );

            if ridge_candidates.len() >= ridge_target_rank {
                if ridge_candidates.len() == ridge_target_rank {
                    ridge_basis.extend(ridge_candidates.iter().copied());
                } else if let Some(selected) = input.storage().select_row_basis_rows(
                    ridge_candidates,
                    ridge_target_rank,
                    redund_mask,
                    eps,
                ) {
                    ridge_basis.extend(selected);
                }
                ridge_basis.sort_unstable();
                ridge_basis.dedup();
                if ridge_basis.len() != ridge_target_rank {
                    ridge_basis.clear();
                }
            }

            // Build drop candidates
            if let Some(to_basis) = facet_witness_basis.get(to) {
                for &v in to_basis.iter() {
                    if drop_candidates.len() >= 8 {
                        break;
                    }
                    if v < input_rows && !face_from.contains(v) {
                        drop_candidates.push(v);
                    }
                }
            }
            for v in face_to.iter().map(|id| id.as_index()) {
                if drop_candidates.len() >= 8 {
                    break;
                }
                if v < input_rows && !face_from.contains(v) {
                    drop_candidates.push(v);
                }
            }
            drop_candidates.sort_unstable();
            drop_candidates.dedup();
            if drop_candidates.len() > 8 {
                drop_candidates.truncate(8);
            }
        }
    }

    DdEdgeHint {
        neighbor: from,
        ridge_basis,
        drop_candidates,
        from_witness,
        minimizers,
        minimizers_complete,
        entered_row,
    }
}

fn build_near_zero_rows<N: Num>(rays: &[OutputRayData<N>]) -> (Option<Vec<usize>>, Vec<Row>) {
    let facet_near_zero_total: usize = rays.iter().map(|ray| ray.near_zero_rows.len()).sum();

    if facet_near_zero_total == 0 {
        return (None, Vec::new());
    }

    let mut offsets: Vec<usize> = Vec::with_capacity(rays.len() + 1);
    offsets.push(0);
    let mut rows: Vec<Row> = Vec::with_capacity(facet_near_zero_total);
    for ray in rays {
        rows.extend_from_slice(&ray.near_zero_rows);
        offsets.push(rows.len());
    }
    (Some(offsets), rows)
}
