//! Certificate-based polyhedron resolution.
//!
//! This module provides exact resolution of polyhedra from inexact floating-point
//! computations using incidence certificates. The resolution process reconstructs
//! exact rational facet normals from the combinatorial structure (incidence).

use super::build::build_input_incidence_artifacts;
use super::int::{
    BareissSolveScratch, IntRowMatrix, scaled_integer_rows, scaled_integer_vec,
    select_row_basis_rows_int, solve_nullspace_1d_rows_with_unit_cols_bareiss_int,
};
use super::{
    AdjacencyBuildProfile, IncidenceRequests, PartialResolveIssue, PartialResolveResult,
    PolyhedronOptions, PolyhedronOutput, PreparedPartialResolveMinimal,
    PreparedPartialResolveResult, ResolveError, ResolveMode, ResolveOptions, VerificationIssue,
    build_adjacency,
};
use crate::HowzatError as Error;
use crate::matrix::{LpMatrix, LpMatrixBuilder};
use calculo::linalg;
use calculo::num::{CoerceFrom, Epsilon, Int, Num, Rat, Sign};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{
    ComputationStatus, DualRepresentation, InequalityKind, Representation, RepresentationKind, Row,
    RowId, RowSet,
};

type DualOf<R> = <R as DualRepresentation>::Dual;

/// Negate a sign value.
#[inline(always)]
fn negate_sign(sign: Sign) -> Sign {
    match sign {
        Sign::Positive => Sign::Negative,
        Sign::Negative => Sign::Positive,
        Sign::Zero => Sign::Zero,
    }
}

/// Context for resolving a single output row from its incidence witness.
struct RowResolutionContext<'a, M: Rat, R: DualRepresentation> {
    input: &'a LpMatrix<M, R>,
    int_input_rows: &'a IntRowMatrix<<M as Rat>::Int>,
    equality_kinds: &'a [InequalityKind],
    dd_witness_bases: Option<&'a [smallvec::SmallVec<[Row; 16]>]>,
    redund_cols: &'a [usize],
    redund_mask: &'a [bool],
    witness_target_rank: usize,
    options: &'a ResolveOptions,
}

/// Result of solving for a witness normal.
enum SolvedRow<M: Rat> {
    Int(Vec<<M as Rat>::Int>),
    Rat(Vec<M>),
}

/// Result of orienting a resolved row.
enum OrientationResult {
    /// Row is valid with the current sign.
    Valid,
    /// Row does not determine orientation.
    Undecided,
    /// Row needs to be negated.
    NeedsNegation,
    /// Both signs violate constraints.
    Infeasible,
}

struct DotSignIntScratch<Z: Int> {
    zero: Z,
    dot: Z,
    tmp: Z,
}

impl<Z: Int> Default for DotSignIntScratch<Z> {
    fn default() -> Self {
        Self {
            zero: Z::zero(),
            dot: Z::zero(),
            tmp: Z::zero(),
        }
    }
}

impl<N: Num, R: DualRepresentation> PolyhedronOutput<N, R> {
    pub(crate) fn resolve_from_incidence_certificate_as<M>(
        &self,
        poly_options: &PolyhedronOptions,
        incidence: &SetFamily,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PolyhedronOutput<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        self.resolve_from_incidence_certificate_prepared_as(
            poly_options,
            incidence,
            options,
            eps,
            ResolveMode::Strict,
        )
        .map(PreparedPartialResolveResult::into_partial)
        .map(PartialResolveResult::into_polyhedron)
    }

    pub(crate) fn resolve_partial_from_incidence_certificate_as<M>(
        &self,
        poly_options: &PolyhedronOptions,
        incidence: &SetFamily,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PartialResolveResult<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        self.resolve_from_incidence_certificate_prepared_as(
            poly_options,
            incidence,
            options,
            eps,
            ResolveMode::Partial,
        )
        .map(PreparedPartialResolveResult::into_partial)
    }

    pub(crate) fn resolve_partial_from_incidence_certificate_as_prepared<M>(
        &self,
        poly_options: &PolyhedronOptions,
        incidence: &SetFamily,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PreparedPartialResolveResult<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        self.resolve_from_incidence_certificate_prepared_as(
            poly_options,
            incidence,
            options,
            eps,
            ResolveMode::Partial,
        )
    }

    pub(crate) fn resolve_partial_from_incidence_certificate_as_prepared_minimal<M>(
        &self,
        _poly_options: &PolyhedronOptions,
        incidence: &SetFamily,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
    ) -> Result<PreparedPartialResolveMinimal<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        // Phase 1: Validation
        self.validate_resolution_inputs(incidence)?;

        let input: LpMatrix<M, R> = self
            .input
            .coerce_as::<M>()
            .map_err(|_| ResolveError::ConversionFailure)?;

        if incidence.set_capacity() != input.row_count() {
            return Err(ResolveError::CertificateShapeMismatch);
        }
        if self.column_mapping.len() != input.col_count() {
            return Err(ResolveError::CertificateShapeMismatch);
        }

        // Phase 2: Setup resolution context
        let cols = input.col_count();
        let redund_cols = Self::redundant_cols_from_column_mapping(&self.column_mapping);
        let redund_mask = build_redund_mask(cols, &redund_cols);
        let reduced_cols = cols.saturating_sub(redund_cols.len());
        let witness_target_rank = reduced_cols.saturating_sub(1);

        let int_input_rows =
            scaled_integer_rows::<M, R>(&input).ok_or(ResolveError::ConversionFailure)?;
        let k = cols
            .checked_sub(1)
            .ok_or(ResolveError::CertificateShapeMismatch)?;
        let mut bareiss_scratch = BareissSolveScratch::<<M as Rat>::Int>::new(k);

        let dd_witness_bases = self
            .dd_repair_hints()
            .map(|hints| hints.facet_witness_basis.as_slice());
        let dd_edge_hints = self
            .dd_repair_hints()
            .and_then(|hints| hints.incoming_edge_hints.as_deref());

        let ctx = RowResolutionContext {
            input: &input,
            int_input_rows: &int_input_rows,
            equality_kinds: &self.equality_kinds,
            dd_witness_bases,
            redund_cols: &redund_cols,
            redund_mask: &redund_mask,
            witness_target_rank,
            options: &options,
        };

        // Phase 3: Resolve each output row
        let output_row_count = self.output.row_count();
        let resolved_capacity = output_row_count
            .checked_mul(cols)
            .ok_or(ResolveError::ComputationError(Error::DimensionTooLarge))?;

        let output_is_generator =
            <DualOf<R> as Representation>::KIND == RepresentationKind::Generator;
        let certificate_only = options.partial_use_certificate_only;

        let mut resolved_int_data: Vec<<M as Rat>::Int> = Vec::with_capacity(resolved_capacity);
        let mut kept_output_rows: Vec<Row> = Vec::with_capacity(output_row_count);
        let mut facet_vertices: Vec<Vec<Row>> = Vec::with_capacity(output_row_count);
        let mut issues: Vec<PartialResolveIssue<M>> = Vec::new();
        let mut scratch_candidates: Vec<Row> = Vec::new();
        let mut dot_scratch = DotSignIntScratch::<<M as Rat>::Int>::default();

        for out_idx in 0..output_row_count {
            let Some(witness) = incidence.set(out_idx) else {
                return Err(ResolveError::CertificateShapeMismatch);
            };

            let solved = solve_witness_normal(
                &ctx,
                witness,
                out_idx,
                &mut bareiss_scratch,
                &mut scratch_candidates,
                eps,
            );

            let Some(solved) = solved else {
                issues.push(PartialResolveIssue::WitnessNotOneDim {
                    output_row: out_idx,
                });
                continue;
            };

            let (mut out_row, mut out_row_int) = convert_solved_row::<M>(solved, eps)?;

            if output_is_generator {
                let row = out_row.get_or_insert_with(|| int_to_rat_vec::<M>(&out_row_int));
                PolyhedronOutput::<M, R>::normalize_generator(row, eps);
                out_row_int =
                    scaled_integer_vec::<M>(row).ok_or(ResolveError::ConversionFailure)?;
            }
            drop(out_row);

            let negates_dot =
                !output_is_generator || out_row_int.first().is_none_or(|v| v.is_zero());

            let to_near_zero = self
                .dd_repair_hints()
                .map(|hints| hints.facet_near_zero_rows(out_idx))
                .unwrap_or(&[]);

            let mut vertices: Vec<Row> =
                Vec::with_capacity(witness.cardinality().saturating_add(to_near_zero.len()));
            let oriented = if certificate_only {
                orient_certificate_only_int(
                    &ctx,
                    &mut out_row_int,
                    &mut dot_scratch,
                    witness,
                    to_near_zero,
                    dd_edge_hints.and_then(|h| h.get(out_idx)),
                    negates_dot,
                    &mut vertices,
                )?
            } else {
                false
            };

            if !oriented {
                let kept = orient_full_scan_int(
                    &ctx,
                    &mut out_row_int,
                    &mut dot_scratch,
                    negates_dot,
                    &mut vertices,
                    out_idx,
                    &mut issues,
                    eps,
                )?;
                if !kept {
                    continue;
                }
            }

            resolved_int_data.extend(out_row_int);
            kept_output_rows.push(out_idx);
            facet_vertices.push(vertices);
        }

        let output_size = kept_output_rows.len();
        let int_output_rows = IntRowMatrix::new(output_size, cols, resolved_int_data)
            .ok_or(ResolveError::ConversionFailure)?;

        let homogeneous = input.is_homogeneous(eps);
        let dimension = if homogeneous {
            input.col_count()
        } else {
            input.col_count().saturating_sub(1)
        };
        let cost_vector = input.row_vec().to_vec();

        Ok(PreparedPartialResolveMinimal {
            template: PolyhedronOutput::<M, R> {
                representation: R::KIND,
                homogeneous,
                dimension,
                input,
                output: PolyhedronOutput::<M, R>::empty_output_matrix(cols),
                equality_kinds: self.equality_kinds.clone(),
                linearity_dimension: self.linearity_dimension,
                output_size: 0,
                incidence: None,
                adjacency: None,
                input_incidence: None,
                input_adjacency: None,
                redundant_rows: self.redundant_rows.clone(),
                dominant_rows: self.dominant_rows.clone(),
                status: ComputationStatus::InProgress,
                is_empty: false,
                cost_vector: Some(cost_vector),
                row_positions: self.row_positions.clone(),
                column_mapping: self.column_mapping.clone(),
                trace: None,
                repair_hints: None,
                adjacency_profile: None,
            },
            kept_output_rows,
            issues,
            facet_vertices,
            int_input_rows,
            int_output_rows,
            redund_cols,
            redund_mask,
        })
    }

    pub(crate) fn resolve_from_incidence_certificate_prepared_as<M>(
        &self,
        poly_options: &PolyhedronOptions,
        incidence: &SetFamily,
        options: ResolveOptions,
        eps: &impl Epsilon<M>,
        mode: ResolveMode,
    ) -> Result<PreparedPartialResolveResult<M, R>, ResolveError<M>>
    where
        M: Rat + CoerceFrom<N>,
    {
        // Phase 1: Validation
        self.validate_resolution_inputs(incidence)?;

        let input: LpMatrix<M, R> = self
            .input
            .coerce_as::<M>()
            .map_err(|_| ResolveError::ConversionFailure)?;

        if incidence.set_capacity() != input.row_count() {
            return Err(ResolveError::CertificateShapeMismatch);
        }
        if self.column_mapping.len() != input.col_count() {
            return Err(ResolveError::CertificateShapeMismatch);
        }

        // Phase 2: Setup resolution context
        let cols = input.col_count();
        let redund_cols = Self::redundant_cols_from_column_mapping(&self.column_mapping);
        let redund_mask = build_redund_mask(cols, &redund_cols);
        let reduced_cols = cols.saturating_sub(redund_cols.len());
        let witness_target_rank = reduced_cols.saturating_sub(1);

        let int_input_rows =
            scaled_integer_rows::<M, R>(&input).ok_or(ResolveError::ConversionFailure)?;
        let k = cols
            .checked_sub(1)
            .ok_or(ResolveError::CertificateShapeMismatch)?;
        let mut bareiss_scratch = BareissSolveScratch::<<M as Rat>::Int>::new(k);

        let dd_witness_bases = self
            .dd_repair_hints()
            .map(|hints| hints.facet_witness_basis.as_slice());
        let dd_edge_hints = self
            .dd_repair_hints()
            .and_then(|hints| hints.incoming_edge_hints.as_deref());

        let ctx = RowResolutionContext {
            input: &input,
            int_input_rows: &int_input_rows,
            equality_kinds: &self.equality_kinds,
            dd_witness_bases,
            redund_cols: &redund_cols,
            redund_mask: &redund_mask,
            witness_target_rank,
            options: &options,
        };

        // Phase 3: Resolve each output row
        let requests = IncidenceRequests::from_options(poly_options);
        let wants_any_incidence = requests.wants_any_incidence();
        let verify_output_incidence = mode == ResolveMode::Strict;
        let verify_input_incidence = mode == ResolveMode::Strict
            && self.input_incidence.is_some()
            && self.redundant_rows.is_some()
            && self.dominant_rows.is_some();
        let store_zero_sets = wants_any_incidence || verify_input_incidence;
        let compute_zero_sets = store_zero_sets || verify_output_incidence;
        let certificate_only = mode == ResolveMode::Partial && options.partial_use_certificate_only;
        let output_is_generator =
            <DualOf<R> as Representation>::KIND == RepresentationKind::Generator;

        let output_row_count = self.output.row_count();
        let resolved_capacity = output_row_count
            .checked_mul(cols)
            .ok_or(ResolveError::ComputationError(Error::DimensionTooLarge))?;

        let mut resolved_data: Vec<M> = Vec::with_capacity(resolved_capacity);
        let mut resolved_int_data: Vec<<M as Rat>::Int> = Vec::with_capacity(resolved_capacity);
        let mut kept_output_rows: Vec<Row> = Vec::with_capacity(output_row_count);
        let mut resolved_zero_sets: Vec<RowSet> =
            Vec::with_capacity(if store_zero_sets { output_row_count } else { 0 });
        let mut issues: Vec<PartialResolveIssue<M>> = Vec::new();
        let mut verification_issues: Vec<VerificationIssue<M>> = Vec::new();
        let mut scratch_candidates: Vec<Row> = Vec::new();
        let mut dot_scratch = DotSignIntScratch::<<M as Rat>::Int>::default();

        for out_idx in 0..output_row_count {
            let Some(witness) = incidence.set(out_idx) else {
                return Err(ResolveError::CertificateShapeMismatch);
            };

            // Solve for the witness normal
            let solved = solve_witness_normal(
                &ctx,
                witness,
                out_idx,
                &mut bareiss_scratch,
                &mut scratch_candidates,
                eps,
            );

            let Some(solved) = solved else {
                match mode {
                    ResolveMode::Strict => {
                        return Err(ResolveError::WitnessNotOneDim {
                            output_row: out_idx,
                        });
                    }
                    ResolveMode::Partial => {
                        issues.push(PartialResolveIssue::WitnessNotOneDim {
                            output_row: out_idx,
                        });
                        continue;
                    }
                }
            };

            // Convert to both rational and integer forms
            let (mut out_row, mut out_row_int) = convert_solved_row::<M>(solved, eps)?;

            // Normalize generators
            if output_is_generator {
                let row = out_row.get_or_insert_with(|| int_to_rat_vec::<M>(&out_row_int));
                PolyhedronOutput::<M, R>::normalize_generator(row, eps);
                out_row_int =
                    scaled_integer_vec::<M>(row).ok_or(ResolveError::ConversionFailure)?;
            }

            // Determine sign orientation
            let negates_dot = !output_is_generator
                || out_row
                    .as_ref()
                    .and_then(|row| row.first())
                    .is_none_or(|v| eps.is_zero(v));

            let to_near_zero = self
                .dd_repair_hints()
                .map(|hints| hints.facet_near_zero_rows(out_idx))
                .unwrap_or(&[]);

            // Orient the row and compute zero set
            let (chosen, zero_set) = orient_and_verify_row(
                &ctx,
                &mut out_row,
                &mut out_row_int,
                &mut dot_scratch,
                witness,
                to_near_zero,
                dd_edge_hints.and_then(|h| h.get(out_idx)),
                negates_dot,
                certificate_only,
                compute_zero_sets,
                mode,
                out_idx,
                eps,
            )?;

            let chosen = match chosen {
                Some(row) => row,
                None => match mode {
                    ResolveMode::Strict => unreachable!("strict mode should have returned error"),
                    ResolveMode::Partial => {
                        // Issue was already recorded in orient_and_verify_row
                        continue;
                    }
                },
            };

            // Verify output incidence if in strict mode
            if verify_output_incidence {
                let observed = zero_set
                    .as_ref()
                    .expect("zero_set must be computed when strict verification is enabled");
                if observed != witness {
                    verification_issues.push(VerificationIssue::OutputIncidenceMismatch {
                        output_row: out_idx,
                        expected: witness.clone(),
                        observed: observed.clone(),
                    });
                }
            }

            resolved_int_data.extend(out_row_int);
            resolved_data.extend(chosen);
            kept_output_rows.push(out_idx);
            if store_zero_sets {
                resolved_zero_sets
                    .push(zero_set.expect("zero_set must be collected when storing zero sets"));
            }
        }

        // Phase 4: Build output structures
        build_resolution_output(
            self,
            input,
            &int_input_rows,
            resolved_data,
            resolved_int_data,
            kept_output_rows,
            resolved_zero_sets,
            issues,
            verification_issues,
            redund_cols,
            redund_mask,
            poly_options,
            &options,
            mode,
            eps,
        )
    }
}

impl<N: Num, R: DualRepresentation> PolyhedronOutput<N, R> {
    fn validate_resolution_inputs<M: Num>(
        &self,
        incidence: &SetFamily,
    ) -> Result<(), ResolveError<M>> {
        if self.is_empty {
            return Err(ResolveError::StatusNotAllFound {
                status: ComputationStatus::RegionEmpty,
            });
        }
        if self.status != ComputationStatus::AllFound {
            return Err(ResolveError::StatusNotAllFound {
                status: self.status,
            });
        }
        if self.output.col_count() == 0 {
            return Err(ResolveError::CertificateShapeMismatch);
        }
        if incidence.family_size() != self.output.row_count() {
            return Err(ResolveError::CertificateShapeMismatch);
        }
        Ok(())
    }
}

fn solve_witness_normal<'a, M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'a, M, R>,
    witness: &RowSet,
    out_idx: usize,
    bareiss_scratch: &mut BareissSolveScratch<<M as Rat>::Int>,
    scratch_candidates: &mut Vec<Row>,
    eps: &impl Epsilon<M>,
) -> Option<SolvedRow<M>>
{
    let has_input_linearity = !ctx.input.linearity().is_empty();
    let mut solved: Option<SolvedRow<M>> = None;

    // Try DD-provided witness basis first (fast path)
    if ctx.witness_target_rank > 0
        && let Some(dd_basis) = ctx
            .dd_witness_bases
            .and_then(|bases| bases.get(out_idx))
            .filter(|basis| {
                basis.len() == ctx.witness_target_rank
                    && basis.iter().all(|&row| row < ctx.input.row_count())
            })
    {
        solved = try_solve_from_basis(
            ctx,
            dd_basis,
            bareiss_scratch,
            scratch_candidates,
            eps,
            has_input_linearity,
        );
    }

    // Fallback: solve from witness incidence
    if solved.is_none() {
        scratch_candidates.clear();
        scratch_candidates.extend(witness.iter().map(|row| row.as_index()));
        scratch_candidates.extend(ctx.input.linearity().iter().map(|id| id.as_index()));

        solved = try_solve_from_candidates(ctx, scratch_candidates, bareiss_scratch, eps);
    }

    // Final fallback: rational nullspace solver
    if solved.is_none() {
        solved = try_solve_nullspace_rational(ctx, witness, has_input_linearity, eps);
    }

    solved
}

fn try_solve_from_basis<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    dd_basis: &[Row],
    bareiss_scratch: &mut BareissSolveScratch<<M as Rat>::Int>,
    scratch_candidates: &mut Vec<Row>,
    eps: &impl Epsilon<M>,
    has_input_linearity: bool,
) -> Option<SolvedRow<M>>
{
    if !has_input_linearity {
        return solve_basis(ctx, dd_basis, bareiss_scratch, eps);
    }

    scratch_candidates.clear();
    scratch_candidates.extend(dd_basis.iter().copied());
    scratch_candidates.extend(ctx.input.linearity().iter().map(|id| id.as_index()));

    if scratch_candidates.len() == ctx.witness_target_rank {
        return solve_basis(ctx, scratch_candidates, bareiss_scratch, eps);
    }

    if scratch_candidates.len() > ctx.witness_target_rank
        && let Some(basis_rows) = select_row_basis_rows_int(
            ctx.int_input_rows,
            scratch_candidates,
            ctx.witness_target_rank,
            ctx.redund_mask,
        )
    {
        return solve_basis(ctx, &basis_rows, bareiss_scratch, eps);
    }

    None
}

fn try_solve_from_candidates<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    scratch_candidates: &[Row],
    bareiss_scratch: &mut BareissSolveScratch<<M as Rat>::Int>,
    eps: &impl Epsilon<M>,
) -> Option<SolvedRow<M>>
{
    if scratch_candidates.len() == ctx.witness_target_rank {
        return solve_basis(ctx, scratch_candidates, bareiss_scratch, eps);
    }

    if scratch_candidates.len() > ctx.witness_target_rank
        && let Some(basis_rows) = select_row_basis_rows_int(
            ctx.int_input_rows,
            scratch_candidates,
            ctx.witness_target_rank,
            ctx.redund_mask,
        )
    {
        return solve_basis(ctx, &basis_rows, bareiss_scratch, eps);
    }

    None
}

fn try_solve_nullspace_rational<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    witness: &RowSet,
    has_input_linearity: bool,
    eps: &impl Epsilon<M>,
) -> Option<SolvedRow<M>> {
    if !has_input_linearity {
        return ctx
            .input
            .storage()
            .solve_nullspace_1d_with_unit_cols(witness, ctx.redund_cols, eps)
            .map(SolvedRow::Rat);
    }

    let mut rows = witness.clone();
    for idx in ctx.input.linearity().iter() {
        rows.insert(idx);
    }
    ctx.input
        .storage()
        .solve_nullspace_1d_with_unit_cols(&rows, ctx.redund_cols, eps)
        .map(SolvedRow::Rat)
}

fn solve_basis<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    basis_rows: &[Row],
    bareiss_scratch: &mut BareissSolveScratch<<M as Rat>::Int>,
    eps: &impl Epsilon<M>,
) -> Option<SolvedRow<M>>
{
    solve_nullspace_1d_rows_with_unit_cols_bareiss_int(
        bareiss_scratch,
        ctx.int_input_rows,
        basis_rows,
        ctx.redund_cols,
        ctx.redund_mask,
    )
    .map(SolvedRow::Int)
    .or_else(|| {
        ctx.input
            .storage()
            .solve_nullspace_1d_rows_with_unit_cols(basis_rows, ctx.redund_cols, eps)
            .map(SolvedRow::Rat)
    })
}

type ConvertedSolvedRow<M> = (Option<Vec<M>>, Vec<<M as Rat>::Int>);

fn convert_solved_row<M: Rat>(
    solved: SolvedRow<M>,
    _eps: &impl Epsilon<M>,
) -> Result<ConvertedSolvedRow<M>, ResolveError<M>> {
    match solved {
        SolvedRow::Rat(row) => {
            let out_row_int =
                scaled_integer_vec::<M>(&row).ok_or(ResolveError::ConversionFailure)?;
            Ok((Some(row), out_row_int))
        }
        SolvedRow::Int(row) => Ok((None, row)),
    }
}

fn int_to_rat_vec<M: Rat>(int_vec: &[<M as Rat>::Int]) -> Vec<M> {
    let denom = <M as Rat>::Int::one();
    int_vec
        .iter()
        .cloned()
        .map(|numer| M::from_frac(numer, denom.clone()))
        .collect()
}

fn build_redund_mask(cols: usize, redund_cols: &[usize]) -> Vec<bool> {
    let mut redund_mask = vec![false; cols];
    for &col in redund_cols {
        if col < cols {
            redund_mask[col] = true;
        }
    }
    redund_mask
}

trait CertificateIncidenceCollector {
    fn clear(&mut self);
    fn push_row(&mut self, row_idx: Row, input_row_count: usize);
    fn finalize(&mut self);
}

impl CertificateIncidenceCollector for RowSet {
    #[inline(always)]
    fn clear(&mut self) {
        RowSet::clear(self);
    }

    #[inline(always)]
    fn push_row(&mut self, row_idx: Row, _input_row_count: usize) {
        self.insert(row_idx);
    }

    #[inline(always)]
    fn finalize(&mut self) {}
}

impl CertificateIncidenceCollector for Vec<Row> {
    #[inline(always)]
    fn clear(&mut self) {
        Vec::clear(self);
    }

    #[inline(always)]
    fn push_row(&mut self, row_idx: Row, input_row_count: usize) {
        if row_idx < input_row_count {
            self.push(row_idx);
        }
    }

    #[inline(always)]
    fn finalize(&mut self) {
        self.sort_unstable();
        self.dedup();
    }
}

type OrientedRow<M> = (Option<Vec<M>>, Option<RowSet>);

#[allow(clippy::too_many_arguments)]
fn orient_and_verify_row<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row: &mut Option<Vec<M>>,
    out_row_int: &mut [<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    witness: &RowSet,
    to_near_zero: &[Row],
    edge_hints: Option<&Vec<super::DdEdgeHint>>,
    negates_dot: bool,
    certificate_only: bool,
    compute_zero_sets: bool,
    mode: ResolveMode,
    out_idx: usize,
    eps: &impl Epsilon<M>,
) -> Result<OrientedRow<M>, ResolveError<M>> {
    let mut zero_set = compute_zero_sets.then(|| RowSet::new(ctx.input.row_count()));

    if certificate_only {
        let result = orient_certificate_only(
            ctx,
            out_row,
            out_row_int,
            dot_scratch,
            witness,
            to_near_zero,
            edge_hints,
            negates_dot,
            zero_set.as_mut(),
        )?;

        if let Some(row) = result {
            return Ok((Some(row), zero_set));
        }
        // Fall through to full scan
    }

    // Full scan orientation
    orient_full_scan(
        ctx,
        out_row,
        out_row_int,
        dot_scratch,
        negates_dot,
        zero_set.as_mut(),
        mode,
        out_idx,
        eps,
    )
    .map(|row| (row, zero_set))
}

#[allow(clippy::too_many_arguments)]
fn orient_certificate_only<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row: &mut Option<Vec<M>>,
    out_row_int: &mut [<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    witness: &RowSet,
    to_near_zero: &[Row],
    edge_hints: Option<&Vec<super::DdEdgeHint>>,
    negates_dot: bool,
    zero_set: Option<&mut RowSet>,
) -> Result<Option<Vec<M>>, ResolveError<M>>
{
    if !orient_certificate_only_common(
        ctx,
        out_row_int,
        dot_scratch,
        witness,
        to_near_zero,
        edge_hints,
        negates_dot,
        zero_set,
        |out_row_int| negate_row(out_row, out_row_int),
    )? {
        return Ok(None);
    }

    let row = out_row
        .take()
        .unwrap_or_else(|| int_to_rat_vec::<M>(out_row_int));
    Ok(Some(row))
}

#[inline(always)]
fn orient_certificate_probe_row<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row_int: &mut [<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    witness: &RowSet,
    row_idx: usize,
    negates_dot: bool,
    negate: &mut impl FnMut(&mut [<M as Rat>::Int]) -> Result<(), ResolveError<M>>,
) -> Result<Option<bool>, ResolveError<M>> {
    if row_idx >= ctx.input.row_count() || witness.contains(row_idx) {
        return Ok(None);
    }

    match orient_using_row(ctx, &*out_row_int, dot_scratch, row_idx, negates_dot)? {
        OrientationResult::Valid => Ok(Some(false)),
        OrientationResult::Undecided => {
            // FIXME: Consider continuing search on `Undecided` once correctness/perf is audited.
            Ok(Some(false))
        }
        OrientationResult::NeedsNegation => {
            negate(out_row_int)?;
            Ok(Some(false))
        }
        OrientationResult::Infeasible => Ok(Some(true)),
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn orient_certificate_only_common<M: Rat, R: DualRepresentation, C: CertificateIncidenceCollector>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row_int: &mut [<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    witness: &RowSet,
    to_near_zero: &[Row],
    edge_hints: Option<&Vec<super::DdEdgeHint>>,
    negates_dot: bool,
    collector: Option<&mut C>,
    mut negate: impl FnMut(&mut [<M as Rat>::Int]) -> Result<(), ResolveError<M>>,
) -> Result<bool, ResolveError<M>> {
    let mut need_full_scan = false;
    let mut did_orient = false;
    if let Some(hints_to) = edge_hints {
        'hint_orient: for hint in hints_to {
            for row_idx in [hint.from_witness, hint.entered_row].into_iter().flatten() {
                if let Some(needs_full) = orient_certificate_probe_row(
                    ctx,
                    out_row_int,
                    dot_scratch,
                    witness,
                    row_idx,
                    negates_dot,
                    &mut negate,
                )? {
                    did_orient = true;
                    need_full_scan = needs_full;
                    break 'hint_orient;
                }
            }
        }
    }

    if !need_full_scan && !did_orient {
        for row_id in witness.iter().complement() {
            let row_idx = row_id.as_index();
            if let Some(needs_full) = orient_certificate_probe_row(
                ctx,
                out_row_int,
                dot_scratch,
                witness,
                row_idx,
                negates_dot,
                &mut negate,
            )? {
                need_full_scan = needs_full;
                break;
            }
        }
    }

    if need_full_scan {
        return Ok(false);
    }

    let Some(collector) = collector else {
        return Ok(true);
    };

    let input_row_count = ctx.input.row_count();
    collector.clear();

    for row_id in witness.iter() {
        let row_idx = row_id.as_index();
        if dot_sign_int(ctx.int_input_rows, row_idx, out_row_int, dot_scratch)? != Sign::Zero {
            return Ok(false);
        }
        collector.push_row(row_idx, input_row_count);
    }

    if !to_near_zero.is_empty() {
        for &row_idx in to_near_zero {
            if row_idx < input_row_count
                && dot_sign_int(ctx.int_input_rows, row_idx, out_row_int, dot_scratch)? == Sign::Zero
            {
                collector.push_row(row_idx, input_row_count);
            }
        }
        collector.finalize();
    }

    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn orient_certificate_only_int<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row_int: &mut [<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    witness: &RowSet,
    to_near_zero: &[Row],
    edge_hints: Option<&Vec<super::DdEdgeHint>>,
    negates_dot: bool,
    vertices: &mut Vec<Row>,
) -> Result<bool, ResolveError<M>>
{
    orient_certificate_only_common(
        ctx,
        out_row_int,
        dot_scratch,
        witness,
        to_near_zero,
        edge_hints,
        negates_dot,
        Some(vertices),
        negate_int_row::<M>,
    )
}

#[inline(always)]
fn orient_using_row<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row_int: &[<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    row_idx: usize,
    negates_dot: bool,
) -> Result<OrientationResult, ResolveError<M>>
{
    let sign_pos = dot_sign_int(ctx.int_input_rows, row_idx, out_row_int, dot_scratch)?;
    if sign_pos == Sign::Zero {
        // Can't determine orientation from this row
        return Ok(OrientationResult::Undecided);
    }

    let kind = ctx
        .equality_kinds
        .get(row_idx)
        .copied()
        .ok_or(ResolveError::CertificateShapeMismatch)?;

    let sign_neg = if negates_dot {
        negate_sign(sign_pos)
    } else {
        sign_pos
    };

    let viol_pos = kind.violates_sign(sign_pos, ctx.options.relaxed);
    let viol_neg = kind.violates_sign(sign_neg, ctx.options.relaxed);

    match (viol_pos, viol_neg) {
        (false, true) => Ok(OrientationResult::Valid),
        (true, false) => Ok(OrientationResult::NeedsNegation),
        (false, false) => {
            if negates_dot {
                Ok(OrientationResult::Undecided)
            } else {
                Ok(OrientationResult::Valid)
            }
        }
        (true, true) => Ok(OrientationResult::Infeasible),
    }
}

#[allow(clippy::too_many_arguments)]
fn orient_full_scan<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row: &mut Option<Vec<M>>,
    out_row_int: &mut [<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    negates_dot: bool,
    mut zero_set: Option<&mut RowSet>,
    mode: ResolveMode,
    out_idx: usize,
    eps: &impl Epsilon<M>,
) -> Result<Option<Vec<M>>, ResolveError<M>>
{
    // Ensure we have rational values for error reporting
    let mut row = out_row
        .take()
        .unwrap_or_else(|| int_to_rat_vec::<M>(out_row_int));

    if let Some(zs) = zero_set.as_mut() {
        zs.clear();
    }

    let mut viol_pos: Option<(Row, InequalityKind, M)> = None;
    let mut viol_neg = false;

    for (row_idx, (constraint, kind)) in ctx.input.rows().zip(ctx.equality_kinds.iter()).enumerate()
    {
        let sign_pos = dot_sign_int(ctx.int_input_rows, row_idx, out_row_int, dot_scratch)?;

        if sign_pos == Sign::Zero
            && let Some(zs) = zero_set.as_deref_mut()
        {
            zs.insert(row_idx);
        }

        if viol_pos.is_none() && kind.violates_sign(sign_pos, ctx.options.relaxed) {
            let value = linalg::dot(constraint, &row);
            viol_pos = Some((row_idx, *kind, value));
        }

        let sign_neg = if negates_dot {
            negate_sign(sign_pos)
        } else {
            sign_pos
        };
        if !viol_neg && kind.violates_sign(sign_neg, ctx.options.relaxed) {
            viol_neg = true;
        }

        if viol_pos.is_some() && viol_neg {
            break;
        }
    }

    match (viol_pos, viol_neg) {
        (None, true) => Ok(Some(row)),
        (Some(_), false) => {
            for v in row.iter_mut() {
                *v = v.ref_neg();
            }
            for v in out_row_int.iter_mut() {
                v.neg_mut().ok().ok_or(ResolveError::ConversionFailure)?;
            }
            Ok(Some(row))
        }
        (None, false) => {
            canonicalize_sign(&mut row, eps);
            canonicalize_sign_int::<M>(out_row_int)?;
            Ok(Some(row))
        }
        (Some((constraint, kind, value)), true) => match mode {
            ResolveMode::Strict => Err(ResolveError::InfeasibleResolvedRow {
                output_row: out_idx,
                constraint,
                kind,
                value,
            }),
            ResolveMode::Partial => Ok(None), // Issue recorded elsewhere
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn orient_full_scan_int<M: Rat, R: DualRepresentation>(
    ctx: &RowResolutionContext<'_, M, R>,
    out_row_int: &mut [<M as Rat>::Int],
    dot_scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
    negates_dot: bool,
    vertices: &mut Vec<Row>,
    out_idx: usize,
    issues: &mut Vec<PartialResolveIssue<M>>,
    _eps: &impl Epsilon<M>,
) -> Result<bool, ResolveError<M>> {
    vertices.clear();

    let mut viol_pos: Option<(Row, InequalityKind, <M as Rat>::Int)> = None;
    let mut viol_neg = false;

    for (row_idx, kind) in ctx.equality_kinds.iter().copied().enumerate() {
        if row_idx >= ctx.input.row_count() {
            continue;
        }

        let sign_pos = dot_sign_int(ctx.int_input_rows, row_idx, out_row_int, dot_scratch)?;
        if sign_pos == Sign::Zero {
            vertices.push(row_idx);
        }

        if viol_pos.is_none() && kind.violates_sign(sign_pos, ctx.options.relaxed) {
            viol_pos = Some((row_idx, kind, dot_scratch.dot.clone()));
        }

        let sign_neg = if negates_dot {
            negate_sign(sign_pos)
        } else {
            sign_pos
        };
        if !viol_neg && kind.violates_sign(sign_neg, ctx.options.relaxed) {
            viol_neg = true;
        }

        if viol_pos.is_some() && viol_neg {
            break;
        }
    }

    match (viol_pos, viol_neg) {
        (None, true) => Ok(true),
        (Some(_), false) => {
            for v in out_row_int.iter_mut() {
                if v.is_zero() {
                    continue;
                }
                v.neg_mut().ok().ok_or(ResolveError::ConversionFailure)?;
            }
            Ok(true)
        }
        (None, false) => {
            canonicalize_sign_int::<M>(out_row_int)?;
            Ok(true)
        }
        (Some((constraint, kind, value)), true) => {
            let denom = <M as Rat>::Int::one();
            issues.push(PartialResolveIssue::InfeasibleResolvedRow {
                output_row: out_idx,
                constraint,
                kind,
                value: M::from_frac(value, denom),
            });
            Ok(false)
        }
    }
}

fn negate_row<M: Rat>(
    out_row: &mut Option<Vec<M>>,
    out_row_int: &mut [<M as Rat>::Int],
) -> Result<(), ResolveError<M>> {
    if let Some(row) = out_row.as_mut() {
        for v in row.iter_mut() {
            *v = v.ref_neg();
        }
    }
    negate_int_row::<M>(out_row_int)?;
    Ok(())
}

#[inline(always)]
fn negate_int_row<M: Rat>(out_row_int: &mut [<M as Rat>::Int]) -> Result<(), ResolveError<M>> {
    for v in out_row_int.iter_mut() {
        if v.is_zero() {
            continue;
        }
        v.neg_mut().ok().ok_or(ResolveError::ConversionFailure)?;
    }
    Ok(())
}

fn dot_sign_int<M: Rat>(
    int_input_rows: &IntRowMatrix<<M as Rat>::Int>,
    row_idx: usize,
    out_row_int: &[<M as Rat>::Int],
    scratch: &mut DotSignIntScratch<<M as Rat>::Int>,
) -> Result<Sign, ResolveError<M>> {
    let constraint_int = int_input_rows
        .row(row_idx)
        .ok_or(ResolveError::CertificateShapeMismatch)?;

    <M as Rat>::Int::assign_from(&mut scratch.dot, &scratch.zero);

    for (a, b) in constraint_int.iter().zip(out_row_int.iter()) {
        <M as Rat>::Int::assign_from(&mut scratch.tmp, a);
        scratch
            .tmp
            .mul_assign(b)
            .ok()
            .ok_or(ResolveError::ConversionFailure)?;
        scratch.dot += &scratch.tmp;
    }

    Ok(if scratch.dot.is_negative() {
        Sign::Negative
    } else if scratch.dot.is_positive() {
        Sign::Positive
    } else {
        Sign::Zero
    })
}

fn canonicalize_sign<M: Num>(vec: &mut [M], eps: &impl Epsilon<M>) {
    for v in vec.iter() {
        if eps.is_zero(v) {
            continue;
        }
        if eps.is_negative(v) {
            for vv in vec.iter_mut() {
                *vv = vv.ref_neg();
            }
        }
        break;
    }
}

fn canonicalize_sign_int<M: Rat>(vec: &mut [<M as Rat>::Int]) -> Result<(), ResolveError<M>> {
    for v in vec.iter() {
        if v.is_zero() {
            continue;
        }
        if v.is_negative() {
            for vv in vec.iter_mut() {
                if vv.is_zero() {
                    continue;
                }
                vv.neg_mut().ok().ok_or(ResolveError::ConversionFailure)?;
            }
        }
        break;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_resolution_output<N, M, R>(
    original: &PolyhedronOutput<N, R>,
    input: LpMatrix<M, R>,
    int_input_rows: &IntRowMatrix<<M as Rat>::Int>,
    resolved_data: Vec<M>,
    resolved_int_data: Vec<<M as Rat>::Int>,
    kept_output_rows: Vec<Row>,
    resolved_zero_sets: Vec<RowSet>,
    issues: Vec<PartialResolveIssue<M>>,
    mut verification_issues: Vec<VerificationIssue<M>>,
    redund_cols: Vec<usize>,
    redund_mask: Vec<bool>,
    poly_options: &PolyhedronOptions,
    options: &ResolveOptions,
    mode: ResolveMode,
    eps: &impl Epsilon<M>,
) -> Result<PreparedPartialResolveResult<M, R>, ResolveError<M>>
where
    N: Num,
    M: CoerceFrom<N> + Rat,
    R: DualRepresentation,
{
    let cols = input.col_count();
    let output_size = resolved_data.len() / cols;
    let requests = IncidenceRequests::from_options(poly_options);

    let int_output_rows = IntRowMatrix::new(output_size, cols, resolved_int_data)
        .ok_or(ResolveError::ConversionFailure)?;

    let mut output: LpMatrix<M, DualOf<R>> =
        LpMatrixBuilder::<M, DualOf<R>>::from_flat(output_size, cols, resolved_data)
            .with_linearity(RowSet::new(output_size))
            .with_row_vec(input.row_vec().to_vec())
            .with_objective(input.objective())
            .build();

    let store_zero_sets = requests.wants_any_incidence()
        || (mode == ResolveMode::Strict
            && original.input_incidence.is_some()
            && original.redundant_rows.is_some()
            && original.dominant_rows.is_some());

    let has_resolved_incidence = store_zero_sets && output_size > 0;
    let mut resolved_incidence = if has_resolved_incidence {
        let mut builder = SetFamily::builder(output_size, input.row_count());
        for (idx, set) in resolved_zero_sets.into_iter().enumerate() {
            builder.replace_set(idx, set);
        }
        Some(builder.build())
    } else {
        None
    };

    // Verify input incidence in strict mode
    if mode == ResolveMode::Strict && output_size > 0 {
        verify_input_incidence(
            original,
            &resolved_incidence,
            &input,
            output_size,
            &mut verification_issues,
        );
    }

    if mode == ResolveMode::Strict && !verification_issues.is_empty() {
        return Err(ResolveError::VerificationFailed {
            issues: verification_issues,
        });
    }

    // Preserve output linearity markers
    let mut output_linearity = RowSet::new(output_size);
    for (new_idx, &orig_idx) in kept_output_rows.iter().enumerate() {
        if original.output.linearity().contains(orig_idx) {
            output_linearity.insert(new_idx);
        }
    }
    output = output.with_linearity_rows(&output_linearity);

    // Build adjacency
    let mut adjacency_profile = poly_options
        .profile_adjacency
        .then(|| Box::new(AdjacencyBuildProfile::default()));

    let adjacency = build_output_adjacency(
        original,
        &input,
        &output,
        &resolved_incidence,
        &kept_output_rows,
        output_size,
        requests,
        options,
        mode,
        adjacency_profile.as_deref_mut(),
        eps,
    );

    let incidence_for_storage = if output_size == 0 || !requests.build_output_incidence {
        None
    } else {
        resolved_incidence.take()
    };

    let (input_incidence, redundant_rows, dominant_rows, input_adjacency) =
        build_input_incidence_artifacts::<R>(
            &incidence_for_storage,
            output_size,
            input.row_count(),
            input.linearity(),
            requests,
            output_size == 0,
        );

    let homogeneous = input.is_homogeneous(eps);
    let dimension = if homogeneous {
        input.col_count()
    } else {
        input.col_count().saturating_sub(1)
    };

    let cost_vector = input.row_vec().to_vec();
    let status = match mode {
        ResolveMode::Strict => ComputationStatus::AllFound,
        ResolveMode::Partial => ComputationStatus::InProgress,
    };

    Ok(PreparedPartialResolveResult {
        partial: PartialResolveResult {
            poly: PolyhedronOutput::<M, R> {
                representation: R::KIND,
                homogeneous,
                dimension,
                input,
                output,
                equality_kinds: original.equality_kinds.clone(),
                linearity_dimension: original.linearity_dimension,
                output_size,
                incidence: incidence_for_storage,
                adjacency,
                input_incidence,
                input_adjacency,
                redundant_rows,
                dominant_rows,
                status,
                is_empty: false,
                cost_vector: Some(cost_vector),
                row_positions: original.row_positions.clone(),
                column_mapping: original.column_mapping.clone(),
                trace: None,
                repair_hints: None,
                adjacency_profile,
            },
            kept_output_rows,
            issues,
        },
        int_input_rows: int_input_rows.clone(),
        int_output_rows,
        redund_cols,
        redund_mask,
    })
}

fn verify_input_incidence<N: Num, M: Num, R: DualRepresentation>(
    original: &PolyhedronOutput<N, R>,
    resolved_incidence: &Option<SetFamily>,
    input: &LpMatrix<M, R>,
    output_size: usize,
    verification_issues: &mut Vec<VerificationIssue<M>>,
) {
    let verify_input_incidence = original.input_incidence.is_some()
        && original.redundant_rows.is_some()
        && original.dominant_rows.is_some();

    if !verify_input_incidence || output_size == 0 {
        return;
    }

    let resolved_incidence = match resolved_incidence.as_ref() {
        Some(inc) => inc,
        None => return,
    };

    let expected_in = resolved_incidence.transpose();

    if let Some(recorded) = original.input_incidence.as_ref()
        && let Some((idx, expected, observed)) = first_set_mismatch(&expected_in, recorded)
    {
        verification_issues.push(VerificationIssue::InputIncidenceMismatch {
            input_row: idx,
            expected,
            observed,
        });
    }

    let (expected_redundant, expected_dominant) = if R::KIND == RepresentationKind::Generator {
        (
            RowSet::new(input.row_count()),
            RowSet::new(input.row_count()),
        )
    } else {
        expected_in.classify_input_incidence(input.linearity(), output_size)
    };

    if let Some(recorded) = original.redundant_rows.as_ref()
        && expected_redundant != *recorded
    {
        verification_issues.push(VerificationIssue::RedundantRowsMismatch {
            expected: expected_redundant,
            observed: recorded.clone(),
        });
    }
    if let Some(recorded) = original.dominant_rows.as_ref()
        && expected_dominant != *recorded
    {
        verification_issues.push(VerificationIssue::DominantRowsMismatch {
            expected: expected_dominant,
            observed: recorded.clone(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn build_output_adjacency<N: Num, M: Num, R: DualRepresentation>(
    original: &PolyhedronOutput<N, R>,
    input: &LpMatrix<M, R>,
    output: &LpMatrix<M, DualOf<R>>,
    resolved_incidence: &Option<SetFamily>,
    kept_output_rows: &[Row],
    output_size: usize,
    requests: IncidenceRequests,
    options: &ResolveOptions,
    mode: ResolveMode,
    adjacency_profile: Option<&mut AdjacencyBuildProfile>,
    eps: &impl Epsilon<M>,
) -> Option<SetFamily> {
    if !requests.build_output_adjacency || output_size == 0 {
        return None;
    }

    let partial_certificate_only =
        mode == ResolveMode::Partial && options.partial_use_certificate_only;

    // Fast path: remap original adjacency in certificate-only mode
    if partial_certificate_only
        && let Some(remapped) = remap_adjacency(original, kept_output_rows, output_size)
    {
        return Some(remapped);
    }

    // Build slack incidence for non-homogeneous H-reps
    let slack_incidence = build_slack_incidence_for_adjacency(
        input,
        output,
        resolved_incidence,
        output_size,
        requests,
        eps,
    );

    let incidence = slack_incidence.as_ref().or(resolved_incidence.as_ref())?;

    let active_rows = RowSet::all(incidence.set_capacity());
    let input_rank = input
        .col_count()
        .saturating_sub(original.linearity_dimension);
    let candidate_edges = build_candidate_edges(original, kept_output_rows);
    build_adjacency(
        incidence,
        output.linearity(),
        &active_rows,
        input_rank,
        candidate_edges.as_deref(),
        false,
        adjacency_profile,
    )
}

fn remap_adjacency<N: Num, R: DualRepresentation>(
    original: &PolyhedronOutput<N, R>,
    kept_output_rows: &[Row],
    output_size: usize,
) -> Option<SetFamily> {
    let original_adj = original
        .adjacency
        .as_ref()
        .filter(|sf| sf.family_size() == original.output.row_count())?;

    let mut orig_to_new: Vec<Option<usize>> = vec![None; original.output.row_count()];
    for (new_idx, &orig_idx) in kept_output_rows.iter().enumerate() {
        if orig_idx < orig_to_new.len() {
            orig_to_new[orig_idx] = Some(new_idx);
        }
    }

    let mut builder = SetFamily::builder(output_size, output_size);
    for (orig_i, neighbors) in original_adj.sets().iter().enumerate() {
        let Some(new_i) = orig_to_new.get(orig_i).copied().flatten() else {
            continue;
        };
        for orig_j in neighbors.iter().map(|j| j.as_index()) {
            let Some(new_j) = orig_to_new.get(orig_j).copied().flatten() else {
                continue;
            };
            if new_i != new_j {
                builder.insert_into_set(new_i, RowId::new(new_j));
            }
        }
    }
    Some(builder.build())
}

fn build_slack_incidence_for_adjacency<M: Num, R: DualRepresentation>(
    input: &LpMatrix<M, R>,
    output: &LpMatrix<M, DualOf<R>>,
    resolved_incidence: &Option<SetFamily>,
    output_size: usize,
    requests: IncidenceRequests,
    eps: &impl Epsilon<M>,
) -> Option<SetFamily> {
    if output_size == 0
        || !requests.build_output_adjacency
        || R::KIND != RepresentationKind::Inequality
        || input.is_homogeneous(eps)
    {
        return None;
    }

    let base = resolved_incidence.as_ref()?;
    let input_rows = input.row_count();
    let mut builder = SetFamily::builder(output_size, input_rows + 1);

    for out_idx in 0..output_size {
        let base_set = base
            .set(out_idx)
            .cloned()
            .unwrap_or_else(|| RowSet::new(input_rows));
        let mut set = base_set;
        set.resize(input_rows + 1);
        if output
            .row(out_idx)
            .and_then(|row| row.first())
            .is_none_or(|v| eps.is_zero(v))
        {
            set.insert(input_rows);
        }
        builder.replace_set(out_idx, set);
    }
    Some(builder.build())
}

fn build_candidate_edges<N: Num, R: DualRepresentation>(
    original: &PolyhedronOutput<N, R>,
    kept_output_rows: &[Row],
) -> Option<Vec<(usize, usize)>> {
    let original_adj = original
        .adjacency
        .as_ref()
        .filter(|sf| sf.family_size() == original.output.row_count())?;

    let mut orig_to_new: Vec<Option<usize>> = vec![None; original.output.row_count()];
    for (new_idx, &orig_idx) in kept_output_rows.iter().enumerate() {
        if orig_idx < orig_to_new.len() {
            orig_to_new[orig_idx] = Some(new_idx);
        }
    }

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (orig_i, neighbors) in original_adj.sets().iter().enumerate() {
        let Some(new_i) = orig_to_new.get(orig_i).copied().flatten() else {
            continue;
        };
        for orig_j in neighbors.iter().map(|j| j.as_index()) {
            if orig_i >= orig_j {
                continue;
            }
            let Some(new_j) = orig_to_new.get(orig_j).copied().flatten() else {
                continue;
            };
            edges.push((new_i, new_j));
        }
    }

    (!edges.is_empty()).then_some(edges)
}

fn first_set_mismatch(lhs: &SetFamily, rhs: &SetFamily) -> Option<(usize, RowSet, RowSet)> {
    if lhs.family_size() != rhs.family_size() {
        return Some((
            usize::MAX,
            RowSet::new(lhs.set_capacity()),
            RowSet::new(rhs.set_capacity()),
        ));
    }
    if lhs.set_capacity() != rhs.set_capacity() {
        return Some((
            usize::MAX,
            RowSet::new(lhs.set_capacity()),
            RowSet::new(rhs.set_capacity()),
        ));
    }
    for idx in 0..lhs.family_size() {
        let expected = lhs
            .set(idx)
            .cloned()
            .unwrap_or_else(|| RowSet::new(lhs.set_capacity()));
        let observed = rhs
            .set(idx)
            .cloned()
            .unwrap_or_else(|| RowSet::new(rhs.set_capacity()));
        if expected != observed {
            return Some((idx, expected, observed));
        }
    }
    None
}
