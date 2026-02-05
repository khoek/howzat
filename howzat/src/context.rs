use crate::HowzatError as Error;
use crate::lp::compare_positive_ratios;
use crate::lp::{LpObjective, LpProblem, LpResult, LpSolver, LpStatus};
use crate::matrix::{
    CanonicalizationResult, LinearityCanonicalization, LpMatrix, LpMatrixBuilder, compose_positions,
};
use crate::polyhedron::{RedundancyCertificate, RelativeInterior, RestrictedFaceWitness};
use calculo::linalg;
use calculo::num::{Epsilon, Num};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{
    ColSet, Inequality, Representation, RepresentationKind, Row, RowIndex, RowSet,
};

impl<N: Num, R: Representation> LpMatrix<N, R> {
    fn redundancy_status(
        &self,
        row: Row,
        eps: &impl Epsilon<N>,
    ) -> Result<(bool, Option<LpResult<N>>), Error> {
        let row_count = self.row_count();
        assert!(
            row < row_count,
            "row index {row} out of bounds (rows={row_count})"
        );
        let target = self.row(row).expect("row index within bounds");
        if self.representation() == RepresentationKind::Inequality
            && self
                .rows()
                .enumerate()
                .filter(|(idx, _)| *idx != row)
                .any(|(idx, candidate)| {
                    if self.linearity().contains(idx) {
                        return false;
                    }
                    candidate
                        .iter()
                        .zip(target.iter())
                        .skip(1)
                        .all(|(a, b)| eps.cmp(a, b).is_eq())
                        && if self.col_count() == 0 {
                            false
                        } else {
                            let c = candidate.first().expect("matrix row missing column 0");
                            let t = target.first().expect("matrix row missing column 0");
                            eps.cmp(c, t).is_le()
                        }
                })
        {
            return Ok((true, None));
        }
        if self.linearity().contains(row) {
            return Ok((false, None));
        }
        let build_lp = || -> Result<LpProblem<N>, Error> {
            match self.representation() {
                RepresentationKind::Inequality => LpProblem::from_redundancy_h(self, row, eps),
                RepresentationKind::Generator => LpProblem::from_redundancy_v(self, row),
            }
        };

        let solution = build_lp()?.solve(LpSolver::DualSimplex, eps);
        match solution.status() {
            LpStatus::PivotFailure => return Err(Error::NumericallyInconsistent),
            LpStatus::PivotLimitExceeded => return Err(Error::LpCycling),
            _ => {}
        };
        let redundant = match solution.status() {
            LpStatus::Optimal => !eps.is_negative(solution.optimal_value()),
            LpStatus::Inconsistent | LpStatus::DualInconsistent => true,
            _ => false,
        };
        Ok((redundant, Some(solution)))
    }

    pub fn redundant_row(
        &self,
        row: Row,
        eps: &impl Epsilon<N>,
    ) -> Result<Option<RedundancyCertificate<N>>, Error> {
        let (redundant, solution) = self.redundancy_status(row, eps)?;
        if redundant {
            let coefficients = solution
                .map(|sol| sol.primal().to_vec())
                .unwrap_or_else(Vec::new);
            return Ok(Some(RedundancyCertificate { coefficients }));
        }
        Ok(None)
    }

    pub fn redundant_rows(&self, eps: &impl Epsilon<N>) -> Result<RowSet, Error> {
        let mut redset = RowSet::new(self.row_count());
        let mut mcopy = self.clone();
        for i in (0..mcopy.row_count()).rev() {
            let (redundant, _) = mcopy.redundancy_status(i, eps)?;
            if redundant {
                redset.insert(i);
                mcopy = mcopy.without_row(i);
            }
        }
        Ok(redset)
    }

    pub fn strongly_redundant_row(
        &self,
        row: Row,
        eps: &impl Epsilon<N>,
    ) -> Result<Option<RedundancyCertificate<N>>, Error> {
        let (_, solution_opt) = self.redundancy_status(row, eps)?;
        let Some(solution) = solution_opt else {
            return Ok(None);
        };
        if solution.status() != LpStatus::Optimal {
            return Ok(None);
        }
        match self.representation() {
            RepresentationKind::Inequality => {
                if eps.is_positive(solution.optimal_value()) {
                    return Ok(Some(RedundancyCertificate {
                        coefficients: solution.primal().to_vec(),
                    }));
                }
                Ok(None)
            }
            RepresentationKind::Generator => {
                if eps.is_negative(solution.optimal_value()) {
                    return Ok(None);
                }
                let lp = LpProblem::from_strong_redundancy_v(self, row)?;
                let boundary_solution = lp.solve(LpSolver::DualSimplex, eps);
                if boundary_solution.status() == LpStatus::Optimal
                    && !eps.is_positive(boundary_solution.optimal_value())
                {
                    return Ok(Some(RedundancyCertificate {
                        coefficients: boundary_solution.primal().to_vec(),
                    }));
                }
                Ok(None)
            }
        }
    }

    pub fn strongly_redundant_rows(&self, eps: &impl Epsilon<N>) -> Result<RowSet, Error> {
        let mut redset = RowSet::new(self.row_count());
        let mut mcopy = self.clone();
        for i in (0..mcopy.row_count()).rev() {
            if mcopy.strongly_redundant_row(i, eps)?.is_some() {
                redset.insert(i);
                mcopy = mcopy.without_row(i);
            }
        }
        Ok(redset)
    }

    pub fn ray_shooting(
        &self,
        interior_point: &[N],
        ray: &[N],
        eps: &impl Epsilon<N>,
    ) -> Result<Option<Row>, Error> {
        let col_count = self.col_count();
        assert!(
            interior_point.len() == col_count,
            "interior point length mismatch (got {} expected {})",
            interior_point.len(),
            col_count
        );
        assert!(
            ray.len() == col_count,
            "ray length mismatch (got {} expected {})",
            ray.len(),
            col_count
        );
        let m = self.row_count();
        let d = self.col_count();
        if m == 0 || d == 0 {
            return Ok(None);
        }
        let rows = self.storage();

        // Track best row and its ratio t2/t1 without performing the division, along with t1 for tie-breaking.
        let mut best: Option<(Row, N, N)> = None;
        for i in 0..m {
            let row = &rows[i];

            // Homogeneous: x0 = 1, direction0 = 0; avoid cloning the input vectors.
            let (t1_tail, t2) = linalg::dot2(&row[1..d], &interior_point[1..d], &ray[1..d]);
            let t1 = row[0].ref_add(&t1_tail);

            if !eps.is_positive(&t1) {
                continue;
            }

            match &best {
                None => best = Some((i, t2.clone(), t1.clone())),
                Some((best_i, best_num, best_den)) => {
                    match compare_positive_ratios(&t2, &t1, best_num, best_den, eps) {
                        std::cmp::Ordering::Less => best = Some((i, t2.clone(), t1.clone())),
                        std::cmp::Ordering::Equal => {
                            let best_row = &rows[*best_i];
                            if lex_cmp_scaled(row, &t1, best_row, best_den, eps)
                                == std::cmp::Ordering::Less
                            {
                                best = Some((i, t2.clone(), t1.clone()));
                            }
                        }
                        std::cmp::Ordering::Greater => {}
                    }
                }
            }
        }
        Ok(best.map(|(i, _, _)| i))
    }

    pub fn implicit_linearity(
        &self,
        row: Row,
        eps: &impl Epsilon<N>,
    ) -> Result<Option<RedundancyCertificate<N>>, Error> {
        if self.linearity().contains(row) {
            return Ok(None);
        }
        let solution = match self.representation() {
            RepresentationKind::Inequality => LpProblem::from_redundancy_h(self, row, eps)?
                .with_objective(LpObjective::Maximize)
                .solve(LpSolver::DualSimplex, eps),
            RepresentationKind::Generator => LpProblem::from_redundancy_v(self, row)?
                .with_objective(LpObjective::Maximize)
                .solve(LpSolver::DualSimplex, eps),
        };
        if solution.status() == LpStatus::Optimal && eps.is_zero(solution.optimal_value()) {
            return Ok(Some(RedundancyCertificate {
                coefficients: solution.primal().to_vec(),
            }));
        }
        Ok(None)
    }

    pub fn implicit_linearity_rows(&self, eps: &impl Epsilon<N>) -> Result<RowSet, Error> {
        let mut set = self.linearity().clone();
        if self.row_count() == 0 {
            return Ok(set);
        }
        let lp = match self.representation() {
            RepresentationKind::Inequality => LpProblem::from_implicit_linearity_h(self, eps)?,
            RepresentationKind::Generator => LpProblem::from_implicit_linearity_v(self)?,
        };
        let lp_fallback = lp.clone();
        let mut solution = lp.solve(LpSolver::DualSimplex, eps);
        if solution.status() != LpStatus::Optimal {
            solution = lp_fallback.solve(LpSolver::CrissCross, eps);
        }
        let status = solution.status();
        let value = solution.optimal_value().clone();
        if status == LpStatus::PivotLimitExceeded {
            return Ok(set);
        }
        if status == LpStatus::Optimal && eps.is_negative(&value) {
            return Ok(RowSet::all(self.row_count()));
        }
        for i in self.linearity().iter().complement() {
            let i = i.as_index();
            if self.implicit_linearity(i, eps)?.is_some() {
                set.insert(i);
            }
        }
        Ok(set)
    }

    pub fn remove_implicit_linearity(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Result<LinearityCanonicalization<N, R>, Error> {
        let impl_rows = self.implicit_linearity_rows(eps)?;
        let (reduced, newpos) = self.submatrix_with_positions(&impl_rows);
        let (reduced, _) = reduced.shifted_linearity_up();
        Ok(LinearityCanonicalization::new(reduced, impl_rows, newpos))
    }

    pub fn canonicalize_linearity(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Result<LinearityCanonicalization<N, R>, Error> {
        let lin_rows = self.implicit_linearity_rows(eps)?;
        let m = self.clone().with_linearity_rows(&lin_rows);
        let ignored_rows = m.linearity().complement();
        let ignored_cols = ColSet::new(m.col_count());
        let rank = m.storage().rank(&ignored_rows, &ignored_cols, eps);
        let lin_basis = rank.row_basis.clone();
        let to_remove = m.linearity().difference(&lin_basis);
        let (reduced, newpos) = m.submatrix_with_positions(&to_remove);
        let (reduced, shift_pos) = reduced.shifted_linearity_up();
        let composed = compose_positions(&newpos, &shift_pos);
        Ok(LinearityCanonicalization::new(reduced, lin_rows, composed))
    }

    pub fn canonicalize(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Result<CanonicalizationResult<N, R>, Error> {
        let (m_lin, impl_lin, lin_pos) = self.canonicalize_linearity(eps)?.into_parts();
        let (ordered, order_pos, duplicates) = m_lin.canonical_row_order(eps);

        let mut redset = RowSet::new(self.row_count());
        lin_pos
            .iter()
            .enumerate()
            .filter(|&(_, &mapped)| mapped >= 0 && duplicates.contains(mapped as usize))
            .for_each(|(idx, _)| redset.insert(idx));

        let composed = compose_positions(&lin_pos, &order_pos);
        let (feasible, feas_pos, infeasible_removed) = ordered.prune_infeasible_rows(eps)?;
        composed
            .iter()
            .enumerate()
            .filter(|&(_, &mapped)| mapped >= 0 && infeasible_removed.contains(mapped as usize))
            .for_each(|(idx, _)| redset.insert(idx));

        let composed_feasible = compose_positions(&composed, &feas_pos);
        let redundant_in_ordered = feasible.redundant_rows(eps)?;
        if redundant_in_ordered.cardinality() == feasible.row_count() && feasible.row_count() > 0 {
            return Ok(CanonicalizationResult::new(
                feasible,
                impl_lin,
                redset,
                composed_feasible,
            ));
        }
        let (reduced, red_pos) = feasible.submatrix_with_positions(&redundant_in_ordered);
        let composed_feasible = compose_positions(&composed, &feas_pos);
        for (idx, &mapped) in composed_feasible.iter().enumerate() {
            if mapped >= 0 && redundant_in_ordered.contains(mapped as usize) {
                redset.insert(idx);
            }
        }

        let final_pos = compose_positions(&composed_feasible, &red_pos);
        Ok(CanonicalizationResult::new(
            reduced, impl_lin, redset, final_pos,
        ))
    }

    pub fn canonicalize_without_redundancy(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Result<CanonicalizationResult<N, R>, Error> {
        let (m_lin, impl_lin, lin_pos) = self.canonicalize_linearity(eps)?.into_parts();
        let (ordered, order_pos, duplicates) = m_lin.canonical_row_order(eps);

        let mut redset = RowSet::new(self.row_count());
        for (idx, &mapped) in lin_pos.iter().enumerate() {
            if mapped >= 0 && duplicates.contains(mapped as usize) {
                redset.insert(idx);
            }
        }

        let composed = compose_positions(&lin_pos, &order_pos);
        let (feasible, feas_pos, infeasible_removed) = ordered.prune_infeasible_rows(eps)?;
        for (idx, &mapped) in composed.iter().enumerate() {
            if mapped >= 0 && infeasible_removed.contains(mapped as usize) {
                redset.insert(idx);
            }
        }

        let final_pos = compose_positions(&composed, &feas_pos);
        Ok(CanonicalizationResult::new(
            feasible, impl_lin, redset, final_pos,
        ))
    }

    pub fn remove_redundancy(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Result<(LpMatrix<N, R>, RowSet, RowIndex), Error> {
        let redset = self.redundant_rows(eps).expect("redundancy check failed");
        let (reduced, newpos) = self.submatrix_with_positions(&redset);
        Ok((reduced, redset, newpos))
    }

    pub fn find_relative_interior(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Result<RelativeInterior<N>, Error> {
        let impl_lin = self.implicit_linearity_rows(eps)?;
        let nonlin = self.linearity().union(&impl_lin).complement();
        let witness =
            self.restricted_face_witness(&impl_lin, &nonlin, LpSolver::DualSimplex, eps)?;
        let ignored_cols = ColSet::new(self.col_count());
        let rank = self.storage().rank(&nonlin, &ignored_cols, eps);
        Ok(RelativeInterior {
            implicit_linearity: impl_lin,
            linearity_basis: rank.row_basis,
            lp_solution: witness.lp_solution,
            exists: witness.exists,
        })
    }

    pub fn restricted_face_exists(
        &self,
        equalities: &RowSet,
        strict_inequalities: &RowSet,
        solver: LpSolver,
        eps: &impl Epsilon<N>,
    ) -> Result<bool, Error> {
        self.restricted_face_witness(equalities, strict_inequalities, solver, eps)
            .map(|w| w.exists)
    }

    pub fn restricted_face_witness(
        &self,
        equalities: &RowSet,
        strict_inequalities: &RowSet,
        solver: LpSolver,
        eps: &impl Epsilon<N>,
    ) -> Result<RestrictedFaceWitness<N>, Error> {
        let sol =
            LpProblem::from_feasibility_restricted(self, equalities, strict_inequalities, eps)?
                .solve(solver, eps);
        let exists = sol.status() == LpStatus::Optimal && !eps.is_negative(sol.optimal_value());
        Ok(RestrictedFaceWitness {
            exists,
            lp_solution: Some(sol),
        })
    }

    pub fn adjacency(&self, eps: &impl Epsilon<N>) -> Result<SetFamily, Error> {
        let m = self.row_count();
        let mut builder = SetFamily::builder(m, m);
        for i in 0..m {
            if self.linearity().contains(i) {
                continue;
            }
            let mut linearity = self.linearity().clone();
            linearity.insert(i);
            let mc = self.clone().with_linearity(linearity.clone());
            let red = mc.redundant_rows(eps)?;
            for idx in linearity.iter() {
                if idx.as_index() < m {
                    builder.insert_into_set(i, idx);
                }
            }
            let mut adj = red.complement();
            adj.difference_inplace(&linearity);
            builder.replace_set(i, adj);
        }
        Ok(builder.build())
    }

    pub fn weak_adjacency(&self, eps: &impl Epsilon<N>) -> Result<SetFamily, Error> {
        let m = self.row_count();
        let mut builder = SetFamily::builder(m, m);
        for i in 0..m {
            if self.linearity().contains(i) {
                continue;
            }
            let mut linearity = self.linearity().clone();
            linearity.insert(i);
            let mc = self.clone().with_linearity(linearity.clone());
            let red = mc.strongly_redundant_rows(eps)?;
            for idx in linearity.iter() {
                if idx.as_index() < m {
                    builder.insert_into_set(i, idx);
                }
            }
            let mut adj = red.complement();
            adj.difference_inplace(&linearity);
            builder.replace_set(i, adj);
        }
        Ok(builder.build())
    }
}

/// Methods only available for H-representation (inequality) matrices.
impl<N: Num> LpMatrix<N, Inequality> {
    /// Redundancy detection using interior point and ray shooting.
    /// This algorithm is more efficient than LP-based redundancy for large systems.
    pub fn redundant_rows_via_shooting(&self, eps: &impl Epsilon<N>) -> Result<RowSet, Error> {
        let m = self.row_count();
        let d = self.col_count();
        let mut redset = RowSet::new(m);
        let mut rowflag: Vec<isize> = vec![0; m];

        let lp_int = LpProblem::from_feasibility(self, eps)?.make_interior_finding();
        let sol = lp_int.solve(LpSolver::DualSimplex, eps);
        if sol.status() != LpStatus::Optimal || !eps.is_positive(sol.optimal_value()) {
            return self.redundant_rows(eps);
        }
        let interior = sol.primal().to_vec();

        let mut working_rows: Vec<Vec<N>> = Vec::new();
        let mut working_linearity = RowSet::new(0);

        let build_matrix = |rows: &Vec<Vec<N>>, linearity: &RowSet| -> LpMatrix<N, Inequality> {
            LpMatrixBuilder::from_matrix(self)
                .with_rows(rows.clone())
                .with_linearity(linearity.clone())
                .build()
        };

        let mut direction = vec![N::zero(); d];
        for j in 1..d {
            direction.fill(N::zero());
            direction[j] = N::one();
            if let Some(hit) = self.ray_shooting(&interior, &direction, eps)?
                && rowflag[hit] <= 0
            {
                rowflag[hit] = (working_rows.len() + 1) as isize;
                working_rows.push(self.row(hit).expect("row index within bounds").to_vec());
                working_linearity.resize(working_rows.len());
            }
            direction[j] = -N::one();
            if let Some(hit) = self.ray_shooting(&interior, &direction, eps)?
                && rowflag[hit] <= 0
            {
                rowflag[hit] = (working_rows.len() + 1) as isize;
                working_rows.push(self.row(hit).expect("row index within bounds").to_vec());
                working_linearity.resize(working_rows.len());
            }
        }

        let mut i = 0usize;
        while i < m {
            if rowflag[i] != 0 {
                i += 1;
                continue;
            }
            working_rows.push(self.row(i).expect("row index within bounds").to_vec());
            working_linearity.resize(working_rows.len());

            let m1 = build_matrix(&working_rows, &working_linearity);
            let row_len = working_rows.len();
            debug_assert!(row_len > 0, "working rows must be non-empty");
            let last_row = row_len - 1;
            let (redundant, solution_opt) = m1.redundancy_status(last_row, eps)?;
            if redundant {
                redset.insert(i);
                rowflag[i] = -1;
                working_rows.pop();
                working_linearity.resize(working_rows.len());
                i += 1;
                continue;
            }

            let Some(solution) = solution_opt else {
                working_rows.pop();
                working_linearity.resize(working_rows.len());
                i += 1;
                continue;
            };
            if solution.status() != LpStatus::Optimal {
                working_rows.pop();
                working_linearity.resize(working_rows.len());
                i += 1;
                continue;
            }

            let mut shootdir: Vec<N> = Vec::with_capacity(d);
            let solution_primal = solution.primal();
            for k in 0..d {
                shootdir.push(solution_primal[k].clone() - interior[k].clone());
            }
            let ired = match self.ray_shooting(&interior, &shootdir, eps)? {
                Some(idx) => idx,
                None => {
                    working_rows.pop();
                    working_linearity.resize(working_rows.len());
                    i += 1;
                    continue;
                }
            };
            rowflag[ired] = working_rows.len() as isize;
            if let Some(row) = self.row(ired)
                && let Some(dest) = working_rows.last_mut()
            {
                dest.clone_from_slice(row);
            }
            if ired != i {
                rowflag[i] = working_rows.len() as isize;
            }
            i += 1;
        }
        Ok(redset)
    }
}

fn lex_cmp_scaled<N: Num>(
    lhs: &[N],
    lhs_den: &N,
    rhs: &[N],
    rhs_den: &N,
    eps: &impl Epsilon<N>,
) -> std::cmp::Ordering {
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        let left = l.ref_mul(rhs_den);
        let right = r.ref_mul(lhs_den);
        let ord = eps.cmp(&left, &right);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}
