use super::util::repair_sorted_pair_u32;
use super::{Error, Result};
use calculo::num::Int;

/// A fraction-free simplex tableau used by the reverse-search traversal.
///
/// Layout notes:
/// - Coefficients are stored row-major with fixed stride `(original_decision_vars + 1)`.
/// - The active tableau columns are `0..=decision_vars` (with `decision_vars <= original_decision_vars`).
/// - `basic_vars` / `cobasic_vars` are sorted by variable index; `basic_rows` / `cobasic_cols`
///   map those sorted positions to physical row/column locations in `coeffs`.
#[derive(Clone, Debug)]
pub(super) struct Tableau<Z: Int> {
    pub(super) coeffs: Vec<Z>,
    pub(super) constraint_count: usize,
    pub(super) storage_row_count: usize,
    pub(super) decision_vars: usize,
    pub(super) original_decision_vars: usize,
    pub(super) is_lex_min: bool,
    pub(super) depth: usize,
    pub(super) det: Z,
    pub(super) basic_vars: Vec<u32>,
    pub(super) cobasic_vars: Vec<u32>,
    pub(super) basic_rows: Vec<u32>,
    pub(super) cobasic_cols: Vec<u32>,
    pub(super) redundant_columns: Vec<usize>,
    pub(super) var_to_constraint: Vec<u32>,
}

impl<Z: Int> Tableau<Z> {
    pub(super) const NO_CONSTRAINT: u32 = u32::MAX;

    /// The current cobasis variables (excluding the fixed RHS variable).
    pub(super) fn cobasis_vars(&self) -> &[u32] {
        &self.cobasic_vars[..self.decision_vars]
    }

    #[inline(always)]
    pub(super) fn constraint_of_var(&self, var: usize) -> Option<usize> {
        let v = *self.var_to_constraint.get(var)?;
        (v != Self::NO_CONSTRAINT).then_some(v as usize)
    }

    #[inline(always)]
    pub(super) fn coeff(&self, row_loc: usize, col_loc: usize) -> &Z {
        let idx = self.index(row_loc, col_loc);
        &self.coeffs[idx]
    }

    #[inline(always)]
    pub(super) fn coeff_mut(&mut self, row_loc: usize, col_loc: usize) -> &mut Z {
        let idx = self.index(row_loc, col_loc);
        &mut self.coeffs[idx]
    }

    #[inline(always)]
    fn index(&self, row_loc: usize, col_loc: usize) -> usize {
        debug_assert!(row_loc <= self.storage_row_count, "row out of bounds");
        debug_assert!(col_loc <= self.original_decision_vars, "col out of bounds");
        row_loc * (self.original_decision_vars + 1) + col_loc
    }

    pub(super) fn new(constraint_count: usize, decision_vars: usize) -> Result<Self> {
        Self::new_with_storage_rows(constraint_count, decision_vars, constraint_count)
    }

    pub(super) fn new_with_storage_rows(
        constraint_count: usize,
        decision_vars: usize,
        storage_row_count: usize,
    ) -> Result<Self> {
        // The pivoting logic assumes the full constraint block is present.
        if storage_row_count < constraint_count {
            return Err(Error::DimensionTooLarge);
        }

        let max_var = constraint_count
            .checked_add(decision_vars)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::DimensionTooLarge)?;
        let _max_var_u32: u32 = max_var.try_into().map_err(|_| Error::DimensionTooLarge)?;

        let rows = storage_row_count
            .checked_add(1)
            .ok_or(Error::DimensionTooLarge)?;
        let cols = decision_vars
            .checked_add(1)
            .ok_or(Error::DimensionTooLarge)?;
        let alloc = rows.checked_mul(cols).ok_or(Error::DimensionTooLarge)?;

        let mut basic_vars = vec![0u32; constraint_count + 1];
        let mut basic_rows = vec![0u32; constraint_count + 1];
        basic_vars[0] = 0;
        basic_rows[0] = 0;
        for i in 1..=constraint_count {
            let bi: u32 = (decision_vars + i)
                .try_into()
                .map_err(|_| Error::DimensionTooLarge)?;
            basic_vars[i] = bi;
            basic_rows[i] = i.try_into().map_err(|_| Error::DimensionTooLarge)?;
        }

        let mut cobasic_vars = vec![0u32; decision_vars + 1];
        let mut cobasic_cols = vec![0u32; decision_vars + 1];
        for j in 0..decision_vars {
            cobasic_vars[j] = (j + 1).try_into().map_err(|_| Error::DimensionTooLarge)?;
            cobasic_cols[j] = (j + 1).try_into().map_err(|_| Error::DimensionTooLarge)?;
        }
        cobasic_vars[decision_vars] = (constraint_count + decision_vars + 1)
            .try_into()
            .map_err(|_| Error::DimensionTooLarge)?;
        cobasic_cols[decision_vars] = 0;

        let mut var_to_constraint = vec![Self::NO_CONSTRAINT; max_var + 1];
        for i in 1..=constraint_count {
            let slack_idx = decision_vars
                .checked_add(i)
                .ok_or(Error::DimensionTooLarge)?;
            var_to_constraint[slack_idx] =
                (i - 1).try_into().map_err(|_| Error::DimensionTooLarge)?;
        }

        Ok(Self {
            coeffs: vec![Z::zero(); alloc],
            constraint_count,
            storage_row_count,
            decision_vars,
            original_decision_vars: decision_vars,
            is_lex_min: false,
            depth: 0,
            det: Z::one(),
            basic_vars,
            cobasic_vars,
            basic_rows,
            cobasic_cols,
            redundant_columns: Vec::new(),
            var_to_constraint,
        })
    }

    /// Allocate an empty tableau with the same backing layout (used for cache reuse).
    pub(super) fn allocate_like(&self) -> Result<Self> {
        let mut out = Self::new_with_storage_rows(
            self.constraint_count,
            self.original_decision_vars,
            self.storage_row_count,
        )?;
        out.redundant_columns = Vec::with_capacity(self.redundant_columns.len());
        Ok(out)
    }

    /// Copy `src` into `self`, reusing existing allocations.
    pub(super) fn copy_from(&mut self, src: &Self) {
        debug_assert_eq!(self.constraint_count, src.constraint_count);
        debug_assert_eq!(self.storage_row_count, src.storage_row_count);
        debug_assert_eq!(self.original_decision_vars, src.original_decision_vars);
        debug_assert_eq!(self.coeffs.len(), src.coeffs.len());

        self.decision_vars = src.decision_vars;
        self.is_lex_min = src.is_lex_min;
        self.depth = src.depth;
        Z::assign_from(&mut self.det, &src.det);

        self.basic_vars.clone_from(&src.basic_vars);
        self.cobasic_vars.clone_from(&src.cobasic_vars);
        self.basic_rows.clone_from(&src.basic_rows);
        self.cobasic_cols.clone_from(&src.cobasic_cols);
        self.redundant_columns.clone_from(&src.redundant_columns);
        self.var_to_constraint.clone_from(&src.var_to_constraint);

        for (dst, v) in self.coeffs.iter_mut().zip(src.coeffs.iter()) {
            Z::assign_from(dst, v);
        }
    }

    /// Remove a cobasis entry at position `k`.
    ///
    /// This is used after detecting redundant columns (column dependencies). The underlying
    /// matrix storage keeps its original stride (`original_decision_vars + 1`).
    pub(super) fn remove_cobasis_pos(&mut self, k: usize) -> Result<()> {
        let constraint_count = self.constraint_count;
        let decision_vars = self.decision_vars;
        if decision_vars == 0 || k >= decision_vars {
            return Err(Error::InvariantViolation);
        }

        let cindex = self.cobasic_vars[k];
        let deloc = self.cobasic_cols[k] as usize;
        let del_index = cindex as usize;

        // Reduce basic variable indices above removed index.
        for i in 1..=constraint_count {
            if self.basic_vars[i] > cindex {
                self.basic_vars[i] -= 1;
            }
        }

        // Keep the variable→constraint mapping consistent with index decrements.
        if del_index < self.var_to_constraint.len() {
            let next = del_index.saturating_add(1);
            if next < self.var_to_constraint.len() {
                self.var_to_constraint.copy_within(next.., del_index);
            }
            if let Some(last) = self.var_to_constraint.last_mut() {
                *last = Self::NO_CONSTRAINT;
            }
        }

        // Shift down other cobasic variables and their physical column locations.
        for j in k..decision_vars {
            // j runs k..d-1, so j+1 is in-bounds.
            let next = self.cobasic_vars[j + 1];
            if next == 0 {
                return Err(Error::InvariantViolation);
            }
            self.cobasic_vars[j] = next - 1;
            self.cobasic_cols[j] = self.cobasic_cols[j + 1];
        }

        // Copy column `d` to `deloc` (if removing a non-last physical column).
        if deloc != decision_vars {
            for row_loc in 0..=self.storage_row_count {
                let src = self.index(row_loc, decision_vars);
                let dst = self.index(row_loc, deloc);
                // Safe because `src` and `dst` are distinct when `deloc != d`.
                unsafe {
                    let src_ptr = self.coeffs.as_ptr().add(src);
                    let dst_ptr = self.coeffs.as_mut_ptr().add(dst);
                    Z::assign_from(&mut *dst_ptr, &*src_ptr);
                }
            }

            // Reassign the physical location for the moved column.
            let mut found = false;
            for j in 0..=decision_vars {
                if self.cobasic_cols[j] as usize == decision_vars {
                    self.cobasic_cols[j] =
                        deloc.try_into().map_err(|_| Error::DimensionTooLarge)?;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(Error::InvariantViolation);
            }
        }

        self.decision_vars -= 1;
        Ok(())
    }

    /// Pivot from the current basis to a target cobasis.
    ///
    /// `target_cobasis` must list exactly `decision_vars` variable indices (excluding the
    /// fixed RHS variable).
    pub(super) fn pivot_to_cobasis(&mut self, target_cobasis: &[u32]) -> Result<()> {
        if target_cobasis.len() != self.decision_vars {
            return Err(Error::InvalidWarmStart);
        }

        let max_var = self
            .constraint_count
            .checked_add(self.decision_vars)
            .and_then(|v| v.checked_add(1))
            .ok_or(Error::DimensionTooLarge)?;

        let mut cobasic = vec![false; max_var + 1];
        for &v in target_cobasis {
            let idx = v as usize;
            if idx == 0 || idx > max_var {
                return Err(Error::InvalidWarmStart);
            }
            if cobasic[idx] {
                return Err(Error::InvalidWarmStart);
            }
            cobasic[idx] = true;
        }

        // The RHS variable is always stored at `C[d]` and must not be part of the target list.
        let rhs_var = self.cobasic_vars[self.decision_vars] as usize;
        if rhs_var <= max_var && cobasic[rhs_var] {
            return Err(Error::InvalidWarmStart);
        }

        // Swap flagged basic variables into the cobasis by pivoting with any unflagged cobasic
        // variable that has a nonzero pivot entry.
        let mut pivot_scratch = Z::PivotScratch::default();
        let mut i = self.constraint_count;
        while i > 0 {
            while cobasic[self.basic_vars[i] as usize] {
                let row_loc = self.basic_rows[i] as usize;
                let mut k_opt = None;
                for k in (0..self.decision_vars).rev() {
                    if cobasic[self.cobasic_vars[k] as usize] {
                        continue;
                    }
                    let col_loc = self.cobasic_cols[k] as usize;
                    if !self.coeff(row_loc, col_loc).is_zero() {
                        k_opt = Some(k);
                        break;
                    }
                }
                let Some(k) = k_opt else {
                    // No eligible entering variable => rank mismatch / invalid cobasis.
                    return Err(Error::InvalidWarmStart);
                };

                self.pivot_with_scratch(i, k, &mut pivot_scratch)?;
            }
            i -= 1;
        }

        // Verify: the current cobasis (excluding RHS) matches the requested one as a set.
        for k in 0..self.decision_vars {
            let idx = self.cobasic_vars[k] as usize;
            if idx > max_var || !cobasic[idx] {
                return Err(Error::InvalidWarmStart);
            }
        }
        Ok(())
    }

    pub(super) fn pivot_with_scratch(
        &mut self,
        bas: usize,
        cob: usize,
        scratch: &mut Z::PivotScratch,
    ) -> Result<(usize, usize)> {
        self.pivot_bareiss_with_scratch(bas, cob, scratch)?;
        self.swap_after_pivot(bas, cob)
    }

    /// Bareiss-style, fraction-free pivot update.
    ///
    /// `bas` and `cob` are indices into the sorted `basic_vars` / `cobasic_vars` arrays.
    fn pivot_bareiss_with_scratch(
        &mut self,
        bas: usize,
        cob: usize,
        scratch: &mut Z::PivotScratch,
    ) -> Result<()> {
        if bas >= self.basic_vars.len() || cob >= self.cobasic_vars.len() {
            return Err(Error::InvariantViolation);
        }
        let r = self.basic_rows[bas] as usize;
        let s = self.cobasic_cols[cob] as usize;
        if r > self.storage_row_count || s > self.decision_vars {
            return Err(Error::InvariantViolation);
        }

        let stride = self.original_decision_vars + 1;
        let idx_rs = r * stride + s;
        if self.coeffs[idx_rs].is_zero() {
            return Err(Error::InvariantViolation);
        }

        // We take the pivot entry out of the table (it's overwritten at the end anyway).
        let ars = std::mem::replace(&mut self.coeffs[idx_rs], Z::zero());
        if ars.is_negative() {
            self.det.neg_mut()?;
        }

        {
            let det_ref = &self.det;
            unsafe {
                let ptr = self.coeffs.as_mut_ptr();
                let pivot_row_ptr = ptr.add(r * stride);

                for i in 0..=self.storage_row_count {
                    if i == r {
                        continue;
                    }
                    let row_ptr = ptr.add(i * stride);
                    let ais_ref: &Z = &*row_ptr.add(s);

                    for j in 0..=self.decision_vars {
                        if j == s {
                            continue;
                        }
                        let aij: &mut Z = &mut *row_ptr.add(j);
                        let arj: &Z = &*pivot_row_ptr.add(j);
                        Z::bareiss_update_in_place(aij, &ars, ais_ref, arj, det_ref, scratch)?;
                    }
                }
            }
        }

        if ars.is_positive() {
            let row_start = r * stride;
            for j in 0..=self.decision_vars {
                if j == s {
                    continue;
                }
                let idx = row_start + j;
                if self.coeffs[idx].is_zero() {
                    continue;
                }
                self.coeffs[idx].neg_mut()?;
            }
        } else {
            for i in 0..=self.storage_row_count {
                if i == r {
                    continue;
                }
                let idx = i * stride + s;
                if self.coeffs[idx].is_zero() {
                    continue;
                }
                self.coeffs[idx].neg_mut()?;
            }
        }

        // det_new = |Ars| and pivot entry receives the old (signed) determinant.
        let mut det_new = ars;
        if det_new.is_negative() {
            det_new.neg_mut()?;
        }
        let old_det = std::mem::replace(&mut self.det, det_new);
        self.coeffs[idx_rs] = old_det;
        Ok(())
    }

    /// Swap entering/leaving variables and repair sorted order + row/column alignment.
    fn swap_after_pivot(&mut self, bas: usize, cob: usize) -> Result<(usize, usize)> {
        // `cob` ranges over decision-variable cobasis entries only (exclude fixed RHS at the end).
        if bas >= self.basic_vars.len() || cob >= self.decision_vars {
            return Err(Error::InvariantViolation);
        }

        let leaving_var = self.basic_vars[bas];
        let entering_var = self.cobasic_vars[cob];

        self.basic_vars[bas] = entering_var;
        repair_sorted_pair_u32(
            &mut self.basic_vars,
            &mut self.basic_rows,
            bas,
            self.constraint_count + 1,
        );

        self.cobasic_vars[cob] = leaving_var;
        // IMPORTANT: only reorder the decision-var portion (do not reorder the fixed RHS entry).
        repair_sorted_pair_u32(
            &mut self.cobasic_vars,
            &mut self.cobasic_cols,
            cob,
            self.decision_vars,
        );

        // Re-find the new positions (the lists are sorted, so linear scan is fine).
        let mut new_bas = 1usize;
        while new_bas < self.basic_vars.len() && self.basic_vars[new_bas] != entering_var {
            new_bas += 1;
        }
        let mut new_cob = 0usize;
        while new_cob < self.decision_vars && self.cobasic_vars[new_cob] != leaving_var {
            new_cob += 1;
        }
        if new_bas >= self.basic_vars.len() || new_cob >= self.decision_vars {
            return Err(Error::InvariantViolation);
        }
        Ok((new_bas, new_cob))
    }
}
