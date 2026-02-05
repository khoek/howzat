//! Integer arithmetic utilities for exact polyhedron resolution.
//!
//! This module provides integer-based linear algebra operations used during certificate
//! resolution to avoid floating-point precision issues.

use crate::matrix::LpMatrix;
use calculo::num::{Int, Rat};
use hullabaloo::types::{Representation, Row};

/// A row-major matrix of integers, used for exact arithmetic during resolution.
#[derive(Clone, Debug)]
pub struct IntRowMatrix<Z: Int> {
    rows: usize,
    cols: usize,
    data: Vec<Z>,
}

impl<Z: Int> IntRowMatrix<Z> {
    pub fn new(rows: usize, cols: usize, data: Vec<Z>) -> Option<Self> {
        if cols == 0 && rows > 0 {
            return None;
        }
        if data.len() != rows.checked_mul(cols)? {
            return None;
        }
        Some(Self { rows, cols, data })
    }

    pub fn row_count(&self) -> usize {
        self.rows
    }

    pub fn col_count(&self) -> usize {
        self.cols
    }

    pub fn row(&self, row: Row) -> Option<&[Z]> {
        let row_start = row.checked_mul(self.cols)?;
        self.data.get(row_start..row_start + self.cols)
    }

    pub fn row_mut(&mut self, row: Row) -> Option<&mut [Z]> {
        let row_start = row.checked_mul(self.cols)?;
        self.data.get_mut(row_start..row_start + self.cols)
    }
}

/// Pre-allocated scratch space for the Bareiss elimination algorithm.
pub(crate) struct BareissSolveScratch<Z: Int> {
    pub a: Vec<Z>,
    pub b: Vec<Z>,
    pub y: Vec<Z>,
    pub pivot_scratch: Z::PivotScratch,
}

impl<Z: Int> BareissSolveScratch<Z> {
    pub fn new(dim: usize) -> Self {
        let size = dim
            .checked_mul(dim)
            .expect("BareissSolveScratch matrix allocation overflow");
        Self {
            a: vec![Z::zero(); size],
            b: vec![Z::zero(); dim],
            y: vec![Z::zero(); dim],
            pivot_scratch: Z::PivotScratch::default(),
        }
    }

    pub fn dim(&self) -> usize {
        self.b.len()
    }
}

/// Solve `A * x = b` using Bareiss elimination, returning `det(A) * x`.
///
/// This uses exact integer arithmetic to avoid floating-point precision issues.
/// Returns `None` if the system is singular or if overflow/underflow occurs.
pub(crate) fn bareiss_solve_det_times_x_in_place<Z>(
    a: &mut [Z],
    b: &mut [Z],
    y: &mut [Z],
    pivot_scratch: &mut Z::PivotScratch,
) -> Option<Z>
where
    Z: Int,
{
    let n = b.len();
    if n == 0 {
        return None;
    }
    if a.len() != n.checked_mul(n)? || y.len() != n {
        return None;
    }

    let mut det_prev = Z::one();

    for r in 0..n {
        let row_split = (r + 1).checked_mul(n)?;
        let (a_head, a_tail) = a.split_at_mut(row_split);
        let (b_head, b_tail) = b.split_at_mut(r + 1);

        let pivot_row_start = r.checked_mul(n)?;
        let pivot_row = a_head.get(pivot_row_start..pivot_row_start + n)?;
        let pivot = pivot_row.get(r)?;
        if pivot.is_zero() {
            return None;
        }

        for i in (r + 1)..n {
            let tail_row = i.checked_sub(r + 1)?;
            let row_start = tail_row.checked_mul(n)?;
            let ais = a_tail.get(row_start + r)?.clone();
            let br = b_head.get(r)?.clone();

            for j in (r + 1)..n {
                let arj = pivot_row.get(j)?;
                let idx = row_start + j;
                Z::bareiss_update_in_place(
                    a_tail.get_mut(idx)?,
                    pivot,
                    &ais,
                    arj,
                    &det_prev,
                    pivot_scratch,
                )
                .ok()?;
            }

            let bi = b_tail.get_mut(tail_row)?;
            Z::bareiss_update_in_place(bi, pivot, &ais, &br, &det_prev, pivot_scratch).ok()?;
            *a_tail.get_mut(row_start + r)? = Z::zero();
        }

        Z::assign_from(&mut det_prev, pivot);
    }

    let detp = a[(n - 1) * n + (n - 1)].clone();
    if detp.is_zero() {
        return None;
    }

    // Back-substitution
    for yi in y.iter_mut() {
        *yi = Z::zero();
    }

    Z::assign_from(&mut y[n - 1], &b[n - 1]);
    for i in (0..(n - 1)).rev() {
        let mut rhs = b[i].clone();
        rhs.mul_assign(&detp).ok()?;
        for j in (i + 1)..n {
            let mut term = a[i * n + j].clone();
            term.mul_assign(&y[j]).ok()?;
            rhs -= &term;
        }

        let val = <Z as Int>::div_exact(&rhs, &a[i * n + i]).ok()?;
        Z::assign_from(&mut y[i], &val);
    }

    Some(detp)
}

/// Solve `A * X = B` using Bareiss elimination, returning `det(A) * X` in-place in `b`.
///
/// - `a` is an `n×n` row-major matrix.
/// - `b` is an `n×rhs_cols` row-major matrix of right-hand sides.
///
/// Returns `None` if the system is singular or if exact divisions fail.
pub(crate) fn bareiss_solve_det_times_matrix_in_place<Z>(
    a: &mut [Z],
    b: &mut [Z],
    rhs_cols: usize,
    pivot_scratch: &mut Z::PivotScratch,
) -> Option<Z>
where
    Z: Int,
{
    if rhs_cols == 0 {
        return None;
    }
    let b_len = b.len();
    let n = b_len.checked_div(rhs_cols)?;
    if n == 0 || n.checked_mul(rhs_cols)? != b_len {
        return None;
    }
    if a.len() != n.checked_mul(n)? {
        return None;
    }

    let mut det_prev = Z::one();

    for r in 0..n {
        // Pivoting: swap in a nonzero pivot if needed.
        let pivot_idx = r.checked_mul(n)?.checked_add(r)?;
        if a.get(pivot_idx)?.is_zero() {
            let mut swap_row = None;
            for i in (r + 1)..n {
                let idx = i.checked_mul(n)?.checked_add(r)?;
                if !a.get(idx)?.is_zero() {
                    swap_row = Some(i);
                    break;
                }
            }
            let Some(swap_row) = swap_row else {
                return None;
            };

            let row_r_start = r.checked_mul(n)?;
            let row_s_start = swap_row.checked_mul(n)?;
            for j in 0..n {
                a.swap(row_r_start + j, row_s_start + j);
            }

            let rhs_r_start = r.checked_mul(rhs_cols)?;
            let rhs_s_start = swap_row.checked_mul(rhs_cols)?;
            for j in 0..rhs_cols {
                b.swap(rhs_r_start + j, rhs_s_start + j);
            }
        }

        let row_split = (r + 1).checked_mul(n)?;
        let (a_head, a_tail) = a.split_at_mut(row_split);
        let row_split_rhs = (r + 1).checked_mul(rhs_cols)?;
        let (b_head, b_tail) = b.split_at_mut(row_split_rhs);

        let pivot_row_start = r.checked_mul(n)?;
        let pivot_row = a_head.get(pivot_row_start..pivot_row_start + n)?;
        let pivot = pivot_row.get(r)?;
        if pivot.is_zero() {
            return None;
        }

        let br_row_start = r.checked_mul(rhs_cols)?;
        let br_row = b_head.get(br_row_start..br_row_start + rhs_cols)?;

        for i in (r + 1)..n {
            let tail_row = i.checked_sub(r + 1)?;
            let row_start = tail_row.checked_mul(n)?;
            let ais = a_tail.get(row_start + r)?.clone();

            for j in (r + 1)..n {
                let arj = pivot_row.get(j)?;
                let idx = row_start + j;
                Z::bareiss_update_in_place(
                    a_tail.get_mut(idx)?,
                    pivot,
                    &ais,
                    arj,
                    &det_prev,
                    pivot_scratch,
                )
                .ok()?;
            }

            let rhs_start = tail_row.checked_mul(rhs_cols)?;
            let rhs_slice = b_tail.get_mut(rhs_start..rhs_start + rhs_cols)?;
            for (bi, br) in rhs_slice.iter_mut().zip(br_row.iter()) {
                Z::bareiss_update_in_place(bi, pivot, &ais, br, &det_prev, pivot_scratch).ok()?;
            }

            *a_tail.get_mut(row_start + r)? = Z::zero();
        }

        Z::assign_from(&mut det_prev, pivot);
    }

    let detp = a[(n - 1) * n + (n - 1)].clone();
    if detp.is_zero() {
        return None;
    }

    // Back-substitution: overwrite `b` with `det(A) * X`.
    for i in (0..n).rev() {
        let diag = a.get(i * n + i)?;
        let row_start = i * n;
        let rhs_start = i * rhs_cols;
        for col in 0..rhs_cols {
            if i == n - 1 {
                continue;
            }
            let mut rhs = b.get(rhs_start + col)?.clone();
            rhs.mul_assign(&detp).ok()?;
            for j in (i + 1)..n {
                let mut term = a.get(row_start + j)?.clone();
                term.mul_assign(b.get(j * rhs_cols + col)?).ok()?;
                rhs -= &term;
            }
            let val = <Z as Int>::div_exact(&rhs, diag).ok()?;
            Z::assign_from(b.get_mut(rhs_start + col)?, &val);
        }
    }

    Some(detp)
}

/// Solve for a 1-dimensional nullspace of the system defined by `rows` with unit columns at `unit_cols`.
///
/// This finds a vector `v` such that `M * v = 0` where `M` is formed from the specified rows,
/// with additional unit constraints forcing certain columns to zero.
pub(crate) fn solve_nullspace_1d_rows_with_unit_cols_bareiss_int<Z>(
    scratch: &mut BareissSolveScratch<Z>,
    int_input_rows: &IntRowMatrix<Z>,
    rows: &[Row],
    unit_cols: &[usize],
    redund_mask: &[bool],
) -> Option<Vec<Z>>
where
    Z: Int,
{
    let cols = int_input_rows.col_count();
    if cols == 0 {
        return None;
    }
    if redund_mask.len() != cols {
        return None;
    }
    for &row in rows {
        if row >= int_input_rows.row_count() {
            return None;
        }
    }
    for &col in unit_cols {
        if col >= cols {
            return None;
        }
    }

    let k = cols.checked_sub(1)?;
    if rows.len().checked_add(unit_cols.len())? != k {
        return None;
    }
    if scratch.dim() != k {
        return None;
    }

    let z0 = Z::zero();
    let z1 = Z::one();

    // Try each non-redundant, non-unit column as the free variable
    for free_col in 0..cols {
        if redund_mask[free_col] || unit_cols.contains(&free_col) {
            continue;
        }

        // Build the system matrix and RHS
        let mut system_row = 0usize;
        for &row_idx in rows {
            let src = int_input_rows.row(row_idx)?;

            Z::assign_from(&mut scratch.b[system_row], &src[free_col]);
            let mut out_col = 0usize;
            for (col, value) in src.iter().enumerate() {
                if col == free_col {
                    continue;
                }
                let idx = system_row * k + out_col;
                Z::assign_from(&mut scratch.a[idx], value);
                out_col += 1;
            }
            system_row += 1;
        }

        // Add unit constraints
        for &unit_col in unit_cols {
            Z::assign_from(&mut scratch.b[system_row], &z0);
            let mut out_col = 0usize;
            for col in 0..cols {
                if col == free_col {
                    continue;
                }
                let idx = system_row * k + out_col;
                if col == unit_col {
                    Z::assign_from(&mut scratch.a[idx], &z1);
                } else {
                    Z::assign_from(&mut scratch.a[idx], &z0);
                }
                out_col += 1;
            }
            system_row += 1;
        }

        if system_row != k {
            return None;
        }

        let Some(det) = bareiss_solve_det_times_x_in_place(
            &mut scratch.a,
            &mut scratch.b,
            &mut scratch.y,
            &mut scratch.pivot_scratch,
        ) else {
            continue;
        };

        if det.is_zero() {
            continue;
        }

        // Reconstruct the full solution vector
        let mut out: Vec<Z> = Vec::with_capacity(cols);
        for col in 0..cols {
            if col == free_col {
                out.push(det.clone());
                continue;
            }
            let idx = if col < free_col { col } else { col - 1 };
            let mut numer = scratch.y.get(idx)?.clone();
            numer.neg_mut().ok()?;
            out.push(numer);
        }
        return Some(out);
    }

    None
}

/// Select a subset of rows that form a basis of the specified rank.
///
/// Uses Gaussian elimination with GCD normalization to find linearly independent rows.
pub(crate) fn select_row_basis_rows_int<Z>(
    int_input_rows: &IntRowMatrix<Z>,
    candidates: &[Row],
    target_rank: usize,
    ignored_cols: &[bool],
) -> Option<Vec<Row>>
where
    Z: Int,
{
    let cols = int_input_rows.col_count();
    if cols == 0 {
        return None;
    }
    if target_rank == 0 {
        return Some(Vec::new());
    }
    if candidates.len() < target_rank {
        return None;
    }
    if ignored_cols.len() != cols {
        return None;
    }

    let mut basis: Vec<Vec<Z>> = Vec::with_capacity(target_rank);
    let mut pivots: Vec<usize> = Vec::with_capacity(target_rank);
    let mut chosen: Vec<Row> = Vec::with_capacity(target_rank);

    let z0 = Z::zero();
    let z1 = Z::one();

    let mut work: Vec<Z> = vec![Z::zero(); cols];
    for &row in candidates {
        let src = int_input_rows.row(row)?;

        work.clone_from_slice(src);
        for (col, ignored) in ignored_cols.iter().enumerate() {
            if *ignored {
                Z::assign_from(&mut work[col], &z0);
            }
        }

        // Reduce by existing basis rows
        for (basis_row, &pivot_col) in basis.iter().zip(pivots.iter()) {
            let pivot = work.get(pivot_col)?.clone();
            if pivot.is_zero() {
                continue;
            }
            let basis_pivot = basis_row.get(pivot_col)?.clone();
            if basis_pivot.is_zero() {
                continue;
            }

            let mut g = basis_pivot.clone();
            g.gcd_assign(&pivot).ok()?;
            if g.is_zero() {
                continue;
            }

            let mut a = basis_pivot;
            a.div_assign_exact(&g).ok()?;
            let mut b = pivot;
            b.div_assign_exact(&g).ok()?;

            for (c, ignored) in ignored_cols.iter().enumerate().skip(pivot_col) {
                if *ignored {
                    continue;
                }
                let mut left = work.get(c)?.clone();
                left.mul_assign(&a).ok()?;
                let mut right = basis_row.get(c)?.clone();
                right.mul_assign(&b).ok()?;
                left -= &right;
                Z::assign_from(work.get_mut(c)?, &left);
            }
        }

        // Find pivot column
        let pivot_col = (0..cols).find(|&c| !ignored_cols[c] && !work[c].is_zero());
        let Some(pivot_col) = pivot_col else {
            continue;
        };

        // Normalize by GCD
        let mut row_gcd: Option<Z> = None;
        for (col, value) in work.iter().enumerate() {
            if ignored_cols[col] || value.is_zero() {
                continue;
            }
            let abs = value.abs().ok()?;
            match row_gcd.as_mut() {
                None => row_gcd = Some(abs),
                Some(g) => g.gcd_assign(&abs).ok()?,
            }
        }
        if let Some(g) = row_gcd.filter(|g| *g != z0 && *g != z1) {
            for (col, value) in work.iter_mut().enumerate() {
                if ignored_cols[col] || value.is_zero() {
                    continue;
                }
                value.div_assign_exact(&g).ok()?;
            }
        }

        // Make pivot positive
        if work.get(pivot_col).is_some_and(|v| v.is_negative()) {
            for value in work.iter_mut() {
                if value.is_zero() {
                    continue;
                }
                value.neg_mut().ok()?;
            }
        }

        basis.push(work.clone());
        pivots.push(pivot_col);
        chosen.push(row);

        if chosen.len() == target_rank {
            return Some(chosen);
        }
    }

    None
}

/// Scale a rational matrix to integer form by finding a common denominator for each row.
pub(crate) fn scaled_integer_rows<M: Rat, R: Representation>(
    input: &LpMatrix<M, R>,
) -> Option<IntRowMatrix<<M as Rat>::Int>> {
    let cols = input.col_count();
    let rows = input.row_count();
    let mut data: Vec<<M as Rat>::Int> = Vec::with_capacity(rows.checked_mul(cols)?);

    for row in input.rows() {
        if row.len() != cols {
            return None;
        }
        data.extend(scaled_integer_vec(row)?);
    }

    IntRowMatrix::new(rows, cols, data)
}

/// Scale a rational vector to integer form using LCM of denominators.
pub(crate) fn scaled_integer_vec<M: Rat>(row: &[M]) -> Option<Vec<<M as Rat>::Int>> {
    let mut parts: Vec<(<M as Rat>::Int, <M as Rat>::Int)> = Vec::with_capacity(row.len());
    for v in row.iter().cloned() {
        let (mut numer, mut denom) = v.into_parts();
        if denom.is_negative() {
            denom.neg_mut().ok()?;
            numer.neg_mut().ok()?;
        }
        parts.push((numer, denom));
    }

    let z0 = <M as Rat>::Int::zero();
    let z1 = <M as Rat>::Int::one();

    let mut scale = z1.clone();
    for (_, denom) in &parts {
        scale.lcm_assign(denom).ok()?;
    }

    if scale == z1 {
        return Some(parts.into_iter().map(|(numer, _)| numer).collect());
    }

    let mut out: Vec<<M as Rat>::Int> = Vec::with_capacity(row.len());
    for (numer, denom) in parts {
        if numer.is_zero() {
            out.push(z0.clone());
            continue;
        }
        if denom == z1 {
            let mut scaled = numer;
            scaled.mul_assign(&scale).ok()?;
            out.push(scaled);
            continue;
        }

        let mut factor = scale.clone();
        factor.div_assign_exact(&denom).ok()?;
        let mut scaled = numer;
        scaled.mul_assign(&factor).ok()?;
        out.push(scaled);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_row_matrix_creation() {
        let data = vec![1i64, 2, 3, 4, 5, 6];
        let mat = IntRowMatrix::new(2, 3, data).expect("create matrix");
        assert_eq!(mat.row_count(), 2);
        assert_eq!(mat.col_count(), 3);
        assert_eq!(mat.row(0), Some(&[1i64, 2, 3][..]));
        assert_eq!(mat.row(1), Some(&[4i64, 5, 6][..]));
    }

    #[test]
    fn int_row_matrix_invalid_size() {
        let data = vec![1i64, 2, 3, 4, 5];
        assert!(IntRowMatrix::new(2, 3, data).is_none());
    }
}
