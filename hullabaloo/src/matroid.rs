use calculo::linalg;
use calculo::num::{Epsilon, Num, Sign};

/// Errors that can occur during matroid computations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatroidError {
    /// The matroid ground set is too large for the requested computation.
    TooLarge { num_elements: usize },
    /// A subset has rank exceeding the total rank, indicating numerical issues or invalid input.
    RankInconsistent {
        subset_rank: usize,
        total_rank: usize,
    },
}

impl std::error::Error for MatroidError {}

impl std::fmt::Display for MatroidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLarge { num_elements } => write!(
                f,
                "matroid ground set too large for characteristic polynomial (n={num_elements})"
            ),
            Self::RankInconsistent {
                subset_rank,
                total_rank,
            } => write!(
                f,
                "subset rank exceeds total rank (subset_rank={subset_rank}, total_rank={total_rank})"
            ),
        }
    }
}

const MAX_CHAR_POLY_GROUND_SET: usize = 22; // safe default (2^22 subsets)

fn two_rows_mut<T>(rows: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    debug_assert_ne!(i, j, "rows must be distinct");
    if i < j {
        let (left, right) = rows.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = rows.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

fn gaussian_elimination_rank<N: Num>(m: &mut [Vec<N>], eps: &impl Epsilon<N>) -> usize {
    let rows = m.len();
    if rows == 0 {
        return 0;
    }
    let cols = m[0].len();
    if cols == 0 {
        return 0;
    }

    let mut rank = 0usize;
    let mut row = 0usize;

    for col in 0..cols {
        if row == rows {
            break;
        }

        let mut pivot_row = None;
        let mut best_abs = None;
        for (i, m_row) in m.iter().enumerate().skip(row) {
            let val = m_row[col].abs();
            if eps.is_zero(&val) {
                continue;
            }
            let better = best_abs
                .as_ref()
                .is_none_or(|b| val.partial_cmp(b).is_some_and(|o| o.is_gt()));
            if better {
                best_abs = Some(val);
                pivot_row = Some(i);
            }
        }

        let Some(pivot_row) = pivot_row else {
            continue;
        };
        if pivot_row != row {
            m.swap(pivot_row, row);
        }

        let pivot = m[row][col].clone();
        let inv_pivot = N::one().ref_div(&pivot);
        for x in &mut m[row][col..] {
            *x = x.ref_mul(&inv_pivot);
        }

        for i in 0..rows {
            if i == row {
                continue;
            }
            let (pivot_row, target_row) = two_rows_mut(m, row, i);
            let factor = target_row[col].clone();
            if eps.is_zero(&factor.abs()) {
                continue;
            }
            let pivot_row = pivot_row.as_slice();
            for (x, p) in target_row[col..].iter_mut().zip(&pivot_row[col..]) {
                linalg::sub_mul_assign(x, &factor, p);
            }
        }

        rank += 1;
        row += 1;
    }

    rank
}

fn full_row_rank<N: Num>(matrix: &[Vec<N>], eps: &impl Epsilon<N>) -> (Vec<Vec<N>>, usize) {
    let mut m = matrix.to_vec();
    let rank = gaussian_elimination_rank(&mut m, eps);
    m.truncate(rank);
    (m, rank)
}

fn rank_of_subset<N: Num>(matrix: &[Vec<N>], cols: &[usize], eps: &impl Epsilon<N>) -> usize {
    if cols.is_empty() {
        return 0;
    }
    if matrix.is_empty() {
        return 0;
    }

    let mut sub = Vec::with_capacity(matrix.len());
    for row in matrix {
        let mut out = Vec::with_capacity(cols.len());
        for &col_idx in cols {
            out.push(row[col_idx].clone());
        }
        sub.push(out);
    }
    gaussian_elimination_rank(&mut sub, eps)
}

fn sign_i8<N: Num>(value: &N, eps: &impl Epsilon<N>) -> i8 {
    match eps.sign(value) {
        Sign::Negative => -1,
        Sign::Zero => 0,
        Sign::Positive => 1,
    }
}

fn determinant_sign<N: Num>(mut m: Vec<Vec<N>>, eps: &impl Epsilon<N>) -> i8 {
    let n = m.len();
    if n == 0 {
        return 1;
    }
    if m.iter().any(|row| row.len() != n) {
        return 0;
    }

    let mut sign = 1i8;

    for i in 0..n {
        let mut pivot_row = None;
        let mut best_abs = None;
        for (r, m_row) in m.iter().enumerate().skip(i) {
            let val = m_row[i].abs();
            if eps.is_zero(&val) {
                continue;
            }
            let better = best_abs
                .as_ref()
                .is_none_or(|b| val.partial_cmp(b).is_some_and(|o| o.is_gt()));
            if better {
                best_abs = Some(val);
                pivot_row = Some(r);
            }
        }

        let Some(pivot_row) = pivot_row else {
            return 0;
        };
        if pivot_row != i {
            m.swap(pivot_row, i);
            sign = -sign;
        }

        let pivot = m[i][i].clone();
        let pivot_sign = sign_i8(&pivot, eps);
        if pivot_sign == 0 {
            return 0;
        }
        sign *= pivot_sign;

        for r in (i + 1)..n {
            let (pivot_row, target_row) = two_rows_mut(&mut m, i, r);
            let factor = target_row[i].ref_div(&pivot);
            let pivot_row = pivot_row.as_slice();
            for (x, p) in target_row[(i + 1)..].iter_mut().zip(&pivot_row[(i + 1)..]) {
                linalg::sub_mul_assign(x, &factor, p);
            }
        }
    }

    sign
}

/// Oriented matroid represented by a matrix of full row rank.
///
/// - `rank` is the matroid rank r.
/// - `n` is the number of elements |E|.
/// - `matrix` has shape r x n, columns are the ground elements.
#[derive(Debug, Clone)]
pub struct LinearOrientedMatroid<N: Num> {
    rank: usize,
    n: usize,
    matrix: Vec<Vec<N>>,
}

impl<N: Num> LinearOrientedMatroid<N> {
    /// Build from a row matrix of shape m x n.
    ///
    /// Rows are row-reduced to full row rank and extra zero rows are dropped; the oriented matroid
    /// is unchanged by these row operations (up to global sign).
    ///
    /// # Panics
    ///
    /// Panics if the matrix is empty or has inconsistent row lengths.
    pub fn from_rows(matrix: Vec<Vec<N>>) -> Self {
        let first = matrix.first().expect("matroid matrix must be nonempty");
        let cols = first.len();
        assert!(
            cols > 0 && matrix.iter().all(|r| r.len() == cols),
            "matroid matrix must have consistent, positive row length"
        );

        let eps = N::default_eps();
        let (reduced, rank) = full_row_rank(&matrix, &eps);
        Self {
            rank,
            n: cols,
            matrix: reduced,
        }
    }

    /// Build from a list of column vectors, each of length m.
    ///
    /// # Panics
    ///
    /// Panics if there are no columns or columns have inconsistent lengths.
    pub fn from_columns(columns: Vec<Vec<N>>) -> Self {
        let first = columns
            .first()
            .expect("matroid must have at least one element");
        let rows = first.len();
        assert!(
            rows > 0 && columns.iter().all(|c| c.len() == rows),
            "matroid columns must have consistent, positive length"
        );

        let n = columns.len();
        let mut row_matrix: Vec<Vec<N>> = vec![vec![N::zero(); n]; rows];
        for (j, col) in columns.iter().enumerate() {
            for (i, v) in col.iter().enumerate() {
                row_matrix[i][j] = v.clone();
            }
        }

        Self::from_rows(row_matrix)
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn num_elements(&self) -> usize {
        self.n
    }

    pub fn matrix(&self) -> &[Vec<N>] {
        &self.matrix
    }

    pub fn chirotope(&self, subset: &[usize]) -> i8 {
        assert_eq!(
            subset.len(),
            self.rank,
            "chirotope is defined only on r-element subsets"
        );

        let eps = N::default_eps();
        let mut sub = Vec::with_capacity(self.rank);
        for row in &self.matrix {
            let mut out = Vec::with_capacity(self.rank);
            for &col_idx in subset {
                out.push(row[col_idx].clone());
            }
            sub.push(out);
        }

        determinant_sign(sub, &eps)
    }

    pub fn rank_of_subset(&self, subset: &[usize]) -> usize {
        let eps = N::default_eps();
        rank_of_subset(&self.matrix, subset, &eps)
    }

    pub fn characteristic_polynomial(&self) -> Result<CharacteristicPolynomial, MatroidError> {
        let n = self.n;
        if n > MAX_CHAR_POLY_GROUND_SET {
            return Err(MatroidError::TooLarge { num_elements: n });
        }

        let eps = N::default_eps();
        let all: Vec<usize> = (0..n).collect();
        let r_total = rank_of_subset(&self.matrix, &all, &eps);

        let mut coeffs = vec![0i64; r_total + 1];

        let total_subsets = 1usize << n;
        let mut indices = Vec::with_capacity(n);
        for mask in 0..total_subsets {
            let subset_size = mask.count_ones() as usize;
            indices.clear();
            for e in 0..n {
                if (mask >> e) & 1 == 1 {
                    indices.push(e);
                }
            }
            let r_s = rank_of_subset(&self.matrix, &indices, &eps);
            if r_s > r_total {
                return Err(MatroidError::RankInconsistent {
                    subset_rank: r_s,
                    total_rank: r_total,
                });
            }
            let exp = r_total - r_s;
            let sign = if subset_size.is_multiple_of(2) { 1 } else { -1 };
            coeffs[exp] += sign;
        }

        Ok(CharacteristicPolynomial {
            coefficients: coeffs,
        })
    }
}

/// chi_M(t) = sum_k c_k t^k with integer coefficients.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacteristicPolynomial {
    pub coefficients: Vec<i64>,
}

impl CharacteristicPolynomial {
    pub fn degree(&self) -> usize {
        assert!(
            !self.coefficients.is_empty(),
            "characteristic polynomial must have at least one coefficient"
        );
        self.coefficients.len() - 1
    }
}

impl std::fmt::Display for CharacteristicPolynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;

        for (k, &c) in self.coefficients.iter().enumerate().rev() {
            if c == 0 {
                continue;
            }

            let sign = if c < 0 { '-' } else { '+' };
            let abs_c = c.abs();

            if first {
                if sign == '-' {
                    write!(f, "-")?;
                }
            } else {
                write!(f, " {} ", sign)?;
            }

            match k {
                0 => write!(f, "{abs_c}")?,
                1 => {
                    if abs_c == 1 {
                        write!(f, "t")?;
                    } else {
                        write!(f, "{abs_c} t")?;
                    }
                }
                _ => {
                    if abs_c == 1 {
                        write!(f, "t^{k}")?;
                    } else {
                        write!(f, "{abs_c} t^{k}")?;
                    }
                }
            }

            first = false;
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}
