use calculo::linalg;
use calculo::num::{CoerceFrom, ConversionError, Epsilon, Num};
use calculo::util::strict_mul;

use crate::types::{ColSet, RowSet};

/// A small square matrix used for basis / pivot tracking.
#[derive(Clone, Debug)]
pub struct BasisMatrix<N: Num> {
    pub(crate) data: Vec<N>,
    dimension: usize,
    is_identity: bool,
}

impl<N: Num> BasisMatrix<N> {
    pub fn identity(dimension: usize) -> Self {
        let alloc = strict_mul(dimension, dimension, "basis matrix dimension overflow");
        let mut data = vec![N::zero(); alloc];
        for i in 0..dimension {
            let idx = i * dimension + i;
            data[idx] = N::one();
        }
        Self {
            data,
            dimension,
            is_identity: true,
        }
    }

    pub fn from_rows(rows: Vec<Vec<N>>) -> Self {
        let dimension = rows.len();
        assert!(
            rows.iter().all(|row| row.len() == dimension),
            "basis matrix must be square"
        );
        let capacity = strict_mul(dimension, dimension, "basis matrix allocation overflow");
        let mut data = Vec::with_capacity(capacity);
        for row in rows {
            data.extend(row);
        }
        Self {
            data,
            dimension,
            is_identity: false,
        }
    }

    pub fn from_row_slices(rows: &[&[N]]) -> Self {
        let dimension = rows.len();
        assert!(
            rows.iter().all(|row| row.len() == dimension),
            "basis matrix must be square"
        );
        let capacity = strict_mul(dimension, dimension, "basis matrix allocation overflow");
        let mut data = Vec::with_capacity(capacity);
        for row in rows {
            data.extend_from_slice(row);
        }
        Self {
            data,
            dimension,
            is_identity: false,
        }
    }

    pub fn from_flat(dimension: usize, data: Vec<N>) -> Self {
        let expected = strict_mul(dimension, dimension, "basis matrix allocation overflow");
        assert!(
            data.len() == expected,
            "basis matrix data length mismatch (expected {}, got {})",
            expected,
            data.len()
        );
        Self {
            data,
            dimension,
            is_identity: false,
        }
    }

    pub fn dim(&self) -> usize {
        self.dimension
    }

    pub fn row(&self, row: usize) -> &[N] {
        let start = row * self.dimension;
        let end = start + self.dimension;
        &self.data[start..end]
    }

    pub fn row_mut(&mut self, row: usize) -> &mut [N] {
        let start = row * self.dimension;
        let end = start + self.dimension;
        &mut self.data[start..end]
    }

    pub fn rows(&self) -> impl std::iter::ExactSizeIterator<Item = &[N]> {
        let dim = self.dimension.max(1);
        self.data.chunks_exact(dim).take(self.dimension)
    }

    pub fn rows_mut(&mut self) -> impl std::iter::ExactSizeIterator<Item = &mut [N]> {
        let dim = self.dimension.max(1);
        self.data.chunks_exact_mut(dim).take(self.dimension)
    }

    pub fn get(&self, row: usize, col: usize) -> &N {
        let idx = self.index(row, col);
        &self.data[idx]
    }

    pub fn set(&mut self, row: usize, col: usize, value: N) {
        let idx = self.index(row, col);
        self.data[idx] = value;
    }

    pub fn column(&self, col: usize) -> Vec<N> {
        (0..self.dimension)
            .map(|row| self.get(row, col).clone())
            .collect()
    }

    pub fn is_identity(&self) -> bool {
        self.is_identity
    }

    pub fn mark_non_identity(&mut self) {
        self.is_identity = false;
    }

    fn index(&self, row: usize, col: usize) -> usize {
        row * self.dimension + col
    }
}

#[inline(always)]
fn swap_rows_in_flat<N: Num>(data: &mut [N], width: usize, r1: usize, r2: usize) {
    debug_assert!(width > 0, "swap_rows_in_flat called with width=0");
    if r1 == r2 {
        return;
    }
    let start1 = r1 * width;
    let start2 = r2 * width;
    debug_assert!(start1 + width <= data.len(), "row 1 out of bounds");
    debug_assert!(start2 + width <= data.len(), "row 2 out of bounds");
    unsafe {
        let ptr = data.as_mut_ptr();
        std::ptr::swap_nonoverlapping(ptr.add(start1), ptr.add(start2), width);
    }
}

pub type MatrixRowIter<'a, N> = std::iter::Take<std::slice::ChunksExact<'a, N>>;
pub type MatrixRowIterMut<'a, N> = std::iter::Take<std::slice::ChunksExactMut<'a, N>>;

/// A dense row-major matrix.
///
/// This is the engine-agnostic storage primitive: it has no DD/LRS/LP annotations.
#[derive(Clone, Debug)]
pub struct Matrix<N: Num> {
    data: Vec<N>,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
pub struct MatrixBuilder<N: Num> {
    expected_rows: Option<usize>,
    col_count: usize,
    storage: Matrix<N>,
}

impl<N: Num> MatrixBuilder<N> {
    pub fn new(row_count: usize, col_count: usize) -> Self {
        let storage = Matrix::with_capacity(row_count, col_count);
        Self {
            expected_rows: Some(row_count),
            col_count,
            storage,
        }
    }

    pub fn with_columns(col_count: usize) -> Self {
        Self {
            expected_rows: None,
            col_count,
            storage: Matrix::new(0, col_count),
        }
    }

    #[inline(always)]
    pub fn row_count(&self) -> usize {
        self.storage.row_count()
    }

    #[inline(always)]
    pub fn col_count(&self) -> usize {
        self.col_count
    }

    #[inline(always)]
    pub fn storage(&self) -> &Matrix<N> {
        &self.storage
    }

    pub fn from_row_slices(rows: &[&[N]], cols: usize) -> Self {
        let storage = Matrix::from_row_slices(rows, cols);
        let mut builder = Self::with_columns(storage.col_count());
        builder.expected_rows = Some(storage.row_count());
        builder.storage = storage;
        builder
    }

    pub fn from_rows(rows: Vec<Vec<N>>) -> Self {
        let storage = Matrix::from_rows(rows);
        let mut builder = Self::with_columns(storage.col_count());
        builder.expected_rows = Some(storage.row_count());
        builder.storage = storage;
        builder
    }

    pub fn from_flat(rows: usize, cols: usize, data: Vec<N>) -> Self {
        let storage = Matrix::from_flat(rows, cols, data);
        let mut builder = Self::with_columns(storage.col_count());
        builder.expected_rows = Some(storage.row_count());
        builder.storage = storage;
        builder
    }

    pub fn with_storage(mut self, storage: Matrix<N>) -> Self {
        assert!(
            storage.col_count() == self.col_count,
            "storage column count mismatch"
        );
        if let Some(expected) = self.expected_rows {
            assert!(
                expected == storage.row_count(),
                "storage row count mismatch"
            );
        } else {
            self.expected_rows = Some(storage.row_count());
        }
        self.storage = storage;
        self
    }

    pub fn with_rows(self, rows: Vec<Vec<N>>) -> Self {
        if rows.is_empty() {
            assert!(
                self.col_count > 0,
                "cannot build empty matrix with zero columns"
            );
            let col_count = self.col_count;
            return self.with_storage(Matrix::new(0, col_count));
        }
        let storage = Matrix::from_rows(rows);
        self.with_storage(storage)
    }

    pub fn push_row<T: AsRef<[N]>>(mut self, row: T) -> Self {
        let row = row.as_ref();
        assert!(
            row.len() == self.col_count,
            "row length {} does not match column count {}",
            row.len(),
            self.col_count
        );
        self.storage.push(row);
        self
    }

    pub fn build(self) -> Matrix<N> {
        if let Some(expected) = self.expected_rows {
            assert!(
                expected == self.storage.row_count(),
                "expected {} rows but have {}",
                expected,
                self.storage.row_count()
            );
        }
        assert!(
            self.col_count == self.storage.col_count(),
            "column count mismatch"
        );
        self.storage
    }
}

#[derive(Clone, Debug)]
pub struct MatrixRank {
    pub rank: usize,
    pub row_basis: RowSet,
    pub col_basis: ColSet,
}

impl<N: Num> Matrix<N> {
    pub fn new(rows: usize, cols: usize) -> Self {
        if cols == 0 {
            return Self {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        let total_size = strict_mul(rows, cols, "matrix allocation overflow");
        Self {
            data: vec![N::zero(); total_size],
            rows,
            cols,
        }
    }

    pub fn with_capacity(row_capacity: usize, cols: usize) -> Self {
        if cols == 0 {
            debug_assert_eq!(
                row_capacity, 0,
                "cannot build a non-empty matrix with zero columns"
            );
            return Self {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        Self {
            data: Vec::with_capacity(strict_mul(row_capacity, cols, "matrix capacity overflow")),
            rows: 0,
            cols,
        }
    }

    pub fn from_row_slices(rows: &[&[N]], cols: usize) -> Self {
        if cols == 0 {
            assert!(
                rows.iter().all(|row| row.is_empty()),
                "zero-column matrix cannot have non-empty rows"
            );
            return Self {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        assert!(
            rows.iter().all(|row| row.len() == cols),
            "all rows must have {} columns",
            cols
        );
        let mut data =
            Vec::with_capacity(strict_mul(rows.len(), cols, "matrix allocation overflow"));
        for row in rows {
            data.extend_from_slice(row);
        }
        Self {
            data,
            rows: rows.len(),
            cols,
        }
    }

    pub fn from_rows(rows: Vec<Vec<N>>) -> Self {
        let cols = rows.first().map_or(0, |r| r.len());
        assert!(
            rows.iter().all(|row| row.len() == cols),
            "all rows must have consistent length"
        );
        if cols == 0 {
            return Self {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        let row_count = rows.len();
        let mut data =
            Vec::with_capacity(strict_mul(row_count, cols, "matrix allocation overflow"));
        for row in rows {
            data.extend(row);
        }
        Self {
            data,
            rows: row_count,
            cols,
        }
    }

    pub fn from_flat(rows: usize, cols: usize, data: Vec<N>) -> Self {
        if cols == 0 {
            assert!(data.is_empty(), "zero-column matrix cannot have data");
            return Self {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        let expected = strict_mul(rows, cols, "matrix allocation overflow");
        assert!(
            data.len() == expected,
            "data length {} does not match {}x{} = {}",
            data.len(),
            rows,
            cols,
            expected
        );
        Self { data, rows, cols }
    }

    pub fn coerce_as<M>(&self) -> Result<Matrix<M>, ConversionError>
    where
        M: Num + CoerceFrom<N>,
    {
        if self.cols == 0 {
            return Ok(Matrix::new(0, 0));
        }
        let mut data: Vec<M> = Vec::with_capacity(self.data.len());
        for v in &self.data {
            data.push(M::coerce_from(v)?);
        }
        Ok(Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    #[inline(always)]
    pub fn row_count(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    pub fn rows(&self) -> MatrixRowIter<'_, N> {
        self.iter()
    }

    #[inline(always)]
    pub fn rows_mut(&mut self) -> MatrixRowIterMut<'_, N> {
        self.iter_mut()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    #[inline(always)]
    pub fn col_count(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    pub fn row(&self, index: usize) -> Option<&[N]> {
        if self.cols == 0 || index >= self.rows {
            return None;
        }
        let start = index * self.cols;
        let end = start + self.cols;
        Some(&self.data[start..end])
    }

    #[inline(always)]
    /// Returns the `index`th row without bounds checks.
    ///
    /// # Safety
    ///
    /// `index` must be strictly less than `self.row_count()`.
    pub unsafe fn row_unchecked(&self, index: usize) -> &[N] {
        debug_assert!(index < self.rows, "row index out of bounds");
        let start = index * self.cols;
        let end = start + self.cols;
        unsafe { self.data.get_unchecked(start..end) }
    }

    #[inline(always)]
    pub fn row_mut(&mut self, index: usize) -> Option<&mut [N]> {
        if self.cols == 0 || index >= self.rows {
            return None;
        }
        let start = index * self.cols;
        let end = start + self.cols;
        Some(&mut self.data[start..end])
    }

    #[inline(always)]
    pub fn push(&mut self, row: &[N]) {
        debug_assert_eq!(row.len(), self.cols, "row length mismatch");
        debug_assert!(self.cols > 0, "cannot push into a matrix with zero columns");
        self.data.extend_from_slice(row);
        self.rows += 1;
    }

    pub fn extend_from_matrix(&mut self, other: &Matrix<N>) {
        assert_eq!(self.cols, other.cols, "column count mismatch");
        if self.cols == 0 {
            return;
        }
        self.data.extend_from_slice(&other.data);
        self.rows += other.rows;
    }

    pub fn into_data(self) -> Vec<N> {
        self.data
    }

    #[inline(always)]
    pub fn iter(&self) -> MatrixRowIter<'_, N> {
        let chunk = self.cols.max(1);
        debug_assert_eq!(self.data.len(), self.rows * chunk);
        self.data.chunks_exact(chunk).take(self.rows)
    }

    #[inline(always)]
    pub fn iter_mut(&mut self) -> MatrixRowIterMut<'_, N> {
        let chunk = self.cols.max(1);
        debug_assert_eq!(self.data.len(), self.rows * chunk);
        self.data.chunks_exact_mut(chunk).take(self.rows)
    }

    pub fn reorder_rows_by_order(&mut self, order: &[usize]) {
        debug_assert_eq!(order.len(), self.rows);
        if self.rows <= 1 || self.cols == 0 {
            return;
        }
        if order.iter().enumerate().all(|(i, &old)| i == old) {
            return;
        }

        let cols = self.cols;
        let total = strict_mul(self.rows, cols, "reorder total overflow");
        debug_assert_eq!(self.data.len(), total);

        use std::mem::{ManuallyDrop, MaybeUninit};
        use std::ptr;

        let mut old = ManuallyDrop::new(std::mem::take(&mut self.data));
        let src = old.as_mut_ptr();
        let cap = old.capacity();

        let mut new: Vec<MaybeUninit<N>> = Vec::with_capacity(total);
        unsafe { new.set_len(total) };
        let dst = new.as_mut_ptr() as *mut N;

        for (new_row, &old_row) in order.iter().enumerate() {
            debug_assert!(old_row < self.rows, "row order out of bounds");
            let src_row = unsafe { src.add(old_row * cols) };
            let dst_row = unsafe { dst.add(new_row * cols) };
            unsafe { ptr::copy_nonoverlapping(src_row, dst_row, cols) };
        }

        unsafe {
            let _ = Vec::from_raw_parts(src, 0, cap);
        }
        self.data = unsafe { std::mem::transmute::<Vec<MaybeUninit<N>>, Vec<N>>(new) };
    }

    pub fn rank(
        &self,
        ignored_rows: &RowSet,
        ignored_cols: &ColSet,
        eps: &impl Epsilon<N>,
    ) -> MatrixRank {
        debug_assert_eq!(
            ignored_rows.len(),
            self.rows,
            "ignored row mask dimension mismatch"
        );
        debug_assert_eq!(
            ignored_cols.len(),
            self.cols,
            "ignored col mask dimension mismatch"
        );

        let row_map: Vec<usize> = (0..self.rows)
            .filter(|r| !ignored_rows.contains(*r))
            .collect();
        let col_map: Vec<usize> = (0..self.cols)
            .filter(|c| !ignored_cols.contains(*c))
            .collect();

        let m = row_map.len();
        let n = col_map.len();
        let mut a = vec![N::zero(); strict_mul(m, n, "rank matrix allocation overflow")];
        let width = n;
        for (i_idx, &r) in row_map.iter().enumerate() {
            let row_start = i_idx * width;
            let src = self.row(r).expect("row index in range");
            for (j_idx, &c) in col_map.iter().enumerate() {
                a[row_start + j_idx] = src[c].clone();
            }
        }

        let mut rank = 0usize;
        let mut row_basis = RowSet::new(self.rows);
        let mut col_basis = ColSet::new(self.cols);
        let mut row = 0usize;
        for col in 0..n {
            let mut pivot_row = None;
            let mut best_abs = None;
            for r in row..m {
                let val = a[r * width + col].abs();
                if eps.is_zero(&val) {
                    continue;
                }
                let better = best_abs
                    .as_ref()
                    .is_none_or(|b| val.partial_cmp(b).is_some_and(|o| o.is_gt()));
                if better {
                    pivot_row = Some(r);
                    best_abs = Some(val);
                }
            }

            if let Some(piv) = pivot_row {
                if piv != row {
                    swap_rows_in_flat(&mut a, width, row, piv);
                }
                let pivot_val = a[row * width + col].clone();
                let inv_pivot = N::one().ref_div(&pivot_val);
                for r in (row + 1)..m {
                    let rstart = r * width;
                    if eps.is_zero(&a[rstart + col]) {
                        continue;
                    }
                    let factor = a[rstart + col].ref_mul(&inv_pivot);
                    for c in col..n {
                        let tmp = factor.ref_mul(&a[row * width + c]);
                        let idx = rstart + c;
                        a[idx] = a[idx].ref_sub(&tmp);
                    }
                }
                rank += 1;
                row_basis.insert(row_map[row]);
                col_basis.insert(col_map[col]);
                row += 1;
                if row == m {
                    break;
                }
            }
        }

        MatrixRank {
            rank,
            row_basis,
            col_basis,
        }
    }

    pub fn select_row_basis_rows(
        &self,
        candidates: &[usize],
        target_rank: usize,
        ignored_cols: &[bool],
        eps: &impl Epsilon<N>,
    ) -> Option<Vec<usize>> {
        let cols = self.cols;
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

        let mut basis: Vec<Vec<N>> = Vec::with_capacity(target_rank);
        let mut pivots: Vec<usize> = Vec::with_capacity(target_rank);
        let mut chosen: Vec<usize> = Vec::with_capacity(target_rank);

        let mut work: Vec<N> = vec![N::zero(); cols];

        for &row in candidates {
            if row >= self.rows {
                return None;
            }

            work.clone_from_slice(&self[row]);
            for (col, ignored) in ignored_cols.iter().enumerate() {
                if *ignored {
                    work[col] = N::zero();
                }
            }

            for (basis_row, &pivot_col) in basis.iter().zip(pivots.iter()) {
                let pivot = &work[pivot_col];
                if eps.is_zero(pivot) {
                    continue;
                }
                let factor = pivot.clone();
                for c in pivot_col..cols {
                    let tmp = factor.ref_mul(&basis_row[c]);
                    work[c] = work[c].ref_sub(&tmp);
                }
            }

            let pivot_col = (0..cols).find(|&c| !ignored_cols[c] && !eps.is_zero(&work[c]));
            let Some(pivot_col) = pivot_col else {
                continue;
            };

            let pivot_val = work[pivot_col].clone();
            for x in &mut work[pivot_col..cols] {
                *x = x.ref_div(&pivot_val);
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

    pub fn solve_nullspace_1d(&self, rows: &RowSet, eps: &impl Epsilon<N>) -> Option<Vec<N>> {
        let n = self.cols;
        if n == 0 {
            return None;
        }

        let m = rows.cardinality();
        let mut a = vec![N::zero(); strict_mul(m, n, "nullspace matrix allocation overflow")];
        let width = n;
        for (i_idx, row_id) in rows.iter().enumerate() {
            let r = row_id.as_index();
            let row_start = i_idx * width;
            let src = &self[r];
            a[row_start..row_start + n].clone_from_slice(&src[..n]);
        }

        Self::solve_nullspace_1d_dense(&mut a, m, n, eps)
    }

    pub fn solve_nullspace_1d_with_unit_cols(
        &self,
        rows: &RowSet,
        unit_cols: &[usize],
        eps: &impl Epsilon<N>,
    ) -> Option<Vec<N>> {
        let n = self.cols;
        if n == 0 {
            return None;
        }

        for &col in unit_cols {
            if col >= n {
                return None;
            }
        }

        let selected = rows.cardinality();
        let m = selected.checked_add(unit_cols.len())?;
        let mut a = vec![N::zero(); strict_mul(m, n, "nullspace matrix allocation overflow")];
        let width = n;
        for (i_idx, row_id) in rows.iter().enumerate() {
            let r = row_id.as_index();
            let row_start = i_idx * width;
            let src = &self[r];
            a[row_start..row_start + n].clone_from_slice(&src[..n]);
        }
        for (unit_idx, &col) in unit_cols.iter().enumerate() {
            let row_start = (selected + unit_idx) * width;
            a[row_start + col] = N::one();
        }

        Self::solve_nullspace_1d_dense(&mut a, m, n, eps)
    }

    pub fn solve_nullspace_1d_rows_with_unit_cols(
        &self,
        rows: &[usize],
        unit_cols: &[usize],
        eps: &impl Epsilon<N>,
    ) -> Option<Vec<N>> {
        let n = self.cols;
        if n == 0 {
            return None;
        }

        for &row in rows {
            if row >= self.rows {
                return None;
            }
        }
        for &col in unit_cols {
            if col >= n {
                return None;
            }
        }

        let m = rows.len().checked_add(unit_cols.len())?;
        let mut a = vec![N::zero(); strict_mul(m, n, "nullspace matrix allocation overflow")];
        let width = n;
        for (i_idx, &r) in rows.iter().enumerate() {
            let row_start = i_idx * width;
            let src = &self[r];
            a[row_start..row_start + n].clone_from_slice(&src[..n]);
        }
        for (unit_idx, &col) in unit_cols.iter().enumerate() {
            let row_start = (rows.len() + unit_idx) * width;
            a[row_start + col] = N::one();
        }

        Self::solve_nullspace_1d_dense(&mut a, m, n, eps)
    }

    fn solve_nullspace_1d_dense(
        a: &mut [N],
        m: usize,
        n: usize,
        eps: &impl Epsilon<N>,
    ) -> Option<Vec<N>> {
        #[inline(always)]
        fn sub_mul_slice_assign<N: Num>(dst: &mut [N], factor: &N, rhs: &[N]) {
            debug_assert_eq!(dst.len(), rhs.len(), "axpy dimension mismatch");
            const SIMD_MIN_LEN: usize = 64;
            if dst.len() >= SIMD_MIN_LEN {
                linalg::axpy_sub(dst, factor, rhs);
                return;
            }
            for (dst_i, rhs_i) in dst.iter_mut().zip(rhs.iter()) {
                linalg::sub_mul_assign(dst_i, factor, rhs_i);
            }
        }

        #[inline(always)]
        fn dot<N: Num>(lhs: &[N], rhs: &[N]) -> N {
            debug_assert_eq!(lhs.len(), rhs.len(), "dot product dimension mismatch");
            const SIMD_MIN_LEN: usize = 64;
            if lhs.len() >= SIMD_MIN_LEN {
                return linalg::dot(lhs, rhs);
            }
            let mut acc = N::zero();
            for (a, b) in lhs.iter().zip(rhs.iter()) {
                linalg::add_mul_assign(&mut acc, a, b);
            }
            acc
        }

        #[inline(always)]
        fn div_slice_assign<N: Num>(dst: &mut [N], divisor: &N) {
            const SIMD_MIN_LEN: usize = 64;
            if dst.len() >= SIMD_MIN_LEN {
                linalg::div_assign(dst, divisor);
                return;
            }
            for dst_i in dst {
                *dst_i = dst_i.ref_div(divisor);
            }
        }

        debug_assert_eq!(
            a.len(),
            strict_mul(m, n, "nullspace matrix dimension mismatch"),
            "dense system shape mismatch"
        );

        let width = n;
        let mut pivot_cols = Vec::new();
        let mut row = 0usize;
        for col in 0..n {
            let mut pivot_row = None;
            let mut best_abs = None;
            for r in row..m {
                let val = a[r * width + col].abs();
                if eps.is_zero(&val) {
                    continue;
                }
                let better = best_abs
                    .as_ref()
                    .is_none_or(|b| val.partial_cmp(b).is_some_and(|o| o.is_gt()));
                if better {
                    pivot_row = Some(r);
                    best_abs = Some(val);
                }
            }
            if let Some(piv) = pivot_row {
                if piv != row {
                    swap_rows_in_flat(a, width, row, piv);
                }
                let pivot_val = a[row * width + col].clone();
                let pivot_row_start = row * width;
                let pivot_row_end = pivot_row_start + width;
                div_slice_assign(&mut a[pivot_row_start + col..pivot_row_end], &pivot_val);

                let (above_pivot, below) = a.split_at_mut(pivot_row_end);
                let pivot_row: &[N] = &above_pivot[pivot_row_start..pivot_row_end];

                let remaining_rows = m.saturating_sub(row + 1);
                for r in 0..remaining_rows {
                    let rstart = r * width;
                    if eps.is_zero(&below[rstart + col]) {
                        continue;
                    }
                    let factor = below[rstart + col].clone();
                    let row_slice = &mut below[rstart..rstart + width];
                    sub_mul_slice_assign(&mut row_slice[col..], &factor, &pivot_row[col..]);
                }

                pivot_cols.push(col);
                row += 1;
                if row == m {
                    break;
                }
            }
        }

        let rank = pivot_cols.len();
        if rank + 1 != n {
            return None;
        }

        let mut is_pivot = vec![false; n];
        for &col in &pivot_cols {
            is_pivot[col] = true;
        }
        let free_col = (0..n).find(|c| !is_pivot[*c])?;

        let mut v = vec![N::zero(); n];
        v[free_col] = N::one();
        for (row_idx, &pivot_col) in pivot_cols.iter().enumerate().rev() {
            let row_start = row_idx * width;
            let col_start = pivot_col + 1;
            let acc = dot(
                &a[row_start + col_start..row_start + width],
                &v[col_start..],
            );
            v[pivot_col] = -acc;
        }
        Some(v)
    }
}

#[inline(always)]
pub fn lex_cmp<N: Num>(a: &[N], b: &[N], eps: &impl Epsilon<N>) -> std::cmp::Ordering {
    let n = a.len().min(b.len());
    for i in 0..n {
        let ord = eps.cmp(&a[i], &b[i]);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

impl<'a, N: Num> IntoIterator for &'a Matrix<N> {
    type Item = &'a [N];
    type IntoIter = MatrixRowIter<'a, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, N: Num> IntoIterator for &'a mut Matrix<N> {
    type Item = &'a mut [N];
    type IntoIter = MatrixRowIterMut<'a, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<N: Num> std::ops::Index<usize> for Matrix<N> {
    type Output = [N];

    fn index(&self, index: usize) -> &Self::Output {
        self.row(index).expect("row index out of bounds")
    }
}

impl<N: Num> std::ops::IndexMut<usize> for Matrix<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.row_mut(index).expect("row index out of bounds")
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn matrix_iter_mut_is_rectangular() {
        let mut m = Matrix::<f64>::new(2, 3);
        for row in m.iter_mut() {
            assert_eq!(row.len(), 3);
        }
    }
}
