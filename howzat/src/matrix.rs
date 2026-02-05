use std::fmt::Write;
use std::marker::PhantomData;

use crate::HowzatError as Error;
use crate::dd::DefaultNormalizer;
use crate::lp::{LpObjective, LpSolver};
use calculo::linalg;
use calculo::num::{CoerceFrom, Epsilon, Normalizer, Num};
use calculo::util::{strict_mul, strict_sub};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{
    Col, ColSet, Inequality, Representation, RepresentationKind, Row, RowId, RowIndex, RowSet,
};

mod row_permutation {
    #[derive(Clone, Debug)]
    pub(super) struct RowPermutation {
        order: Vec<usize>,
    }

    impl RowPermutation {
        pub(super) fn sorted_by<F>(size: usize, mut compare: F) -> Self
        where
            F: FnMut(usize, usize) -> std::cmp::Ordering,
        {
            let mut order: Vec<usize> = (0..size).collect();
            order.sort_unstable_by(|&a, &b| compare(a, b));
            Self { order }
        }

        pub(super) fn partition_by_flags(flags: &[bool]) -> Self {
            let size = flags.len();
            let mut order = Vec::with_capacity(size);
            for (idx, &flag) in flags.iter().enumerate() {
                if flag {
                    order.push(idx);
                }
            }
            for (idx, &flag) in flags.iter().enumerate() {
                if !flag {
                    order.push(idx);
                }
            }
            debug_assert_eq!(order.len(), size);
            Self { order }
        }

        pub(super) fn as_slice(&self) -> &[usize] {
            &self.order
        }
    }
}

use row_permutation::RowPermutation;

use hullabaloo::matrix::MatrixBuilder as DenseMatrixBuilder;
pub use hullabaloo::matrix::{Matrix, MatrixRank, MatrixRowIter, MatrixRowIterMut, lex_cmp};

#[derive(Clone, Copy, Debug)]
pub struct IncidenceConfig<'a> {
    pub column_mapping: Option<&'a [Option<usize>]>,
    pub include_slack: bool,
}

impl<'a> IncidenceConfig<'a> {
    pub fn with_column_mapping(column_mapping: &'a [Option<usize>]) -> Self {
        Self {
            column_mapping: Some(column_mapping),
            ..Self::default()
        }
    }

    pub fn no_slack(mut self) -> Self {
        self.include_slack = false;
        self
    }
}

impl<'a> Default for IncidenceConfig<'a> {
    fn default() -> Self {
        Self {
            column_mapping: None,
            include_slack: true,
        }
    }
}

#[derive(Debug)]
pub struct LpMatrix<N: Num, R: Representation> {
    storage: Matrix<N>,
    linearity: RowSet,
    row_vec: Vec<N>,
    objective: LpObjective,
    _repr: PhantomData<R>,
}

impl<N: Num, R: Representation> Clone for LpMatrix<N, R> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            linearity: self.linearity.clone(),
            row_vec: self.row_vec.clone(),
            objective: self.objective,
            _repr: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LpMatrixBuilder<N: Num, R: Representation = hullabaloo::types::Inequality> {
    storage: DenseMatrixBuilder<N>,
    linearity: RowSet,
    row_vec: Vec<N>,
    objective: LpObjective,
    _repr: PhantomData<R>,
}

#[derive(Clone, Debug)]
pub struct LinearityCanonicalization<N: Num, R: Representation = hullabaloo::types::Inequality> {
    matrix: LpMatrix<N, R>,
    implicit_linearity: RowSet,
    positions: RowIndex,
}

impl<N: Num, R: Representation> LinearityCanonicalization<N, R> {
    pub fn new(matrix: LpMatrix<N, R>, implicit_linearity: RowSet, positions: RowIndex) -> Self {
        Self {
            matrix,
            implicit_linearity,
            positions,
        }
    }

    pub fn matrix(&self) -> &LpMatrix<N, R> {
        &self.matrix
    }

    pub fn implicit_linearity(&self) -> &RowSet {
        &self.implicit_linearity
    }

    pub fn positions(&self) -> &RowIndex {
        &self.positions
    }

    pub fn into_parts(self) -> (LpMatrix<N, R>, RowSet, RowIndex) {
        (self.matrix, self.implicit_linearity, self.positions)
    }

    pub fn into_matrix(self) -> LpMatrix<N, R> {
        self.matrix
    }
}

#[derive(Clone, Debug)]
pub struct CanonicalizationResult<N: Num, R: Representation = hullabaloo::types::Inequality> {
    matrix: LpMatrix<N, R>,
    implicit_linearity: RowSet,
    redundant_rows: RowSet,
    positions: RowIndex,
}

impl<N: Num, R: Representation> CanonicalizationResult<N, R> {
    pub fn new(
        matrix: LpMatrix<N, R>,
        implicit_linearity: RowSet,
        redundant_rows: RowSet,
        positions: RowIndex,
    ) -> Self {
        Self {
            matrix,
            implicit_linearity,
            redundant_rows,
            positions,
        }
    }

    pub fn matrix(&self) -> &LpMatrix<N, R> {
        &self.matrix
    }

    pub fn implicit_linearity(&self) -> &RowSet {
        &self.implicit_linearity
    }

    pub fn redundant_rows(&self) -> &RowSet {
        &self.redundant_rows
    }

    pub fn positions(&self) -> &RowIndex {
        &self.positions
    }

    pub fn into_parts(self) -> (LpMatrix<N, R>, RowSet, RowSet, RowIndex) {
        (
            self.matrix,
            self.implicit_linearity,
            self.redundant_rows,
            self.positions,
        )
    }

    pub fn into_matrix(self) -> LpMatrix<N, R> {
        self.matrix
    }
}

impl<N: Num, R: Representation> LpMatrixBuilder<N, R> {
    pub fn new(row_count: usize, col_count: usize) -> Self {
        let storage = DenseMatrixBuilder::new(row_count, col_count);
        Self {
            storage,
            linearity: RowSet::new(row_count),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn with_columns(col_count: usize) -> Self {
        Self {
            storage: DenseMatrixBuilder::with_columns(col_count),
            linearity: RowSet::new(0),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn from_row_slices(rows: &[&[N]], cols: usize) -> Self {
        let storage = DenseMatrixBuilder::from_row_slices(rows, cols);
        let col_count = storage.col_count();
        let row_count = storage.row_count();
        Self {
            storage,
            linearity: RowSet::new(row_count),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn from_rows(rows: Vec<Vec<N>>) -> Self {
        let storage = DenseMatrixBuilder::from_rows(rows);
        let col_count = storage.col_count();
        let row_count = storage.row_count();
        Self {
            storage,
            linearity: RowSet::new(row_count),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn from_flat(rows: usize, cols: usize, data: Vec<N>) -> Self {
        let storage = DenseMatrixBuilder::from_flat(rows, cols, data);
        let col_count = storage.col_count();
        let row_count = storage.row_count();
        Self {
            storage,
            linearity: RowSet::new(row_count),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn from_matrix(template: &LpMatrix<N, R>) -> Self {
        let mut builder = Self::with_columns(template.col_count_internal());
        builder.row_vec = template.row_vec.clone();
        builder.objective = template.objective;
        builder
    }

    pub fn with_storage(mut self, storage: Matrix<N>) -> Self {
        let row_count = storage.row_count();
        let col_count = storage.col_count();
        self.storage = self.storage.with_storage(storage);
        self.linearity = RowSet::new(row_count);
        if self.row_vec.len() != col_count {
            self.row_vec = vec![N::zero(); col_count];
        }
        self
    }

    pub fn with_rows(mut self, rows: Vec<Vec<N>>) -> Self {
        self.storage = self.storage.with_rows(rows);
        let row_count = self.storage.row_count();
        let col_count = self.storage.col_count();
        self.linearity = RowSet::new(row_count);
        if self.row_vec.len() != col_count {
            self.row_vec = vec![N::zero(); col_count];
        }
        self
    }

    pub fn push_row<T: AsRef<[N]>>(mut self, row: T, is_linearity: bool) -> Self {
        let row = row.as_ref();
        if self.storage.col_count() == 0 && self.row_vec.is_empty() {
            self.row_vec.resize(row.len(), N::zero());
            self.storage = DenseMatrixBuilder::with_columns(row.len());
        }
        self.storage = self.storage.push_row(row);
        let rows_len = self.storage.row_count();
        if self.linearity.len() < rows_len {
            self.linearity.resize(rows_len);
        }
        if is_linearity {
            debug_assert!(rows_len > 0, "matrix must contain the inserted row");
            self.linearity.insert(RowId::new(rows_len - 1));
        }
        self
    }

    pub fn with_linearity(mut self, linearity: RowSet) -> Self {
        assert_eq!(
            linearity.len(),
            self.storage.row_count(),
            "linearity set size must match row count"
        );
        self.linearity = linearity;
        self
    }

    pub fn with_linearity_rows(mut self, rows: &RowSet) -> Self {
        assert_eq!(
            rows.len(),
            self.storage.row_count(),
            "linearity set size must match row count"
        );
        self.linearity = rows.clone();
        self
    }

    pub fn with_row_vec(mut self, row_vec: Vec<N>) -> Self {
        assert_eq!(
            row_vec.len(),
            self.storage.col_count(),
            "row_vec length must match column count"
        );
        self.row_vec = row_vec;
        self
    }

    pub fn with_objective(mut self, objective: LpObjective) -> Self {
        self.objective = objective;
        self
    }

    pub fn build(self) -> LpMatrix<N, R> {
        assert_eq!(
            self.linearity.len(),
            self.storage.row_count(),
            "linearity set size must match row count"
        );
        assert_eq!(
            self.row_vec.len(),
            self.storage.col_count(),
            "row_vec length must match column count"
        );
        let storage = self.storage.build();
        LpMatrix {
            storage,
            linearity: self.linearity,
            row_vec: self.row_vec,
            objective: self.objective,
            _repr: PhantomData,
        }
    }
}

impl<N: Num, R: Representation> LpMatrix<N, R> {
    #[inline(always)]
    pub fn storage(&self) -> &Matrix<N> {
        &self.storage
    }

    #[inline(always)]
    pub fn storage_mut(&mut self) -> &mut Matrix<N> {
        &mut self.storage
    }

    #[inline(always)]
    pub fn rows(&self) -> MatrixRowIter<'_, N> {
        self.storage.iter()
    }

    #[inline(always)]
    pub fn rows_mut(&mut self) -> MatrixRowIterMut<'_, N> {
        self.storage.iter_mut()
    }

    pub fn into_matrix_and_linearity(self) -> (Matrix<N>, RowSet) {
        (self.storage, self.linearity)
    }

    pub fn new(row_count: usize, col_count: usize) -> Self {
        let storage = Matrix::new(row_count, col_count);
        let linearity = RowSet::new(row_count);
        Self {
            storage,
            linearity,
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn from_row_slices(rows: &[&[N]], cols: usize) -> Self {
        let storage = Matrix::from_row_slices(rows, cols);
        let row_count = storage.row_count();
        let col_count = storage.col_count();
        Self {
            storage,
            linearity: RowSet::new(row_count),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn from_rows(rows: Vec<Vec<N>>) -> Self {
        let storage = Matrix::from_rows(rows);
        let row_count = storage.row_count();
        let col_count = storage.col_count();
        Self {
            storage,
            linearity: RowSet::new(row_count),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn from_flat(rows: usize, cols: usize, data: Vec<N>) -> Self {
        let storage = Matrix::from_flat(rows, cols, data);
        let row_count = storage.row_count();
        let col_count = storage.col_count();
        Self {
            storage,
            linearity: RowSet::new(row_count),
            row_vec: vec![N::zero(); col_count],
            objective: LpObjective::None,
            _repr: PhantomData,
        }
    }

    pub fn row(&self, row: Row) -> Option<&[N]> {
        self.storage.row(row)
    }

    #[inline(always)]
    pub fn row_mut(&mut self, row: Row) -> Option<&mut [N]> {
        self.storage.row_mut(row)
    }

    pub fn row_count(&self) -> usize {
        self.storage.row_count()
    }

    pub fn col_count(&self) -> usize {
        self.col_count_internal()
    }

    fn col_count_internal(&self) -> usize {
        self.storage.col_count()
    }

    pub fn linearity(&self) -> &RowSet {
        &self.linearity
    }

    pub fn representation(&self) -> RepresentationKind {
        R::KIND
    }

    pub fn row_vec(&self) -> &[N] {
        &self.row_vec
    }

    pub fn objective(&self) -> LpObjective {
        self.objective
    }

    pub fn coerce_as<M>(&self) -> Result<LpMatrix<M, R>, Error>
    where
        M: Num + CoerceFrom<N>,
    {
        let storage = self
            .storage
            .coerce_as::<M>()
            .map_err(|_| Error::ConversionFailure)?;
        let mut row_vec: Vec<M> = Vec::with_capacity(self.row_vec.len());
        for v in &self.row_vec {
            row_vec.push(M::coerce_from(v).map_err(|_| Error::ConversionFailure)?);
        }
        Ok(LpMatrix {
            storage,
            linearity: self.linearity.clone(),
            row_vec,
            objective: self.objective,
            _repr: PhantomData,
        })
    }

    pub fn with_row_vec(self, row_vec: Vec<N>) -> Self {
        self.builder().with_row_vec(row_vec).build()
    }

    pub fn with_linearity(self, linearity: RowSet) -> Self {
        self.builder().with_linearity(linearity).build()
    }

    pub fn with_objective(mut self, objective: LpObjective) -> Self {
        self.objective = objective;
        self
    }

    pub fn with_linearity_rows(&self, rows: &RowSet) -> Self {
        assert_eq!(
            rows.len(),
            self.row_count(),
            "linearity set size must match row count"
        );
        let mut builder = LpMatrixBuilder::from_matrix(self);
        builder = builder.with_storage(self.storage.clone());
        builder = builder.with_linearity_rows(rows);
        builder.build()
    }

    fn clone_with_storage(&self, storage: Matrix<N>, linearity: RowSet) -> Self {
        let mut builder = LpMatrixBuilder::from_matrix(self);
        builder = builder.with_storage(storage);
        builder = builder.with_linearity(linearity);
        builder.build()
    }

    fn builder(&self) -> LpMatrixBuilder<N, R> {
        let mut builder = LpMatrixBuilder::from_matrix(self);
        builder = builder.with_storage(self.storage.clone());
        builder = builder.with_linearity(self.linearity.clone());
        builder = builder.with_row_vec(self.row_vec.clone());
        builder.with_objective(self.objective)
    }

    fn rebuild_with_row_vec(&self, storage: Matrix<N>, linearity: RowSet, row_vec: Vec<N>) -> Self {
        let mut builder = LpMatrixBuilder::<N, R>::with_columns(row_vec.len());
        builder = builder.with_storage(storage);
        builder = builder.with_linearity(linearity);
        builder = builder.with_row_vec(row_vec);
        builder = builder.with_objective(self.objective);
        builder.build()
    }

    pub fn normalized(&self, eps: &impl Epsilon<N>) -> Self {
        let m = self.storage.row_count();
        let cols = self.col_count_internal();
        if m == 0 || cols == 0 {
            return self.clone();
        }
        let mut data = Vec::with_capacity(strict_mul(m, cols, "normalized allocation overflow"));
        for row in self.storage.iter() {
            let start = data.len();
            data.extend_from_slice(row);
            canonicalize_row(&mut data[start..start + cols], eps);
        }
        let storage = Matrix::from_flat(m, cols, data);
        self.clone_with_storage(storage, self.linearity.clone())
    }

    pub fn normalized_sorted(&self, eps: &impl Epsilon<N>) -> (Self, RowIndex) {
        let m = self.storage.row_count();
        let cols = self.col_count_internal();
        if m == 0 || cols == 0 {
            return (self.clone(), Vec::new());
        }
        let mut data = Vec::with_capacity(strict_mul(m, cols, "normalized_sorted overflow"));
        for row in self.storage.iter() {
            let start = data.len();
            data.extend_from_slice(row);
            canonicalize_row(&mut data[start..start + cols], eps);
        }

        let order = RowPermutation::sorted_by(m, |a, b| {
            let a0 = a * cols;
            let b0 = b * cols;
            lex_cmp(&data[a0..a0 + cols], &data[b0..b0 + cols], eps)
        });

        let mut newpos: RowIndex = vec![0; m];
        for (new_idx, &old_idx) in order.as_slice().iter().enumerate() {
            newpos[old_idx] = new_idx as isize;
        }

        let mut storage = Matrix::from_flat(m, cols, data);
        storage.reorder_rows_by_order(order.as_slice());

        let mut linearity = RowSet::new(m);
        for idx in self.linearity.iter() {
            let old = idx.as_index();
            linearity.insert(newpos[old] as usize);
        }

        (self.clone_with_storage(storage, linearity), newpos)
    }

    pub fn unique(&self, eps: &impl Epsilon<N>) -> (Self, RowIndex) {
        if self.storage.is_empty() {
            return (self.clone(), Vec::new());
        }
        let mut newpos: RowIndex = vec![0; self.storage.row_count()];
        let mut reps: Vec<usize> = Vec::new();
        let mut current_rep = 0usize;
        let mut current_pos: isize = 0;
        let mut rep_is_lin = self.linearity.contains(RowId::new(0));
        newpos[0] = current_pos;
        reps.push(current_rep);

        for i in 1..self.storage.row_count() {
            let is_lin = self.linearity.contains(RowId::new(i));
            if rows_equal(&self.storage[i], &self.storage[current_rep], eps) {
                if is_lin && !rep_is_lin {
                    newpos[current_rep] = -((i + 1) as isize);
                    current_rep = i;
                    rep_is_lin = true;
                    debug_assert!(
                        current_pos >= 0,
                        "representative position must be nonnegative"
                    );
                    let pos = current_pos as usize;
                    debug_assert!(pos < reps.len(), "representative position out of range");
                    reps[pos] = i;
                    newpos[i] = current_pos;
                } else {
                    newpos[i] = -((current_rep + 1) as isize);
                }
            } else {
                current_pos += 1;
                current_rep = i;
                rep_is_lin = is_lin;
                reps.push(i);
                newpos[i] = current_pos;
            }
        }

        let cols = self.col_count_internal();
        let mut linearity = RowSet::new(reps.len());
        let mut data = Vec::with_capacity(strict_mul(
            reps.len(),
            cols,
            "unique output allocation overflow",
        ));
        for (pos_idx, &orig_idx) in reps.iter().enumerate() {
            data.extend_from_slice(&self.storage[orig_idx]);
            if self.linearity.contains(orig_idx) {
                linearity.insert(pos_idx);
            }
        }
        let storage = Matrix::from_flat(reps.len(), cols, data);
        (self.clone_with_storage(storage, linearity), newpos)
    }

    pub fn normalized_sorted_unique(&self, eps: &impl Epsilon<N>) -> (Self, RowIndex) {
        let (matrix, positions, _dups) = self.canonical_row_order(eps);
        (matrix, positions)
    }

    pub fn append(self, other: &LpMatrix<N, R>) -> LpMatrix<N, R> {
        assert_eq!(self.representation(), other.representation());
        let mut storage = self.storage;
        let objective = self.objective;
        let row_vec = self.row_vec;
        storage.extend_from_matrix(&other.storage);
        let mut linearity = RowSet::new(storage.row_count());
        for idx in self.linearity.iter() {
            linearity.insert(idx);
        }
        let offset = strict_sub(
            storage.row_count(),
            other.storage.row_count(),
            "append row offset underflow",
        );
        for idx in other.linearity.iter() {
            linearity.insert(idx + offset);
        }
        LpMatrix {
            storage,
            linearity,
            row_vec,
            objective,
            _repr: PhantomData,
        }
    }

    pub fn without_row(&self, row: Row) -> LpMatrix<N, R> {
        assert!(
            row < self.storage.row_count(),
            "without_row: index {} out of bounds (len {})",
            row,
            self.storage.row_count()
        );
        let mut delete = RowSet::new(self.storage.row_count());
        delete.insert(row);
        self.submatrix(&delete)
    }

    pub fn without_rows(&self, rows: &RowSet) -> (LpMatrix<N, R>, RowIndex) {
        self.submatrix_with_positions(rows)
    }

    pub fn submatrix(&self, delete: &RowSet) -> LpMatrix<N, R> {
        self.submatrix_with_positions(delete).0
    }

    pub fn submatrix_with_positions(&self, delete: &RowSet) -> (LpMatrix<N, R>, RowIndex) {
        let remaining = strict_sub(
            self.storage.row_count(),
            delete.cardinality(),
            "submatrix delete count exceeds row count",
        );
        let cols = self.col_count_internal();
        let mut flat: Vec<N> =
            Vec::with_capacity(strict_mul(remaining, cols, "submatrix allocation overflow"));
        let mut newpos = vec![-1; self.storage.row_count()];
        let mut new_idx = 0;
        for (i, row) in self.storage.iter().enumerate() {
            if delete.contains(i) {
                newpos[i] = -1;
            } else {
                flat.extend_from_slice(row);
                newpos[i] = new_idx as isize;
                new_idx += 1;
            }
        }
        let mut linearity = RowSet::new(remaining);
        for idx in self.linearity.iter() {
            let mapped = newpos[idx.as_index()];
            if mapped >= 0 {
                linearity.insert(mapped as usize);
            }
        }
        let storage = Matrix::from_flat(remaining, cols, flat);
        (self.clone_with_storage(storage, linearity), newpos)
    }

    pub fn submatrix_with_linearity_shift(&self, delete: &RowSet) -> (LpMatrix<N, R>, RowIndex) {
        let (m, newpos_submatrix) = self.submatrix_with_positions(delete);
        let (m, shift_pos) = m.shifted_linearity_up();
        let mut composed = vec![-1; self.storage.row_count()];
        for (orig_idx, &submatrix_idx) in newpos_submatrix.iter().enumerate() {
            if submatrix_idx >= 0 {
                let shifted_idx = shift_pos[submatrix_idx as usize];
                composed[orig_idx] = shifted_idx;
            }
        }
        (m, composed)
    }

    pub fn shifted_linearity_up(mut self) -> (LpMatrix<N, R>, RowIndex) {
        if self.linearity.is_empty() {
            let positions = (0..self.storage.row_count()).map(|v| v as isize).collect();
            return (self, positions);
        }
        let m = self.storage.row_count();
        let mut is_lin = vec![false; m];
        for idx in self.linearity.iter() {
            is_lin[idx.as_index()] = true;
        }

        let order = RowPermutation::partition_by_flags(&is_lin);

        let mut newpos = vec![0isize; m];
        for (new_idx, &old_idx) in order.as_slice().iter().enumerate() {
            newpos[old_idx] = new_idx as isize;
        }
        self.storage.reorder_rows_by_order(order.as_slice());
        let mut new_lin = RowSet::new(self.storage.row_count());
        let lin_count = self.linearity.cardinality();
        for i in 0..lin_count {
            new_lin.insert(i);
        }
        self.linearity = new_lin;
        (self, newpos)
    }

    pub fn select_columns(&self, columns: &[usize]) -> Result<LpMatrix<N, R>, Error> {
        if columns.is_empty() {
            return Err(Error::InvalidArgument {
                context: "select_columns requires a non-empty column list",
            });
        }
        let col_count = self.col_count_internal();
        let mut seen = vec![false; col_count];
        for &col in columns {
            if col >= col_count {
                return Err(Error::InvalidArgument {
                    context: "select_columns column out of bounds",
                });
            }
            if seen[col] {
                return Err(Error::InvalidArgument {
                    context: "select_columns requires unique columns",
                });
            }
            seen[col] = true;
        }

        let m = self.storage.row_count();
        let mut data = Vec::with_capacity(strict_mul(
            m,
            columns.len(),
            "select_columns allocation overflow",
        ));
        for row in self.storage.iter() {
            for &col in columns {
                data.push(row[col].clone());
            }
        }
        let storage = Matrix::from_flat(m, columns.len(), data);
        let filtered_row_vec = columns
            .iter()
            .map(|&col| self.row_vec[col].clone())
            .collect();

        Ok(self.rebuild_with_row_vec(storage, self.linearity.clone(), filtered_row_vec))
    }

    fn reorder_columns(&self, order: &[Col]) -> Result<LpMatrix<N, R>, Error> {
        let col_count = self.col_count_internal();
        if order.len() != col_count {
            return Err(Error::InvalidArgument {
                context: "reorder_columns order length mismatch",
            });
        }
        let mut seen = vec![false; col_count];
        for &col in order {
            if col >= col_count {
                return Err(Error::InvalidArgument {
                    context: "reorder_columns column out of bounds",
                });
            }
            if seen[col] {
                return Err(Error::InvalidArgument {
                    context: "reorder_columns requires a permutation",
                });
            }
            seen[col] = true;
        }

        let m = self.storage.row_count();
        let new_cols = order.len();
        let mut data = Vec::with_capacity(strict_mul(
            m,
            new_cols,
            "reorder_columns allocation overflow",
        ));
        for row in self.storage.iter() {
            for &orig_col in order {
                data.push(row[orig_col].clone());
            }
        }
        let storage = Matrix::from_flat(m, new_cols, data);
        let mut row_vec = Vec::with_capacity(new_cols);
        for &orig_col in order {
            row_vec.push(self.row_vec[orig_col].clone());
        }

        Ok(self.rebuild_with_row_vec(storage, self.linearity.clone(), row_vec))
    }

    fn prepare_columns(&self, columns: &[Col], sort: bool) -> Result<Vec<Col>, Error> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }
        let col_count = self.col_count_internal();
        if col_count == 0 {
            return Err(Error::InvalidArgument {
                context: "prepare_columns called with zero-column matrix",
            });
        }
        let mut seen = vec![false; col_count];
        let mut out = Vec::with_capacity(columns.len());
        for &col in columns {
            if col >= col_count {
                return Err(Error::InvalidArgument {
                    context: "prepare_columns column out of bounds",
                });
            }
            if seen[col] {
                return Err(Error::InvalidArgument {
                    context: "prepare_columns requires unique columns",
                });
            }
            seen[col] = true;
            out.push(col);
        }
        if out.len() >= col_count {
            return Err(Error::InvalidArgument {
                context: "prepare_columns cannot remove all columns",
            });
        }
        if sort {
            out.sort_unstable();
        }
        Ok(out)
    }

    pub(crate) fn canonical_row_order(
        &self,
        eps: &impl Epsilon<N>,
    ) -> (LpMatrix<N, R>, RowIndex, RowSet) {
        let m = self.storage.row_count();
        let cols = self.col_count_internal();
        if m == 0 || cols == 0 {
            return (self.clone(), Vec::new(), RowSet::new(m));
        }

        let mut data = Vec::with_capacity(strict_mul(m, cols, "canonical_row_order overflow"));
        for row in self.storage.iter() {
            let start = data.len();
            data.extend_from_slice(row);
            canonicalize_row(&mut data[start..start + cols], eps);
        }

        let mut order: Vec<usize> = (0..m).collect();
        order.sort_unstable_by(|&a, &b| {
            let a0 = a * cols;
            let b0 = b * cols;
            lex_cmp(&data[a0..a0 + cols], &data[b0..b0 + cols], eps)
        });

        let mut is_linearity = vec![false; m];
        for idx in self.linearity.iter() {
            is_linearity[idx.as_index()] = true;
        }

        let mut reps: Vec<usize> = Vec::new();
        let mut rep_is_lin: Vec<bool> = Vec::new();
        let mut newpos: RowIndex = vec![0; m];
        let mut duplicates = RowSet::new(m);

        let mut i = 0usize;
        while i < m {
            let start_old = order[i];
            let start0 = start_old * cols;
            let start_row = &data[start0..start0 + cols];

            let mut j = i + 1;
            let mut rep_old = start_old;
            let mut rep_found_lin = is_linearity[start_old];
            while j < m {
                let next_old = order[j];
                let next0 = next_old * cols;
                let next_row = &data[next0..next0 + cols];
                if !rows_equal(start_row, next_row, eps) {
                    break;
                }
                if !rep_found_lin && is_linearity[next_old] {
                    rep_old = next_old;
                    rep_found_lin = true;
                }
                j += 1;
            }

            let rep_id = reps.len();
            reps.push(rep_old);
            rep_is_lin.push(rep_found_lin);

            for &old in &order[i..j] {
                if old == rep_old {
                    newpos[old] = rep_id as isize;
                } else {
                    newpos[old] = -((rep_old as isize) + 1);
                    duplicates.insert(old);
                }
            }
            i = j;
        }

        let mut out_data = Vec::with_capacity(strict_mul(
            reps.len(),
            cols,
            "canonical_row_order output overflow",
        ));
        for &rep in &reps {
            let o0 = rep * cols;
            out_data.extend_from_slice(&data[o0..o0 + cols]);
        }

        let storage = Matrix::from_flat(reps.len(), cols, out_data);

        let mut new_linearity = RowSet::new(reps.len());
        for (idx, &is_lin) in rep_is_lin.iter().enumerate() {
            if is_lin {
                new_linearity.insert(idx);
            }
        }

        (
            self.clone_with_storage(storage, new_linearity),
            newpos,
            duplicates,
        )
    }

    pub(crate) fn prune_infeasible_rows(
        &self,
        eps: &impl Epsilon<N>,
    ) -> Result<(LpMatrix<N, R>, RowIndex, RowSet), Error> {
        let mut keep = self.linearity.clone();
        let mut removed = RowSet::new(self.storage.row_count());
        for i in 0..self.storage.row_count() {
            if keep.contains(i) {
                continue;
            }
            let mut trial_keep = keep.clone();
            trial_keep.insert(i);
            let delete = trial_keep.complement();
            let candidate = self.submatrix(&delete);
            let strict = RowSet::new(candidate.row_count());
            if candidate.restricted_face_exists(
                &candidate.linearity,
                &strict,
                LpSolver::DualSimplex,
                eps,
            )? {
                keep.insert(i);
            } else {
                removed.insert(i);
            }
        }
        if keep.is_empty() && !self.storage.is_empty() {
            keep.insert(0);
        }
        let delete = keep.complement();
        let (pruned, newpos) = self.submatrix_with_positions(&delete);
        Ok((pruned, newpos, removed))
    }

    pub fn zero_set_into(&self, vector: &[N], out: &mut RowSet, eps: &impl Epsilon<N>) {
        let cols = self.col_count_internal();
        debug_assert_eq!(vector.len(), cols);
        out.resize(self.storage.row_count());
        out.clear();
        for (i, row) in self.storage.iter().enumerate() {
            let sum = linalg::dot(row, vector);
            if eps.is_zero(&sum) {
                out.insert(i);
            }
        }
    }

    pub fn zero_set(&self, vector: &[N], eps: &impl Epsilon<N>) -> RowSet {
        let mut set = RowSet::new(self.storage.row_count());
        self.zero_set_into(vector, &mut set, eps);
        set
    }

    pub fn is_homogeneous(&self, eps: &impl Epsilon<N>) -> bool {
        self.storage
            .iter()
            .all(|row| row.first().is_none_or(|v| eps.is_zero(v)))
    }

    fn input_incidence_internal<S: Representation>(
        &self,
        outputs: &LpMatrix<N, S>,
        eps: &impl Epsilon<N>,
        newcol: Option<&[Option<usize>]>,
        add_slack_row: bool,
    ) -> Result<(SetFamily, RowSet, RowSet), Error> {
        let expected_cols = self.col_count_internal();
        if newcol.is_none() {
            if outputs.col_count_internal() != expected_cols {
                return Err(Error::InvalidArgument {
                    context: "incidence input/output column mismatch",
                });
            }
        }
        let remap_pairs = match newcol {
            None => None,
            Some(map) => {
                if map.len() != expected_cols {
                    return Err(Error::InvalidArgument {
                        context: "column mapping length mismatch",
                    });
                }
                let compact_cols = outputs.col_count_internal();
                let mut pairs = Vec::new();
                for m in map {
                    if let Some(pos) = *m
                        && pos >= compact_cols
                    {
                        return Err(Error::InvalidArgument {
                            context: "column mapping out of range",
                        });
                    }
                }
                for (orig_idx, m) in map.iter().enumerate() {
                    let Some(pos) = *m else { continue };
                    pairs.push((orig_idx, pos));
                }
                Some(pairs)
            }
        };

        let mut ainc = SetFamily::builder(self.storage.row_count(), outputs.row_count());
        let homogeneous = self.is_homogeneous(eps);
        let m = self.storage.row_count();
        let include_slack = add_slack_row
            && !homogeneous
            && self.representation() == RepresentationKind::Inequality;
        let m1 = if include_slack { m + 1 } else { m };
        if m1 > ainc.family_size() {
            ainc.resize(m1, outputs.row_count());
        }
        let mut zero = RowSet::new(m);
        let mut mapped = Vec::new();
        let mut touched: Vec<usize> = Vec::new();
        if let Some(pairs) = remap_pairs.as_ref() {
            mapped = vec![N::zero(); expected_cols];
            touched = Vec::with_capacity(pairs.len());
        }
        for (k, raw) in outputs.rows().enumerate() {
            if let Some(pairs) = remap_pairs.as_ref() {
                for &(orig_idx, pos) in pairs {
                    mapped[orig_idx] = raw[pos].clone();
                    touched.push(orig_idx);
                }
                self.zero_set_into(&mapped, &mut zero, eps);
                for i in zero.iter() {
                    ainc.insert_into_set(i.as_index(), RowId::new(k));
                }
                if include_slack && mapped.first().is_none_or(|v| eps.is_zero(v)) {
                    ainc.insert_into_set(m, RowId::new(k));
                }
                for &idx in &touched {
                    mapped[idx] = N::zero();
                }
                touched.clear();
            } else {
                self.zero_set_into(raw, &mut zero, eps);
                for i in zero.iter() {
                    ainc.insert_into_set(i.as_index(), RowId::new(k));
                }
                if include_slack && raw.first().is_none_or(|v| eps.is_zero(v)) {
                    ainc.insert_into_set(m, RowId::new(k));
                }
            }
        }

        let ainc = ainc.build();
        let (redundant, dominant) =
            ainc.classify_input_incidence(&self.linearity, outputs.row_count());
        Ok((ainc, redundant, dominant))
    }

    pub fn input_incidence_against<S: Representation>(
        &self,
        outputs: &LpMatrix<N, S>,
        eps: &impl Epsilon<N>,
        config: IncidenceConfig<'_>,
    ) -> Result<(SetFamily, RowSet, RowSet), Error> {
        self.input_incidence_internal(outputs, eps, config.column_mapping, config.include_slack)
    }

    pub fn input_adjacency_against<S: Representation>(
        &self,
        outputs: &LpMatrix<N, S>,
        eps: &impl Epsilon<N>,
        config: IncidenceConfig<'_>,
    ) -> Result<SetFamily, Error> {
        let (ainc, redundant, dominant) = self.input_incidence_against(outputs, eps, config)?;
        Ok(
            hullabaloo::adjacency::input_adjacency_from_incidence_set_family(
                &ainc, &redundant, &dominant,
            ),
        )
    }

    pub fn output_incidence_against<S: Representation>(
        &self,
        outputs: &LpMatrix<N, S>,
        eps: &impl Epsilon<N>,
        config: IncidenceConfig<'_>,
    ) -> Result<SetFamily, Error> {
        let (ainc, _, _) = self.input_incidence_against(outputs, eps, config)?;
        Ok(ainc.transpose())
    }

    pub fn copy_incidence(&self, eps: &impl Epsilon<N>) -> Result<SetFamily, Error> {
        self.weak_adjacency(eps)
    }

    pub fn copy_adjacency(&self, eps: &impl Epsilon<N>) -> Result<SetFamily, Error> {
        self.adjacency(eps)
    }
}

/// Methods only available for H-representation (inequality) matrices.
/// These operations are not defined for V-representation (generator) matrices.
impl<N: Num> LpMatrix<N, Inequality> {
    /// Performs Fourier-Motzkin elimination on the last column.
    pub fn fourier_elimination(&self, eps: &impl Epsilon<N>) -> Result<Self, Error> {
        if self.col_count_internal() <= 1 {
            return Err(Error::InvalidArgument {
                context: "fourier_elimination requires at least two columns",
            });
        }
        if !self.linearity.is_empty() {
            return Err(Error::CannotHandleLinearity);
        }

        let d = self.col_count_internal();
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        let mut zero = Vec::new();
        for (i, row) in self.storage.iter().enumerate() {
            let last = &row[d - 1];
            if eps.is_positive(last) {
                pos.push(i);
            } else if eps.is_negative(last) {
                neg.push(i);
            } else {
                zero.push(i);
            }
        }
        let mnew = zero.len() + pos.len() * neg.len();
        let dnew = d - 1;
        let mut data = vec![N::zero(); strict_mul(mnew, dnew, "fourier output overflow")];
        let mut row_vec = vec![N::zero(); dnew];
        row_vec[..].clone_from_slice(&self.row_vec[..dnew]);
        for (dest, &src_idx) in zero.iter().enumerate() {
            let dst = dest * dnew;
            let row = &self.storage[src_idx];
            data[dst..dst + dnew].clone_from_slice(&row[..dnew]);
        }
        let mut inew = zero.len();
        for &pi in &pos {
            for &ni in &neg {
                let row_p = &self.storage[pi];
                let row_n = &self.storage[ni];
                let neg_coeff = row_n[d - 1].ref_neg();
                let pos_coeff = row_p[d - 1].clone();
                let dst = inew * dnew;
                linalg::lin_comb2_into(
                    &mut data[dst..dst + dnew],
                    &row_p[..dnew],
                    &neg_coeff,
                    &row_n[..dnew],
                    &pos_coeff,
                );
                canonicalize_row(&mut data[dst..dst + dnew], eps);
                inew += 1;
            }
        }
        let storage = Matrix::from_flat(mnew, dnew, data);
        Ok(self.rebuild_with_row_vec(storage, RowSet::new(mnew), row_vec))
    }

    fn fourier_elimination_sequence(
        &self,
        targets: &[Col],
        canonicalize_each_step: bool,
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        if targets.is_empty() {
            return Ok(self.clone());
        }
        if !self.linearity.is_empty() {
            return Err(Error::CannotHandleLinearity);
        }

        let mut current = self.clone();
        let mut active: Vec<Col> = (0..self.col_count_internal()).collect();
        for &target in targets {
            if active.len() <= 1 {
                return Err(Error::InvalidArgument {
                    context: "fourier_elimination_sequence has no eliminable columns remaining",
                });
            }
            let Some(pos) = active.iter().position(|c| *c == target) else {
                return Err(Error::InvalidArgument {
                    context: "fourier_elimination_sequence target not active",
                });
            };
            let last_idx = active.len() - 1;
            if pos != last_idx {
                active.swap(pos, last_idx);
                for row in current.storage.iter_mut() {
                    row.swap(pos, last_idx);
                }
                current.row_vec.swap(pos, last_idx);
            }
            active.pop();
            current = current.fourier_elimination(eps)?;
            if canonicalize_each_step {
                current = current.coalesce_directions_max(eps);
                let canon = current.canonicalize(eps)?.into_matrix();
                current = canon.normalized_sorted_unique(eps).0;
            } else {
                current = current
                    .coalesce_directions_max(eps)
                    .normalized_sorted_unique(eps)
                    .0;
            }
        }
        Ok(current)
    }

    fn coalesce_directions_max(&self, eps: &impl Epsilon<N>) -> Self {
        if self.storage.is_empty() {
            return self.clone();
        }

        let m = self.storage.row_count();
        let cols = self.col_count_internal();
        if cols == 0 {
            return self.clone();
        }

        let mut canon = Vec::with_capacity(strict_mul(m, cols, "coalesce canonical buffer"));
        let mut is_lin = vec![false; m];
        for (i, row) in self.storage.iter().enumerate() {
            let start = canon.len();
            canon.extend_from_slice(row);
            canonicalize_row(&mut canon[start..start + cols], eps);
            is_lin[i] = self.linearity.contains(i);
        }

        let mut order: Vec<usize> = (0..m).collect();
        order.sort_unstable_by(|&a, &b| {
            let a0 = a * cols;
            let b0 = b * cols;
            lex_cmp(&canon[a0 + 1..a0 + cols], &canon[b0 + 1..b0 + cols], eps)
        });

        let mut out_data: Vec<N> = Vec::with_capacity(canon.len());
        let mut out_linearity = RowSet::new(0);

        let mut idx = 0usize;
        while idx < m {
            let first = order[idx];
            let first0 = first * cols;
            let dir_first = &canon[first0 + 1..first0 + cols];

            let mut min_idx = first;
            let mut max_idx = first;
            let mut group_is_lin = is_lin[first];

            idx += 1;
            while idx < m {
                let next = order[idx];
                let next0 = next * cols;
                if !rows_equal(dir_first, &canon[next0 + 1..next0 + cols], eps) {
                    break;
                }

                group_is_lin |= is_lin[next];

                if canon[next0]
                    .partial_cmp(&canon[min_idx * cols])
                    .map(|o| o.is_lt())
                    .unwrap_or(false)
                {
                    min_idx = next;
                }

                if canon[next0]
                    .partial_cmp(&canon[max_idx * cols])
                    .map(|o| o.is_gt())
                    .unwrap_or(false)
                {
                    max_idx = next;
                }

                idx += 1;
            }

            let base = out_data.len() / cols;
            out_data.extend_from_slice(&canon[min_idx * cols..min_idx * cols + cols]);
            if group_is_lin {
                out_linearity.resize(base + 1);
                out_linearity.insert(base);
            }

            if !rows_equal(
                &canon[min_idx * cols..min_idx * cols + cols],
                &canon[max_idx * cols..max_idx * cols + cols],
                eps,
            ) {
                let pos = out_data.len() / cols;
                out_data.extend_from_slice(&canon[max_idx * cols..max_idx * cols + cols]);
                if group_is_lin {
                    out_linearity.resize(pos + 1);
                    out_linearity.insert(pos);
                }
            }
        }

        let out_rows = out_data.len() / cols;
        out_linearity.resize(out_rows);
        let storage = Matrix::from_flat(out_rows, cols, out_data);
        self.clone_with_storage(storage, out_linearity)
    }

    fn validate_keep_columns(&self, keep_variables: &[Col]) -> Result<Vec<Col>, Error> {
        let col_count = self.col_count_internal();
        if keep_variables.is_empty() {
            return Ok(Vec::new());
        }
        let mut seen = vec![false; col_count];
        let mut out = Vec::with_capacity(keep_variables.len());
        for &col in keep_variables {
            if col >= col_count {
                return Err(Error::InvalidArgument {
                    context: "keep column out of bounds",
                });
            }
            if seen[col] {
                return Err(Error::InvalidArgument {
                    context: "keep columns must be unique",
                });
            }
            seen[col] = true;
            out.push(col);
        }
        Ok(out)
    }

    pub fn fourier_eliminate_columns(
        &self,
        columns_to_remove: &[Col],
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        let targets = self.prepare_columns(columns_to_remove, true)?;
        if targets.is_empty() {
            return Ok(self.clone());
        }

        let reduced = self.fourier_elimination_sequence(&targets, false, eps)?;
        let canon = reduced.canonicalize(eps)?.into_matrix();
        Ok(canon)
    }

    pub fn fourier_eliminate_last(
        &self,
        count: usize,
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        if count == 0 {
            return Ok(self.clone());
        }
        if count >= self.col_count_internal() {
            return Err(Error::InvalidArgument {
                context: "fourier_eliminate_last count out of bounds",
            });
        }
        let total = self.col_count_internal();
        let start = strict_sub(
            total,
            count,
            "fourier elimination count exceeds column count",
        );
        let cols: Vec<Col> = (start..total).collect();
        self.fourier_eliminate_columns(&cols, eps)
    }

    pub fn fourier_eliminate_columns_with_cleanup(
        &self,
        columns_to_remove: &[Col],
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        let targets = self.prepare_columns(columns_to_remove, false)?;
        if targets.is_empty() {
            return Ok(self.clone());
        }
        let reduced = self.fourier_elimination_sequence(&targets, false, eps)?;
        let canon = reduced.canonicalize(eps)?.into_matrix();
        Ok(canon)
    }

    pub fn fourier_project(
        &self,
        projection_dimension: usize,
        keep_variables: &[Col],
        eps: &impl Epsilon<N>,
    ) -> Result<Self, Error> {
        if !self.linearity.is_empty() {
            return Err(Error::CannotHandleLinearity);
        }
        let col_count = self.col_count_internal();
        let required_cols = projection_dimension
            .checked_add(1)
            .ok_or(Error::DimensionTooLarge)?;
        if required_cols > col_count {
            return Err(Error::InvalidArgument {
                context: "projection dimension too large",
            });
        }

        let keep = self.validate_keep_columns(keep_variables)?;
        let mut keep_flags = vec![false; col_count];
        keep_flags[0] = true; // Always keep constant term
        for &col in &keep {
            keep_flags[col] = true;
        }

        let kept = keep_flags.iter().filter(|flag| **flag).count();
        if kept != required_cols {
            return Err(Error::InvalidArgument {
                context: "keep column count mismatch",
            });
        }

        let mut removal_order = Vec::new();
        for (c, keep) in keep_flags.iter().enumerate() {
            if !*keep {
                removal_order.push(c);
            }
        }

        if removal_order.is_empty() {
            let canon = self.canonicalize(eps)?.into_matrix();
            return Ok(canon);
        }

        let mut removal_set = ColSet::new(col_count);
        for &c in &removal_order {
            removal_set.insert(c);
        }

        let mut column_order = vec![0]; // Start with constant
        for c in 1..col_count {
            if !removal_set.contains(c) {
                column_order.push(c);
            }
        }
        for c in 1..col_count {
            if removal_set.contains(c) {
                column_order.push(c);
            }
        }

        let reordered = self.reorder_columns(&column_order)?;

        let elim_start = strict_sub(
            reordered.col_count_internal(),
            removal_set.cardinality(),
            "block elimination removal exceeds column count",
        );
        let elimination_targets: Vec<Col> = (elim_start..reordered.col_count_internal()).collect();
        let cleaned = reordered
            .fourier_elimination_sequence(&elimination_targets, false, eps)?
            .normalized_sorted_unique(eps)
            .0;
        let canon = cleaned.canonicalize(eps)?.into_matrix();
        Ok(canon)
    }

    pub fn block_elimination(
        &self,
        columns_to_remove: &ColSet,
        eps: &(impl Epsilon<N> + Clone),
    ) -> Result<Self, Error>
    where
        N: DefaultNormalizer,
    {
        self.block_elimination_with_normalizer(
            columns_to_remove,
            eps,
            <N as DefaultNormalizer>::Norm::default(),
        )
    }

    pub fn block_elimination_with_normalizer<NM: Normalizer<N>>(
        &self,
        columns_to_remove: &ColSet,
        eps: &(impl Epsilon<N> + Clone),
        normalizer: NM,
    ) -> Result<Self, Error>
    {
        let del_indices: Vec<usize> = (0..self.col_count_internal())
            .filter(|c| columns_to_remove.contains(*c))
            .collect();
        if del_indices.is_empty() {
            return Ok(self.clone());
        }
        let m = self.storage.row_count();
        let d = self.col_count_internal();
        let delsize = del_indices.len();
        let linsize = self.linearity.cardinality();
        let mdual = delsize + strict_sub(m, linsize, "block elimination linearity exceeds rows");
        let ddual = m + 1;
        let mut dual_rows = Vec::with_capacity(strict_mul(mdual, ddual, "dual matrix overflow"));
        let mut dual_linearity = RowSet::new(mdual);
        for (i, &col_idx) in del_indices.iter().enumerate() {
            dual_linearity.insert(i);
            for j in 0..m {
                let row = &self.storage[j];
                dual_rows.push(row[col_idx].clone());
            }
            dual_rows.resize((i + 1) * ddual, N::zero());
        }
        let mut k = 0usize;
        for i in 0..m {
            if self.linearity.contains(i) {
                continue;
            }
            let row_idx = delsize + k;
            dual_rows.resize((row_idx + 1) * ddual, N::zero());
            let start = row_idx * ddual;
            dual_rows[start + i] = N::one();
            k += 1;
        }

        let dual_storage = Matrix::from_flat(mdual, ddual, dual_rows);

        let mut dual_builder = LpMatrixBuilder::<N, Inequality>::with_columns(ddual);
        dual_builder = dual_builder.with_storage(dual_storage);
        dual_builder = dual_builder.with_linearity(dual_linearity);
        let dual = dual_builder.build();

        let dual_poly = crate::polyhedron::Polyhedron::<N>::builder(dual)
            .cone_options(crate::dd::ConeOptions::default())
            .run_dd_with_eps_and_normalizer((*eps).clone(), normalizer)?;
        let gdual = dual_poly.into_output_required();
        let mproj = gdual.row_count();
        let dproj = d - delsize;
        let mut proj_rows =
            vec![N::zero(); strict_mul(mproj, dproj, "projection allocation overflow")];
        let proj_linearity = gdual.linearity().clone();
        let keep_columns: Vec<usize> = (0..d).filter(|&j| !columns_to_remove.contains(j)).collect();
        for (i, grow) in gdual.rows().enumerate() {
            let row_start = i * dproj;
            let row_slice = &mut proj_rows[row_start..row_start + dproj];
            for (constraint, coeff) in self.storage.iter().zip(grow.iter()) {
                if eps.is_zero(coeff) {
                    continue;
                }
                for (k_new, &j) in keep_columns.iter().enumerate() {
                    linalg::add_mul_assign(&mut row_slice[k_new], &constraint[j], coeff);
                }
            }
        }
        let storage = Matrix::from_flat(mproj, dproj, proj_rows);
        let mut builder = LpMatrixBuilder::<N, Inequality>::with_columns(dproj);
        builder = builder.with_storage(storage);
        builder = builder.with_linearity(proj_linearity);
        Ok(builder.build())
    }
}

fn rows_equal<N: Num>(a: &[N], b: &[N], eps: &impl Epsilon<N>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if eps.cmp(&a[i], &b[i]) != std::cmp::Ordering::Equal {
            return false;
        }
    }
    true
}

pub(crate) fn canonicalize_row<N: Num>(row: &mut [N], eps: &impl Epsilon<N>) {
    if row.is_empty() {
        return;
    }
    let mut min_abs: Option<N> = None;
    for v in row.iter() {
        let abs = v.abs();
        if eps.is_zero(&abs) {
            continue;
        }
        if min_abs
            .as_ref()
            .is_none_or(|current| abs.partial_cmp(current).is_some_and(|o| o.is_lt()))
        {
            min_abs = Some(abs);
        }
    }

    let Some(scale) = min_abs else {
        for v in row.iter_mut() {
            *v = N::zero();
        }
        return;
    };

    let inv = N::one().ref_div(&scale);
    for v in row.iter_mut() {
        *v = v.ref_mul(&inv);
        if eps.is_zero(v) {
            *v = N::zero();
        }
    }
}

pub(crate) fn compose_positions(first: &[isize], second: &[isize]) -> RowIndex {
    let mut out: RowIndex = vec![-1; first.len()];
    for (idx, &mid) in first.iter().enumerate() {
        if mid < 0 {
            out[idx] = mid;
            continue;
        }
        let mid_idx = mid as usize;
        out[idx] = second[mid_idx];
    }
    out
}

fn push_linearity(linearity: &RowSet, out: &mut String) {
    if linearity.is_empty() {
        return;
    }
    let _ = write!(out, "linearity {} ", linearity.cardinality());
    for idx in linearity.iter() {
        let _ = write!(out, "{idx} ");
    }
    out.push('\n');
}

fn push_matrix_block<N: Num + std::fmt::Display, R: Representation>(
    matrix: &LpMatrix<N, R>,
    out: &mut String,
) {
    out.push_str("begin\n");
    let kind = "real";
    let _ = writeln!(
        out,
        " {:4} {:4} {kind}",
        matrix.row_count(),
        matrix.col_count()
    );
    for row in matrix.rows() {
        for val in row {
            let _ = write!(out, " {}", val);
        }
        out.push('\n');
    }
    out.push_str("end\n");
}

fn push_objective_row<N: Num + std::fmt::Display, R: Representation>(
    matrix: &LpMatrix<N, R>,
    out: &mut String,
) {
    match matrix.objective {
        LpObjective::Maximize => out.push_str("maximize\n"),
        LpObjective::Minimize => out.push_str("minimize\n"),
        LpObjective::None => return,
    }
    for val in &matrix.row_vec {
        let _ = write!(out, " {}", val);
    }
    out.push('\n');
}

impl<N: Num + std::fmt::Display + 'static, R: Representation> LpMatrix<N, R> {
    pub fn to_cdd_string(&self) -> String {
        let mut out = String::new();
        match self.representation() {
            RepresentationKind::Inequality => out.push_str("H-representation\n"),
            RepresentationKind::Generator => out.push_str("V-representation\n"),
        }
        push_linearity(&self.linearity, &mut out);
        push_matrix_block(self, &mut out);
        push_objective_row(self, &mut out);
        out
    }

    pub fn to_cdd_normalized_string(&self, eps: &impl Epsilon<N>) -> (String, RowIndex) {
        let (normalized, positions) = self.normalized_sorted_unique(eps);
        (normalized.to_cdd_string(), positions)
    }
}

#[cfg(test)]
mod tests {
    use super::{LpMatrix, LpMatrixBuilder, canonicalize_row, compose_positions, lex_cmp};
    use calculo::linalg;
    use calculo::num::{Epsilon, Num};
    use hullabaloo::types::{Inequality, RowId, RowSet};
    use std::cmp::Ordering;

    #[test]
    fn lex_cmp_prefers_longer_when_values_match() {
        let eps = f64::default_eps();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(lex_cmp(&a, &b, &eps), Ordering::Less);
        assert_eq!(lex_cmp(&b, &a, &eps), Ordering::Greater);
    }

    #[test]
    fn lex_cmp_treats_small_differences_as_equal_for_floats() {
        let eps = f64::default_eps();
        let delta = *eps.eps() / 10.0;
        let a = vec![1.0, 1.0 + delta];
        let b = vec![1.0, 1.0];
        assert_eq!(lex_cmp(&a, &b, &eps), Ordering::Equal);
    }

    #[test]
    fn canonicalize_row_scales_by_min_abs_value() {
        let eps = f64::default_eps();
        let mut row = vec![2.0, -6.0, 4.0];
        canonicalize_row(&mut row, &eps);
        assert_eq!(row, vec![1.0, -3.0, 2.0]);
    }

    #[test]
    fn compose_positions_preserves_negative_mapping() {
        let first = vec![0, -2, 1, -1];
        let second = vec![10, 11];
        assert_eq!(compose_positions(&first, &second), vec![10, -2, 11, -1]);
    }

    #[test]
    fn canonical_row_order_marks_duplicates_with_representative_origin() {
        let builder: LpMatrixBuilder<f64, Inequality> = LpMatrixBuilder::with_columns(2)
            .push_row([1.0, 2.0], false)
            .push_row([1.0, 2.0], true);
        let matrix = builder.build();

        let eps = f64::default_eps();
        let (canon, newpos, duplicates) = matrix.canonical_row_order(&eps);

        assert_eq!(canon.row_count(), 1);
        assert_eq!(canon.row(0).unwrap(), &[1.0, 2.0]);
        assert!(canon.linearity().contains(RowId::new(0)));
        assert_eq!(newpos, vec![-2, 0]);
        assert!(duplicates.contains(RowId::new(0)));
        assert!(!duplicates.contains(RowId::new(1)));
    }

    #[test]
    fn canonical_row_order_keeps_lexicographic_row_ordering() {
        let builder: LpMatrixBuilder<f64, Inequality> = LpMatrixBuilder::with_columns(2)
            .push_row([2.0, 1.0], true)
            .push_row([1.0, 3.0], false);
        let matrix = builder.build();

        let eps = f64::default_eps();
        let (canon, newpos, duplicates) = matrix.canonical_row_order(&eps);

        assert_eq!(canon.row(0).unwrap(), &[1.0, 3.0]);
        assert_eq!(canon.row(1).unwrap(), &[2.0, 1.0]);
        assert_eq!(newpos, vec![1, 0]);
        assert!(canon.linearity().contains(RowId::new(1)));
        assert!(duplicates.is_empty());
    }

    #[test]
    fn solve_nullspace_1d_finds_axis_vector() {
        let eps = f64::default_eps();
        let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        let mut rows = RowSet::new(matrix.row_count());
        rows.insert(0);

        let v = matrix
            .storage()
            .solve_nullspace_1d(&rows, &eps)
            .expect("nullspace should be 1D");
        assert_eq!(v.len(), 2);
        let dot = linalg::dot(matrix.row(0).unwrap(), &v);
        assert!(eps.is_zero(&dot));
        assert!(eps.is_zero(&v[0]));
        assert!(!eps.is_zero(&v[1]));
    }

    #[test]
    fn solve_nullspace_1d_returns_none_when_dimension_is_not_one() {
        let eps = f64::default_eps();
        let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0, 0.0]]);

        let mut rows = RowSet::new(matrix.row_count());
        rows.insert(0);

        assert!(matrix.storage().solve_nullspace_1d(&rows, &eps).is_none());
    }

    #[test]
    fn solve_nullspace_1d_returns_none_for_full_rank_system() {
        let eps = f64::default_eps();
        let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        let mut rows = RowSet::new(matrix.row_count());
        rows.insert(0);
        rows.insert(1);

        assert!(matrix.storage().solve_nullspace_1d(&rows, &eps).is_none());
    }
}
