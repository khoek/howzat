use crate::{
    CddError, CddErrorCode, CddResult, CddWrapperError, LpObjective, LpStatus, NumberType,
    backend::CddNumber,
};
use hullabaloo::types::RepresentationKind;
use std::ffi::c_void;
use std::mem;
use std::os::raw::c_long;
use std::ptr::{self, NonNull};

#[repr(C)]
pub(crate) struct RawMatrixData {
    pub(crate) rowsize: c_long,
    pub(crate) linset: *mut libc::c_ulong,
    pub(crate) colsize: c_long,
    pub(crate) representation: u32,
    pub(crate) numbtype: u32,
    pub(crate) matrix: *mut *mut c_void,
    pub(crate) objective: u32,
    pub(crate) rowvec: *mut c_void,
}

#[repr(C)]
pub(crate) struct RawSetFamily {
    pub(crate) famsize: c_long,
    pub(crate) setsize: c_long,
    pub(crate) set: *mut *mut libc::c_ulong,
}

#[derive(Debug)]
pub(crate) struct MatrixData<N: CddNumber> {
    ptr: NonNull<RawMatrixData>,
    _marker: std::marker::PhantomData<N>,
}

pub(crate) type CanonicalizeOutput<N> = (MatrixData<N>, Vec<usize>, Vec<usize>, Vec<isize>);

impl<N: CddNumber> MatrixData<N> {
    pub(crate) fn new(
        rows: usize,
        cols: usize,
        repr: RepresentationKind,
        num_type: NumberType,
    ) -> CddResult<Self> {
        N::ensure_initialized();

        if rows == 0 || cols == 0 {
            return Err(CddWrapperError::InvalidMatrix { rows, cols });
        }

        let rowrange =
            c_long::try_from(rows).map_err(|_| CddWrapperError::InvalidMatrix { rows, cols })?;
        let colrange =
            c_long::try_from(cols).map_err(|_| CddWrapperError::InvalidMatrix { rows, cols })?;

        let ptr = unsafe { N::dd_create_matrix(rowrange, colrange) };
        let mut ptr = NonNull::new(ptr.cast::<RawMatrixData>()).ok_or(CddError::NullPointer)?;

        unsafe {
            ptr.as_mut().representation = crate::representation_to_raw(repr);
            ptr.as_mut().numbtype = num_type.to_raw();
        }

        Ok(Self {
            ptr,
            _marker: std::marker::PhantomData,
        })
    }

    pub(crate) fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr: NonNull::new(ptr.cast()).expect("from_raw called with null dd_MatrixPtr"),
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }

    #[inline(always)]
    pub(crate) fn rows(&self) -> usize {
        unsafe { self.ptr.as_ref().rowsize as usize }
    }

    #[inline(always)]
    pub(crate) fn cols(&self) -> usize {
        unsafe { self.ptr.as_ref().colsize as usize }
    }

    #[inline(always)]
    pub(crate) fn representation(&self) -> RepresentationKind {
        crate::representation_from_raw(unsafe { self.ptr.as_ref().representation })
    }

    #[inline(always)]
    pub(crate) fn number_type(&self) -> NumberType {
        unsafe { NumberType::from_raw(self.ptr.as_ref().numbtype) }
    }

    pub(crate) fn clone_cdd(&self) -> CddResult<Self> {
        N::ensure_initialized();

        let ptr = unsafe { N::dd_copy_matrix(self.as_raw()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(Self::from_raw(ptr.as_ptr()))
    }

    #[inline(always)]
    pub(crate) fn set_mytype(&mut self, row: usize, col: usize, value: &N) {
        assert!(row < self.rows());
        assert!(col < self.cols());
        let cell = unsafe { matrix_cell::<N>(self.ptr, row, col) };
        unsafe {
            N::write_mytype(cell, value);
        }
    }

    #[inline(always)]
    pub(crate) fn get_mytype(&self, row: usize, col: usize) -> N {
        assert!(row < self.rows());
        assert!(col < self.cols());
        let cell = unsafe { matrix_cell::<N>(self.ptr, row, col) };
        unsafe { N::read_mytype(cell.cast_const()) }
    }

    #[inline(always)]
    pub(crate) fn set_real(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.rows());
        assert!(col < self.cols());
        let cell = unsafe { matrix_cell::<N>(self.ptr, row, col) };
        unsafe {
            N::write_mytype_real(cell, value);
        }
    }

    #[inline(always)]
    pub(crate) fn set_int(&mut self, row: usize, col: usize, value: c_long) {
        assert!(row < self.rows());
        assert!(col < self.cols());
        let cell = unsafe { matrix_cell::<N>(self.ptr, row, col) };
        unsafe {
            N::write_mytype_int(cell, value);
        }
    }

    #[inline(always)]
    pub(crate) fn get_real(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows());
        assert!(col < self.cols());
        let cell = unsafe { matrix_cell::<N>(self.ptr, row, col) };
        unsafe { N::read_mytype_real(cell.cast_const()) }
    }

    #[inline(always)]
    pub(crate) fn set_generator_type(&mut self, row: usize, is_vertex: bool) {
        assert!(row < self.rows());
        let cell = unsafe { matrix_cell::<N>(self.ptr, row, 0) };
        let value = if is_vertex { 1 } else { 0 };
        unsafe {
            N::write_mytype_int(cell, value);
        }
    }

    pub(crate) fn set_objective_real(&mut self, coeffs: &[f64]) {
        assert_eq!(coeffs.len(), self.cols());
        let rowvec = unsafe { self.ptr.as_ref().rowvec };
        assert!(!rowvec.is_null(), "matrix objective rowvec is null");
        for (j, &c) in coeffs.iter().enumerate() {
            let cell = unsafe { rowvec_cell::<N>(rowvec, j) };
            unsafe {
                N::write_mytype_real(cell, c);
            }
        }
    }

    pub(crate) fn append_rows_in_place(&mut self, rows: &Self) -> CddResult<()> {
        N::ensure_initialized();

        if self.cols() != rows.cols() {
            return Err(CddWrapperError::InvalidMatrix {
                rows: self.rows() + rows.rows(),
                cols: rows.cols(),
            });
        }

        if self.representation() != rows.representation()
            || self.number_type() != rows.number_type()
        {
            return Err(CddWrapperError::InvalidMatrix {
                rows: self.rows() + rows.rows(),
                cols: rows.cols(),
            });
        }

        let mut raw = self.as_raw();
        let ok = unsafe { N::dd_matrix_append_to(&mut raw, rows.as_raw()) };
        if ok == 0 {
            return Err(CddError::OpFailed.into());
        }

        let raw = NonNull::new(raw.cast()).ok_or(CddError::NullPointer)?;
        if raw != self.ptr {
            self.ptr = raw;
        }
        Ok(())
    }

    pub(crate) fn append_rows(&self, rows: &Self) -> CddResult<Self> {
        N::ensure_initialized();

        if self.cols() != rows.cols() {
            return Err(CddWrapperError::InvalidMatrix {
                rows: self.rows() + rows.rows(),
                cols: rows.cols(),
            });
        }

        if self.representation() != rows.representation()
            || self.number_type() != rows.number_type()
        {
            return Err(CddWrapperError::InvalidMatrix {
                rows: self.rows() + rows.rows(),
                cols: rows.cols(),
            });
        }

        let ptr = unsafe { N::dd_append_matrix(self.as_raw(), rows.as_raw()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(Self::from_raw(ptr.as_ptr()))
    }

    pub(crate) fn remove_row(&mut self, row: usize) -> CddResult<()> {
        N::ensure_initialized();

        if row >= self.rows() {
            return Err(CddWrapperError::InvalidMatrix {
                rows: self.rows(),
                cols: self.cols(),
            });
        }

        let row_idx = c_long::try_from(row + 1).map_err(|_| CddWrapperError::InvalidMatrix {
            rows: self.rows(),
            cols: self.cols(),
        })?;

        let mut raw = self.as_raw();
        let ok = unsafe { N::dd_matrix_row_remove(&mut raw, row_idx) };
        if ok == 0 {
            return Err(CddError::OpFailed.into());
        }

        let raw = NonNull::new(raw.cast()).ok_or(CddError::NullPointer)?;
        if raw != self.ptr {
            self.ptr = raw;
        }
        Ok(())
    }

    pub(crate) fn is_row_redundant(&self, row: usize) -> CddResult<bool> {
        N::ensure_initialized();

        if row >= self.rows() {
            return Err(CddWrapperError::InvalidMatrix {
                rows: self.rows(),
                cols: self.cols(),
            });
        }

        let row_idx = c_long::try_from(row + 1).map_err(|_| CddWrapperError::InvalidMatrix {
            rows: self.rows(),
            cols: self.cols(),
        })?;

        let mut err = N::dd_no_error();
        let mut cert: *mut c_void = ptr::null_mut();

        let cert_cols = match self.representation() {
            RepresentationKind::Generator => self.cols() + 1,
            RepresentationKind::Inequality => self.cols(),
        };

        let colrange = c_long::try_from(cert_cols).map_err(|_| CddWrapperError::InvalidMatrix {
            rows: self.rows(),
            cols: self.cols(),
        })?;

        unsafe {
            N::dd_initialize_arow(colrange, &mut cert);
        }

        let redundant = unsafe { N::dd_redundant(self.as_raw(), row_idx, cert, &mut err) };

        unsafe {
            N::dd_free_arow(colrange, cert);
        }

        let err = CddErrorCode::from_raw(err);
        if err != CddErrorCode::NoError {
            return Err(CddError::Cdd(err).into());
        }
        Ok(redundant != 0)
    }

    pub(crate) fn redundant_rows(&self) -> CddResult<Vec<usize>> {
        N::ensure_initialized();

        let mut err = N::dd_no_error();
        let set = unsafe { N::dd_redundant_rows(self.as_raw(), &mut err) };
        let err = CddErrorCode::from_raw(err);
        if err != CddErrorCode::NoError {
            return Err(CddError::Cdd(err).into());
        }
        let set = NonNull::new(set).ok_or(CddError::NullPointer)?;

        let ground = unsafe { N::set_groundsize(set.as_ptr()) };
        let ground = usize::try_from(ground).map_err(|_| CddWrapperError::InvalidMatrix {
            rows: self.rows(),
            cols: self.cols(),
        })?;

        let mut rows = Vec::new();
        for i in 1..=ground {
            let member = unsafe { N::set_member(i as c_long, set.as_ptr()) };
            if member != 0 {
                rows.push(i - 1);
            }
        }

        unsafe {
            N::set_free(set.as_ptr());
        }

        Ok(rows)
    }

    pub(crate) fn canonicalize(&self) -> CddResult<CanonicalizeOutput<N>> {
        N::ensure_initialized();

        let mut canon = self.clone_cdd()?;
        let mut raw = canon.as_raw();
        let mut impl_lin: *mut libc::c_ulong = ptr::null_mut();
        let mut redset: *mut libc::c_ulong = ptr::null_mut();
        let mut newpos: *mut c_long = ptr::null_mut();
        let mut err = N::dd_no_error();

        let ok = unsafe {
            N::dd_matrix_canonicalize(&mut raw, &mut impl_lin, &mut redset, &mut newpos, &mut err)
        };
        let err = CddErrorCode::from_raw(err);
        if ok == 0 || err != CddErrorCode::NoError {
            unsafe {
                if !impl_lin.is_null() {
                    N::set_free(impl_lin);
                }
                if !redset.is_null() {
                    N::set_free(redset);
                }
                if !newpos.is_null() {
                    libc::free(newpos.cast());
                }
            }
            if ok == 0 && err == CddErrorCode::NoError {
                return Err(CddError::OpFailed.into());
            }
            return Err(CddError::Cdd(err).into());
        }

        canon.ptr = NonNull::new(raw.cast()).ok_or(CddError::NullPointer)?;
        let rows = canon.rows();
        let implicit = unsafe { set_members_bounded(impl_lin, rows) };
        let redundant = unsafe { set_members_bounded(redset, rows) };
        let positions = unsafe { read_rowindex(newpos, rows) };

        unsafe {
            N::set_free(impl_lin);
            N::set_free(redset);
            libc::free(newpos.cast());
        }

        Ok((canon, implicit, redundant, positions))
    }

    pub(crate) fn copy_objective_from<Src: CddNumber>(&mut self, src: &MatrixData<Src>) {
        unsafe {
            self.ptr.as_mut().objective = src.ptr.as_ref().objective;
        }
    }

    pub(crate) fn objective_row_coeffs_real(&self) -> Option<Vec<f64>> {
        let cols = self.cols();
        let rowvec = unsafe { self.ptr.as_ref().rowvec };
        if rowvec.is_null() {
            return None;
        }
        let mut coeffs = Vec::with_capacity(cols);
        for j in 0..cols {
            let cell = unsafe { rowvec_cell::<N>(rowvec, j) };
            coeffs.push(unsafe { N::read_mytype_real(cell.cast_const()) });
        }
        Some(coeffs)
    }
}

impl<N: CddNumber> Drop for MatrixData<N> {
    fn drop(&mut self) {
        unsafe {
            N::dd_free_matrix(self.as_raw());
        }
    }
}

#[derive(Debug)]
pub(crate) struct SetFamilyData<N: CddNumber> {
    ptr: NonNull<RawSetFamily>,
    _marker: std::marker::PhantomData<N>,
}

impl<N: CddNumber> SetFamilyData<N> {
    pub(crate) fn from_raw(ptr: *mut c_void) -> Self {
        Self {
            ptr: NonNull::new(ptr.cast()).expect("from_raw called with null dd_SetFamilyPtr"),
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        unsafe { self.ptr.as_ref().famsize as usize }
    }

    #[inline(always)]
    pub(crate) fn universe_size(&self) -> usize {
        unsafe { self.ptr.as_ref().setsize as usize }
    }

    pub(crate) fn members(&self, idx: usize) -> Vec<usize> {
        assert!(idx < self.len(), "set index out of bounds");
        let universe = self.universe_size();
        let set = unsafe { *self.ptr.as_ref().set.add(idx) };
        unsafe { set_members_bounded(set, universe) }
    }

    pub(crate) fn to_adjacency_lists(&self) -> Vec<Vec<usize>> {
        let n = self.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            result.push(self.members(i));
        }

        result
    }
}

impl<N: CddNumber> Drop for SetFamilyData<N> {
    fn drop(&mut self) {
        unsafe {
            N::dd_free_set_family(self.as_raw());
        }
    }
}

#[derive(Debug)]
pub(crate) struct PolyhedronData<N: CddNumber> {
    ptr: NonNull<c_void>,
    _marker: std::marker::PhantomData<N>,
}

impl<N: CddNumber> PolyhedronData<N> {
    pub(crate) fn from_matrix(m: &MatrixData<N>) -> CddResult<Self> {
        N::ensure_initialized();
        let mut err = N::dd_no_error();
        let ptr = unsafe { N::dd_matrix2poly(m.as_raw(), &mut err) };
        let err = CddErrorCode::from_raw(err);
        if err != CddErrorCode::NoError {
            return Err(CddError::Cdd(err).into());
        }
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(Self {
            ptr,
            _marker: std::marker::PhantomData,
        })
    }

    pub(crate) fn facets(&self) -> CddResult<MatrixData<N>> {
        let ptr = unsafe { N::dd_copy_inequalities(self.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(MatrixData::<N>::from_raw(ptr.as_ptr()))
    }

    pub(crate) fn generators(&self) -> CddResult<MatrixData<N>> {
        let ptr = unsafe { N::dd_copy_generators(self.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(MatrixData::<N>::from_raw(ptr.as_ptr()))
    }

    pub(crate) fn adjacency(&self) -> CddResult<SetFamilyData<N>> {
        let ptr = unsafe { N::dd_copy_adjacency(self.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(SetFamilyData::<N>::from_raw(ptr.as_ptr()))
    }

    pub(crate) fn input_adjacency(&self) -> CddResult<SetFamilyData<N>> {
        let ptr = unsafe { N::dd_copy_input_adjacency(self.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(SetFamilyData::<N>::from_raw(ptr.as_ptr()))
    }

    pub(crate) fn incidence(&self) -> CddResult<SetFamilyData<N>> {
        let ptr = unsafe { N::dd_copy_incidence(self.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(SetFamilyData::<N>::from_raw(ptr.as_ptr()))
    }

    pub(crate) fn input_incidence(&self) -> CddResult<SetFamilyData<N>> {
        let ptr = unsafe { N::dd_copy_input_incidence(self.ptr.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(SetFamilyData::<N>::from_raw(ptr.as_ptr()))
    }

    pub(crate) fn append_input_rows(&mut self, rows: &MatrixData<N>) -> CddResult<()> {
        let mut raw = self.ptr.as_ptr();
        let ok = unsafe { N::dd_append_matrix2poly(&mut raw, rows.as_raw()) };
        if ok == 0 {
            return Err(CddError::OpFailed.into());
        }
        if raw != self.ptr.as_ptr() {
            self.ptr = NonNull::new(raw).ok_or(CddError::NullPointer)?;
        }
        Ok(())
    }
}

impl<N: CddNumber> Drop for PolyhedronData<N> {
    fn drop(&mut self) {
        unsafe {
            N::dd_free_polyhedra(self.ptr.as_ptr());
        }
    }
}

#[derive(Debug)]
pub(crate) struct LpData<N: CddNumber> {
    ptr: NonNull<c_void>,
    _marker: std::marker::PhantomData<N>,
}

impl<N: CddNumber> LpData<N> {
    pub(crate) fn from_matrix(
        matrix: &mut MatrixData<N>,
        objective: LpObjective,
    ) -> CddResult<Self> {
        N::ensure_initialized();
        unsafe {
            matrix.ptr.as_mut().objective = objective.to_raw();
        }
        let mut err = N::dd_no_error();
        let ptr = unsafe { N::dd_matrix2lp(matrix.as_raw(), &mut err) };
        let err = CddErrorCode::from_raw(err);
        if err != CddErrorCode::NoError {
            return Err(CddError::Cdd(err).into());
        }
        let ptr = NonNull::new(ptr).ok_or(CddError::NullPointer)?;
        Ok(Self {
            ptr,
            _marker: std::marker::PhantomData,
        })
    }

    pub(crate) fn solve(&self) -> CddResult<LpSolutionData<N>> {
        N::ensure_initialized();

        let mut err = N::dd_no_error();
        let ok = unsafe { N::dd_lp_solve_dual_simplex(self.ptr.as_ptr(), &mut err) };
        let err = CddErrorCode::from_raw(err);
        if err != CddErrorCode::NoError {
            return Err(CddError::Cdd(err).into());
        }
        if ok == 0 {
            return Err(CddError::LpError.into());
        }
        let status = unsafe { N::lp_status_raw(self.ptr.as_ptr()) };
        let status = LpStatus::from_raw(status);
        if status != LpStatus::Optimal {
            return Err(CddError::LpStatus(status).into());
        }
        let sol_ptr = unsafe { N::dd_copy_lp_solution(self.ptr.as_ptr()) };
        let sol_ptr = NonNull::new(sol_ptr).ok_or(CddError::NullPointer)?;
        Ok(LpSolutionData {
            ptr: sol_ptr,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<N: CddNumber> Drop for LpData<N> {
    fn drop(&mut self) {
        unsafe {
            N::dd_free_lp_data(self.ptr.as_ptr());
        }
    }
}

#[derive(Debug)]
pub(crate) struct LpSolutionData<N: CddNumber> {
    ptr: NonNull<c_void>,
    _marker: std::marker::PhantomData<N>,
}

impl<N: CddNumber> LpSolutionData<N> {
    pub(crate) fn opt_value_real(&self) -> f64 {
        unsafe {
            let opt_ptr = N::lp_solution_optvalue_ptr(self.ptr.as_ptr());
            N::read_mytype_real(opt_ptr)
        }
    }
}

impl<N: CddNumber> Drop for LpSolutionData<N> {
    fn drop(&mut self) {
        unsafe {
            N::dd_free_lp_solution(self.ptr.as_ptr());
        }
    }
}

unsafe fn matrix_cell<N: CddNumber>(
    matrix: NonNull<RawMatrixData>,
    row: usize,
    col: usize,
) -> *mut c_void {
    unsafe {
        let matrix_rows = matrix.as_ref().matrix;
        assert!(!matrix_rows.is_null(), "matrix rows pointer is null");
        let row_ptr = *matrix_rows.add(row);
        assert!(!row_ptr.is_null(), "matrix row pointer is null");
        row_ptr
            .cast::<u8>()
            .add(col * N::MYTYPE_SIZE)
            .cast::<c_void>()
    }
}

unsafe fn rowvec_cell<N: CddNumber>(rowvec: *mut c_void, col: usize) -> *mut c_void {
    unsafe {
        rowvec
            .cast::<u8>()
            .add(col * N::MYTYPE_SIZE)
            .cast::<c_void>()
    }
}

unsafe fn set_members_bounded(set: *mut libc::c_ulong, max_elems: usize) -> Vec<usize> {
    assert!(!set.is_null(), "set pointer is null");

    let ground = unsafe { *set } as usize;
    let bound = max_elems.min(ground);
    if bound == 0 {
        return Vec::new();
    }

    let bits_per_block = mem::size_of::<libc::c_ulong>() * 8;
    let blocks = (bound - 1) / bits_per_block + 1;

    let mut out = Vec::new();
    for block_idx in 0..blocks {
        let mut word = unsafe { *set.add(1 + block_idx) };
        let base = block_idx * bits_per_block;
        while word != 0 {
            let tz = word.trailing_zeros() as usize;
            let idx = base + tz;
            if idx < bound {
                out.push(idx);
            }
            word &= word - 1;
        }
    }

    out
}

unsafe fn read_rowindex(index: *mut c_long, rows: usize) -> Vec<isize> {
    unsafe {
        let mut out = Vec::with_capacity(rows);
        for i in 1..=rows {
            out.push(*index.add(i) as isize);
        }
        out
    }
}
