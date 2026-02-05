// Internal backend bindings and numeric backends.
//
// This module owns the `CddNumber` trait and all backend-specific FFI glue.

use std::ffi::c_void;
use std::mem;
use std::os::raw::c_long;
use std::sync::Once;

#[cfg(any(feature = "gmp", feature = "gmprational"))]
use std::mem::MaybeUninit;

use crate::NumberType;

use cddlib_sys as sys;

#[cfg(feature = "gmp")]
pub struct CddFloat {
    inner: sys::gmpfloat::mytype,
}

#[cfg(feature = "gmprational")]
pub struct CddRational {
    inner: sys::gmprational::mytype,
}

#[cfg(feature = "f64")]
pub type DefaultNumber = f64;

#[cfg(all(not(feature = "f64"), feature = "gmprational"))]
pub type DefaultNumber = CddRational;

#[cfg(all(not(feature = "f64"), not(feature = "gmprational"), feature = "gmp"))]
pub type DefaultNumber = CddFloat;

mod sealed {
    pub trait Sealed {}
}

pub trait CddNumber: sealed::Sealed + 'static {
    const MYTYPE_SIZE: usize;
    const DEFAULT_NUMBER_TYPE: NumberType;

    fn ensure_initialized();

    #[doc(hidden)]
    fn dd_no_error() -> u32;

    #[doc(hidden)]
    unsafe fn dd_create_matrix(rows: c_long, cols: c_long) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_copy_matrix(matrix: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_free_matrix(matrix: *mut c_void);

    #[doc(hidden)]
    unsafe fn dd_matrix_append_to(dst: *mut *mut c_void, src: *mut c_void) -> i32;

    #[doc(hidden)]
    unsafe fn dd_append_matrix(a: *mut c_void, b: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_matrix_row_remove(matrix: *mut *mut c_void, row: c_long) -> i32;

    #[doc(hidden)]
    unsafe fn dd_initialize_arow(cols: c_long, out: *mut *mut c_void);

    #[doc(hidden)]
    unsafe fn dd_free_arow(cols: c_long, arow: *mut c_void);

    #[doc(hidden)]
    unsafe fn dd_redundant(
        matrix: *mut c_void,
        row: c_long,
        cert: *mut c_void,
        err: *mut u32,
    ) -> i32;

    #[doc(hidden)]
    unsafe fn dd_redundant_rows(matrix: *mut c_void, err: *mut u32) -> *mut libc::c_ulong;

    #[doc(hidden)]
    unsafe fn dd_matrix_canonicalize(
        matrix: *mut *mut c_void,
        impl_lin: *mut *mut libc::c_ulong,
        redset: *mut *mut libc::c_ulong,
        newpos: *mut *mut c_long,
        err: *mut u32,
    ) -> i32;

    #[doc(hidden)]
    unsafe fn set_groundsize(set: *mut libc::c_ulong) -> c_long;

    #[doc(hidden)]
    unsafe fn set_member(elem: c_long, set: *mut libc::c_ulong) -> i32;

    #[doc(hidden)]
    unsafe fn set_free(set: *mut libc::c_ulong);

    #[doc(hidden)]
    unsafe fn dd_matrix2poly(matrix: *mut c_void, err: *mut u32) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_free_polyhedra(poly: *mut c_void);

    #[doc(hidden)]
    unsafe fn dd_copy_inequalities(poly: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_copy_generators(poly: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_copy_adjacency(poly: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_copy_input_adjacency(poly: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_copy_incidence(poly: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_copy_input_incidence(poly: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_append_matrix2poly(poly: *mut *mut c_void, rows: *mut c_void) -> i32;

    #[doc(hidden)]
    unsafe fn dd_free_set_family(family: *mut c_void);

    #[doc(hidden)]
    unsafe fn dd_matrix2lp(matrix: *mut c_void, err: *mut u32) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_free_lp_data(lp: *mut c_void);

    #[doc(hidden)]
    unsafe fn dd_lp_solve_dual_simplex(lp: *mut c_void, err: *mut u32) -> i32;

    #[doc(hidden)]
    unsafe fn lp_status_raw(lp: *mut c_void) -> u32;

    #[doc(hidden)]
    unsafe fn dd_copy_lp_solution(lp: *mut c_void) -> *mut c_void;

    #[doc(hidden)]
    unsafe fn dd_free_lp_solution(sol: *mut c_void);

    #[doc(hidden)]
    unsafe fn lp_solution_optvalue_ptr(sol: *mut c_void) -> *const c_void;

    #[doc(hidden)]
    unsafe fn write_mytype_real(target: *mut c_void, value: f64);

    #[doc(hidden)]
    unsafe fn write_mytype_int(target: *mut c_void, value: c_long);

    #[doc(hidden)]
    unsafe fn read_mytype_real(source: *const c_void) -> f64;

    #[doc(hidden)]
    unsafe fn write_mytype(target: *mut c_void, value: &Self);

    #[doc(hidden)]
    unsafe fn read_mytype(source: *const c_void) -> Self;
}

#[cfg(feature = "f64")]
impl sealed::Sealed for f64 {}

#[cfg(feature = "f64")]
impl CddNumber for f64 {
    const MYTYPE_SIZE: usize = mem::size_of::<sys::f64::mytype>();
    const DEFAULT_NUMBER_TYPE: NumberType = NumberType::Real;

    fn ensure_initialized() {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe {
            sys::f64::dd_set_global_constants();
        });
    }

    fn dd_no_error() -> u32 {
        sys::f64::dd_ErrorType_dd_NoError
    }

    unsafe fn dd_create_matrix(rows: c_long, cols: c_long) -> *mut c_void {
        unsafe { sys::f64::dd_CreateMatrix(rows, cols).cast() }
    }

    unsafe fn dd_copy_matrix(matrix: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyMatrix(matrix.cast()).cast() }
    }

    unsafe fn dd_free_matrix(matrix: *mut c_void) {
        unsafe {
            sys::f64::dd_FreeMatrix(matrix.cast());
        }
    }

    unsafe fn dd_matrix_append_to(dst: *mut *mut c_void, src: *mut c_void) -> i32 {
        unsafe { sys::f64::dd_MatrixAppendTo(dst.cast(), src.cast()) }
    }

    unsafe fn dd_append_matrix(a: *mut c_void, b: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_AppendMatrix(a.cast(), b.cast()).cast() }
    }

    unsafe fn dd_matrix_row_remove(matrix: *mut *mut c_void, row: c_long) -> i32 {
        unsafe { sys::f64::dd_MatrixRowRemove(matrix.cast(), row) }
    }

    unsafe fn dd_initialize_arow(cols: c_long, out: *mut *mut c_void) {
        unsafe {
            sys::f64::dd_InitializeArow(cols, out.cast());
        }
    }

    unsafe fn dd_free_arow(cols: c_long, arow: *mut c_void) {
        unsafe {
            sys::f64::dd_FreeArow(cols, arow.cast());
        }
    }

    unsafe fn dd_redundant(
        matrix: *mut c_void,
        row: c_long,
        cert: *mut c_void,
        err: *mut u32,
    ) -> i32 {
        unsafe { sys::f64::dd_Redundant(matrix.cast(), row, cert.cast(), err.cast()) }
    }

    unsafe fn dd_redundant_rows(matrix: *mut c_void, err: *mut u32) -> *mut libc::c_ulong {
        unsafe { sys::f64::dd_RedundantRows(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_matrix_canonicalize(
        matrix: *mut *mut c_void,
        impl_lin: *mut *mut libc::c_ulong,
        redset: *mut *mut libc::c_ulong,
        newpos: *mut *mut c_long,
        err: *mut u32,
    ) -> i32 {
        unsafe {
            sys::f64::dd_MatrixCanonicalize(
                matrix.cast(),
                impl_lin.cast(),
                redset.cast(),
                newpos.cast(),
                err.cast(),
            )
        }
    }

    unsafe fn set_groundsize(set: *mut libc::c_ulong) -> c_long {
        unsafe { sys::f64::set_groundsize(set.cast()) }
    }

    unsafe fn set_member(elem: c_long, set: *mut libc::c_ulong) -> i32 {
        unsafe { sys::f64::set_member(elem, set.cast()) }
    }

    unsafe fn set_free(set: *mut libc::c_ulong) {
        unsafe {
            sys::f64::set_free(set.cast());
        }
    }

    unsafe fn dd_matrix2poly(matrix: *mut c_void, err: *mut u32) -> *mut c_void {
        unsafe { sys::f64::dd_DDMatrix2Poly(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_free_polyhedra(poly: *mut c_void) {
        unsafe {
            sys::f64::dd_FreePolyhedra(poly.cast());
        }
    }

    unsafe fn dd_copy_inequalities(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyInequalities(poly.cast()).cast() }
    }

    unsafe fn dd_copy_generators(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyGenerators(poly.cast()).cast() }
    }

    unsafe fn dd_copy_adjacency(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyAdjacency(poly.cast()).cast() }
    }

    unsafe fn dd_copy_input_adjacency(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyInputAdjacency(poly.cast()).cast() }
    }

    unsafe fn dd_copy_incidence(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyIncidence(poly.cast()).cast() }
    }

    unsafe fn dd_copy_input_incidence(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyInputIncidence(poly.cast()).cast() }
    }

    unsafe fn dd_append_matrix2poly(poly: *mut *mut c_void, rows: *mut c_void) -> i32 {
        unsafe { sys::f64::dd_AppendMatrix2Poly(poly.cast(), rows.cast()) }
    }

    unsafe fn dd_free_set_family(family: *mut c_void) {
        unsafe {
            sys::f64::dd_FreeSetFamily(family.cast());
        }
    }

    unsafe fn dd_matrix2lp(matrix: *mut c_void, err: *mut u32) -> *mut c_void {
        unsafe { sys::f64::dd_Matrix2LP(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_free_lp_data(lp: *mut c_void) {
        unsafe {
            sys::f64::dd_FreeLPData(lp.cast());
        }
    }

    unsafe fn dd_lp_solve_dual_simplex(lp: *mut c_void, err: *mut u32) -> i32 {
        unsafe {
            sys::f64::dd_LPSolve(
                lp.cast(),
                sys::f64::dd_LPSolverType_dd_DualSimplex,
                err.cast(),
            )
        }
    }

    unsafe fn lp_status_raw(lp: *mut c_void) -> u32 {
        unsafe { (*lp.cast::<sys::f64::dd_lpdata>()).LPS }
    }

    unsafe fn dd_copy_lp_solution(lp: *mut c_void) -> *mut c_void {
        unsafe { sys::f64::dd_CopyLPSolution(lp.cast()).cast() }
    }

    unsafe fn dd_free_lp_solution(sol: *mut c_void) {
        unsafe {
            sys::f64::dd_FreeLPSolution(sol.cast());
        }
    }

    unsafe fn lp_solution_optvalue_ptr(sol: *mut c_void) -> *const c_void {
        unsafe {
            let sol = sol.cast::<sys::f64::dd_lpsolution>();
            (&(*sol).optvalue as *const sys::f64::mytype).cast()
        }
    }

    unsafe fn write_mytype_real(target: *mut c_void, value: f64) {
        unsafe {
            (*target.cast::<sys::f64::mytype>())[0] = value;
        }
    }

    unsafe fn write_mytype_int(target: *mut c_void, value: c_long) {
        unsafe {
            (*target.cast::<sys::f64::mytype>())[0] = value as f64;
        }
    }

    unsafe fn read_mytype_real(source: *const c_void) -> f64 {
        unsafe { (*source.cast::<sys::f64::mytype>())[0] }
    }

    unsafe fn write_mytype(target: *mut c_void, value: &Self) {
        unsafe {
            Self::write_mytype_real(target, *value);
        }
    }

    unsafe fn read_mytype(source: *const c_void) -> Self {
        unsafe { Self::read_mytype_real(source) }
    }
}

#[cfg(feature = "gmp")]
impl sealed::Sealed for CddFloat {}

#[cfg(feature = "gmp")]
impl CddNumber for CddFloat {
    const MYTYPE_SIZE: usize = mem::size_of::<sys::gmpfloat::mytype>();
    const DEFAULT_NUMBER_TYPE: NumberType = NumberType::Real;

    fn ensure_initialized() {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe {
            sys::gmpfloat::dd_set_global_constants();
        });
    }

    fn dd_no_error() -> u32 {
        sys::gmpfloat::dd_ErrorType_dd_NoError
    }

    unsafe fn dd_create_matrix(rows: c_long, cols: c_long) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CreateMatrix(rows, cols).cast() }
    }

    unsafe fn dd_copy_matrix(matrix: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyMatrix(matrix.cast()).cast() }
    }

    unsafe fn dd_free_matrix(matrix: *mut c_void) {
        unsafe {
            sys::gmpfloat::dd_FreeMatrix(matrix.cast());
        }
    }

    unsafe fn dd_matrix_append_to(dst: *mut *mut c_void, src: *mut c_void) -> i32 {
        unsafe { sys::gmpfloat::dd_MatrixAppendTo(dst.cast(), src.cast()) }
    }

    unsafe fn dd_append_matrix(a: *mut c_void, b: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_AppendMatrix(a.cast(), b.cast()).cast() }
    }

    unsafe fn dd_matrix_row_remove(matrix: *mut *mut c_void, row: c_long) -> i32 {
        unsafe { sys::gmpfloat::dd_MatrixRowRemove(matrix.cast(), row) }
    }

    unsafe fn dd_initialize_arow(cols: c_long, out: *mut *mut c_void) {
        unsafe {
            sys::gmpfloat::dd_InitializeArow(cols, out.cast());
        }
    }

    unsafe fn dd_free_arow(cols: c_long, arow: *mut c_void) {
        unsafe {
            sys::gmpfloat::dd_FreeArow(cols, arow.cast());
        }
    }

    unsafe fn dd_redundant(
        matrix: *mut c_void,
        row: c_long,
        cert: *mut c_void,
        err: *mut u32,
    ) -> i32 {
        unsafe { sys::gmpfloat::dd_Redundant(matrix.cast(), row, cert.cast(), err.cast()) }
    }

    unsafe fn dd_redundant_rows(matrix: *mut c_void, err: *mut u32) -> *mut libc::c_ulong {
        unsafe { sys::gmpfloat::dd_RedundantRows(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_matrix_canonicalize(
        matrix: *mut *mut c_void,
        impl_lin: *mut *mut libc::c_ulong,
        redset: *mut *mut libc::c_ulong,
        newpos: *mut *mut c_long,
        err: *mut u32,
    ) -> i32 {
        unsafe {
            sys::gmpfloat::dd_MatrixCanonicalize(
                matrix.cast(),
                impl_lin.cast(),
                redset.cast(),
                newpos.cast(),
                err.cast(),
            )
        }
    }

    unsafe fn set_groundsize(set: *mut libc::c_ulong) -> c_long {
        unsafe { sys::gmpfloat::set_groundsize(set.cast()) }
    }

    unsafe fn set_member(elem: c_long, set: *mut libc::c_ulong) -> i32 {
        unsafe { sys::gmpfloat::set_member(elem, set.cast()) }
    }

    unsafe fn set_free(set: *mut libc::c_ulong) {
        unsafe {
            sys::gmpfloat::set_free(set.cast());
        }
    }

    unsafe fn dd_matrix2poly(matrix: *mut c_void, err: *mut u32) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_DDMatrix2Poly(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_free_polyhedra(poly: *mut c_void) {
        unsafe {
            sys::gmpfloat::dd_FreePolyhedra(poly.cast());
        }
    }

    unsafe fn dd_copy_inequalities(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyInequalities(poly.cast()).cast() }
    }

    unsafe fn dd_copy_generators(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyGenerators(poly.cast()).cast() }
    }

    unsafe fn dd_copy_adjacency(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyAdjacency(poly.cast()).cast() }
    }

    unsafe fn dd_copy_input_adjacency(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyInputAdjacency(poly.cast()).cast() }
    }

    unsafe fn dd_copy_incidence(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyIncidence(poly.cast()).cast() }
    }

    unsafe fn dd_copy_input_incidence(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyInputIncidence(poly.cast()).cast() }
    }

    unsafe fn dd_append_matrix2poly(poly: *mut *mut c_void, rows: *mut c_void) -> i32 {
        unsafe { sys::gmpfloat::dd_AppendMatrix2Poly(poly.cast(), rows.cast()) }
    }

    unsafe fn dd_free_set_family(family: *mut c_void) {
        unsafe {
            sys::gmpfloat::dd_FreeSetFamily(family.cast());
        }
    }

    unsafe fn dd_matrix2lp(matrix: *mut c_void, err: *mut u32) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_Matrix2LP(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_free_lp_data(lp: *mut c_void) {
        unsafe {
            sys::gmpfloat::dd_FreeLPData(lp.cast());
        }
    }

    unsafe fn dd_lp_solve_dual_simplex(lp: *mut c_void, err: *mut u32) -> i32 {
        unsafe {
            sys::gmpfloat::dd_LPSolve(
                lp.cast(),
                sys::gmpfloat::dd_LPSolverType_dd_DualSimplex,
                err.cast(),
            )
        }
    }

    unsafe fn lp_status_raw(lp: *mut c_void) -> u32 {
        unsafe { (*lp.cast::<sys::gmpfloat::dd_lpdata>()).LPS }
    }

    unsafe fn dd_copy_lp_solution(lp: *mut c_void) -> *mut c_void {
        unsafe { sys::gmpfloat::dd_CopyLPSolution(lp.cast()).cast() }
    }

    unsafe fn dd_free_lp_solution(sol: *mut c_void) {
        unsafe {
            sys::gmpfloat::dd_FreeLPSolution(sol.cast());
        }
    }

    unsafe fn lp_solution_optvalue_ptr(sol: *mut c_void) -> *const c_void {
        unsafe {
            let sol = sol.cast::<sys::gmpfloat::dd_lpsolution>();
            (&(*sol).optvalue as *const sys::gmpfloat::mytype).cast()
        }
    }

    unsafe fn write_mytype_real(target: *mut c_void, value: f64) {
        unsafe {
            sys::gmpfloat::__gmpf_set_d(target.cast(), value);
        }
    }

    unsafe fn write_mytype_int(target: *mut c_void, value: c_long) {
        unsafe {
            sys::gmpfloat::__gmpf_set_si(target.cast(), value);
        }
    }

    unsafe fn read_mytype_real(source: *const c_void) -> f64 {
        unsafe { sys::gmpfloat::__gmpf_get_d(source.cast()) }
    }

    unsafe fn write_mytype(target: *mut c_void, value: &Self) {
        unsafe {
            sys::gmpfloat::__gmpf_set(target.cast(), value.inner.as_ptr());
        }
    }

    unsafe fn read_mytype(source: *const c_void) -> Self {
        unsafe {
            Self::ensure_initialized();
            let mut inner = MaybeUninit::<sys::gmpfloat::mytype>::uninit();
            sys::gmpfloat::__gmpf_init(inner.as_mut_ptr().cast());
            sys::gmpfloat::__gmpf_set(inner.as_mut_ptr().cast(), source.cast());
            CddFloat {
                inner: inner.assume_init(),
            }
        }
    }
}

#[cfg(feature = "gmprational")]
impl sealed::Sealed for CddRational {}

#[cfg(feature = "gmprational")]
impl CddNumber for CddRational {
    const MYTYPE_SIZE: usize = mem::size_of::<sys::gmprational::mytype>();
    const DEFAULT_NUMBER_TYPE: NumberType = NumberType::Rational;

    fn ensure_initialized() {
        static INIT: Once = Once::new();
        INIT.call_once(|| unsafe {
            sys::gmprational::dd_set_global_constants();
        });
    }

    fn dd_no_error() -> u32 {
        sys::gmprational::dd_ErrorType_dd_NoError
    }

    unsafe fn dd_create_matrix(rows: c_long, cols: c_long) -> *mut c_void {
        unsafe { sys::gmprational::dd_CreateMatrix(rows, cols).cast() }
    }

    unsafe fn dd_copy_matrix(matrix: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyMatrix(matrix.cast()).cast() }
    }

    unsafe fn dd_free_matrix(matrix: *mut c_void) {
        unsafe {
            sys::gmprational::dd_FreeMatrix(matrix.cast());
        }
    }

    unsafe fn dd_matrix_append_to(dst: *mut *mut c_void, src: *mut c_void) -> i32 {
        unsafe { sys::gmprational::dd_MatrixAppendTo(dst.cast(), src.cast()) }
    }

    unsafe fn dd_append_matrix(a: *mut c_void, b: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_AppendMatrix(a.cast(), b.cast()).cast() }
    }

    unsafe fn dd_matrix_row_remove(matrix: *mut *mut c_void, row: c_long) -> i32 {
        unsafe { sys::gmprational::dd_MatrixRowRemove(matrix.cast(), row) }
    }

    unsafe fn dd_initialize_arow(cols: c_long, out: *mut *mut c_void) {
        unsafe {
            sys::gmprational::dd_InitializeArow(cols, out.cast());
        }
    }

    unsafe fn dd_free_arow(cols: c_long, arow: *mut c_void) {
        unsafe {
            sys::gmprational::dd_FreeArow(cols, arow.cast());
        }
    }

    unsafe fn dd_redundant(
        matrix: *mut c_void,
        row: c_long,
        cert: *mut c_void,
        err: *mut u32,
    ) -> i32 {
        unsafe { sys::gmprational::dd_Redundant(matrix.cast(), row, cert.cast(), err.cast()) }
    }

    unsafe fn dd_redundant_rows(matrix: *mut c_void, err: *mut u32) -> *mut libc::c_ulong {
        unsafe { sys::gmprational::dd_RedundantRows(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_matrix_canonicalize(
        matrix: *mut *mut c_void,
        impl_lin: *mut *mut libc::c_ulong,
        redset: *mut *mut libc::c_ulong,
        newpos: *mut *mut c_long,
        err: *mut u32,
    ) -> i32 {
        unsafe {
            sys::gmprational::dd_MatrixCanonicalize(
                matrix.cast(),
                impl_lin.cast(),
                redset.cast(),
                newpos.cast(),
                err.cast(),
            )
        }
    }

    unsafe fn set_groundsize(set: *mut libc::c_ulong) -> c_long {
        unsafe { sys::gmprational::set_groundsize(set.cast()) }
    }

    unsafe fn set_member(elem: c_long, set: *mut libc::c_ulong) -> i32 {
        unsafe { sys::gmprational::set_member(elem, set.cast()) }
    }

    unsafe fn set_free(set: *mut libc::c_ulong) {
        unsafe {
            sys::gmprational::set_free(set.cast());
        }
    }

    unsafe fn dd_matrix2poly(matrix: *mut c_void, err: *mut u32) -> *mut c_void {
        unsafe { sys::gmprational::dd_DDMatrix2Poly(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_free_polyhedra(poly: *mut c_void) {
        unsafe {
            sys::gmprational::dd_FreePolyhedra(poly.cast());
        }
    }

    unsafe fn dd_copy_inequalities(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyInequalities(poly.cast()).cast() }
    }

    unsafe fn dd_copy_generators(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyGenerators(poly.cast()).cast() }
    }

    unsafe fn dd_copy_adjacency(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyAdjacency(poly.cast()).cast() }
    }

    unsafe fn dd_copy_input_adjacency(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyInputAdjacency(poly.cast()).cast() }
    }

    unsafe fn dd_copy_incidence(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyIncidence(poly.cast()).cast() }
    }

    unsafe fn dd_copy_input_incidence(poly: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyInputIncidence(poly.cast()).cast() }
    }

    unsafe fn dd_append_matrix2poly(poly: *mut *mut c_void, rows: *mut c_void) -> i32 {
        unsafe { sys::gmprational::dd_AppendMatrix2Poly(poly.cast(), rows.cast()) }
    }

    unsafe fn dd_free_set_family(family: *mut c_void) {
        unsafe {
            sys::gmprational::dd_FreeSetFamily(family.cast());
        }
    }

    unsafe fn dd_matrix2lp(matrix: *mut c_void, err: *mut u32) -> *mut c_void {
        unsafe { sys::gmprational::dd_Matrix2LP(matrix.cast(), err.cast()).cast() }
    }

    unsafe fn dd_free_lp_data(lp: *mut c_void) {
        unsafe {
            sys::gmprational::dd_FreeLPData(lp.cast());
        }
    }

    unsafe fn dd_lp_solve_dual_simplex(lp: *mut c_void, err: *mut u32) -> i32 {
        unsafe {
            sys::gmprational::dd_LPSolve(
                lp.cast(),
                sys::gmprational::dd_LPSolverType_dd_DualSimplex,
                err.cast(),
            )
        }
    }

    unsafe fn lp_status_raw(lp: *mut c_void) -> u32 {
        unsafe { (*lp.cast::<sys::gmprational::dd_lpdata>()).LPS }
    }

    unsafe fn dd_copy_lp_solution(lp: *mut c_void) -> *mut c_void {
        unsafe { sys::gmprational::dd_CopyLPSolution(lp.cast()).cast() }
    }

    unsafe fn dd_free_lp_solution(sol: *mut c_void) {
        unsafe {
            sys::gmprational::dd_FreeLPSolution(sol.cast());
        }
    }

    unsafe fn lp_solution_optvalue_ptr(sol: *mut c_void) -> *const c_void {
        unsafe {
            let sol = sol.cast::<sys::gmprational::dd_lpsolution>();
            (&(*sol).optvalue as *const sys::gmprational::mytype).cast()
        }
    }

    unsafe fn write_mytype_real(target: *mut c_void, value: f64) {
        unsafe {
            sys::gmprational::__gmpq_set_d(target.cast(), value);
        }
    }

    unsafe fn write_mytype_int(target: *mut c_void, value: c_long) {
        unsafe {
            sys::gmprational::__gmpq_set_si(target.cast(), value, 1);
        }
    }

    unsafe fn read_mytype_real(source: *const c_void) -> f64 {
        unsafe { sys::gmprational::__gmpq_get_d(source.cast()) }
    }

    unsafe fn write_mytype(target: *mut c_void, value: &Self) {
        unsafe {
            sys::gmprational::__gmpq_set(target.cast(), value.inner.as_ptr());
        }
    }

    unsafe fn read_mytype(source: *const c_void) -> Self {
        unsafe {
            Self::ensure_initialized();
            let mut inner = MaybeUninit::<sys::gmprational::mytype>::uninit();
            sys::gmprational::__gmpq_init(inner.as_mut_ptr().cast());
            sys::gmprational::__gmpq_set(inner.as_mut_ptr().cast(), source.cast());
            CddRational {
                inner: inner.assume_init(),
            }
        }
    }
}

#[cfg(feature = "gmp")]
impl CddFloat {
    pub fn to_f64(&self) -> f64 {
        <Self as CddNumber>::ensure_initialized();
        unsafe { sys::gmpfloat::__gmpf_get_d(self.inner.as_ptr()) }
    }
}

#[cfg(feature = "gmp")]
impl Clone for CddFloat {
    fn clone(&self) -> Self {
        <Self as CddNumber>::ensure_initialized();
        unsafe {
            let mut inner = MaybeUninit::<sys::gmpfloat::mytype>::uninit();
            sys::gmpfloat::__gmpf_init(inner.as_mut_ptr().cast());
            sys::gmpfloat::__gmpf_set(inner.as_mut_ptr().cast(), self.inner.as_ptr());
            CddFloat {
                inner: inner.assume_init(),
            }
        }
    }
}

#[cfg(feature = "gmp")]
impl Drop for CddFloat {
    fn drop(&mut self) {
        <Self as CddNumber>::ensure_initialized();
        unsafe {
            sys::gmpfloat::__gmpf_clear(self.inner.as_mut_ptr());
        }
    }
}

#[cfg(feature = "gmp")]
impl From<f64> for CddFloat {
    fn from(value: f64) -> Self {
        <Self as CddNumber>::ensure_initialized();
        unsafe {
            let mut inner = MaybeUninit::<sys::gmpfloat::mytype>::uninit();
            sys::gmpfloat::__gmpf_init(inner.as_mut_ptr().cast());
            sys::gmpfloat::__gmpf_set_d(inner.as_mut_ptr().cast(), value);
            CddFloat {
                inner: inner.assume_init(),
            }
        }
    }
}

#[cfg(feature = "gmprational")]
impl CddRational {
    pub fn to_f64(&self) -> f64 {
        <Self as CddNumber>::ensure_initialized();
        unsafe { sys::gmprational::__gmpq_get_d(self.inner.as_ptr()) }
    }
}

#[cfg(feature = "gmprational")]
impl std::fmt::Display for CddRational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::ffi::CStr;

        <Self as CddNumber>::ensure_initialized();
        unsafe {
            let ptr = sys::gmprational::__gmpq_get_str(std::ptr::null_mut(), 10, self.inner.as_ptr());
            if ptr.is_null() {
                return Err(std::fmt::Error);
            }
            let s = CStr::from_ptr(ptr)
                .to_str()
                .map_err(|_| std::fmt::Error)?;
            let result = f.write_str(s);
            libc::free(ptr.cast());
            result
        }
    }
}

#[cfg(feature = "gmprational")]
impl Clone for CddRational {
    fn clone(&self) -> Self {
        <Self as CddNumber>::ensure_initialized();
        unsafe {
            let mut inner = MaybeUninit::<sys::gmprational::mytype>::uninit();
            sys::gmprational::__gmpq_init(inner.as_mut_ptr().cast());
            sys::gmprational::__gmpq_set(inner.as_mut_ptr().cast(), self.inner.as_ptr());
            CddRational {
                inner: inner.assume_init(),
            }
        }
    }
}

#[cfg(feature = "gmprational")]
impl Drop for CddRational {
    fn drop(&mut self) {
        <Self as CddNumber>::ensure_initialized();
        unsafe {
            sys::gmprational::__gmpq_clear(self.inner.as_mut_ptr());
        }
    }
}

#[cfg(feature = "gmprational")]
impl From<f64> for CddRational {
    fn from(value: f64) -> Self {
        <Self as CddNumber>::ensure_initialized();
        unsafe {
            let mut inner = MaybeUninit::<sys::gmprational::mytype>::uninit();
            sys::gmprational::__gmpq_init(inner.as_mut_ptr().cast());
            sys::gmprational::__gmpq_set_d(inner.as_mut_ptr().cast(), value);
            CddRational {
                inner: inner.assume_init(),
            }
        }
    }
}
