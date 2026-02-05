#![allow(non_camel_case_types)]

use std::ffi::{CStr, CString, c_char};
use std::panic::AssertUnwindSafe;
use std::ptr;
use std::sync::OnceLock;

use howzat_kit::{Backend, BackendGeometry, BackendRunAny, BackendRunConfig, Representation, Stats};
use howzat_kit::backend::{AnyPolytopeCoefficients, CoefficientMatrix, RowMajorMatrix};
use hullabaloo::AdjacencyList;
use hullabaloo::set_family::{ListFamily, SetFamily};

const DEFAULT_BACKEND_SPEC: &str = "howzat-dd[purify[snap]]:f64[eps[1e-12]]";
static DEFAULT_BACKEND: OnceLock<Backend> = OnceLock::new();

const DEFAULT_EXACT_BACKEND_SPEC: &str = "howzat-dd:gmprat";
static DEFAULT_EXACT_BACKEND: OnceLock<Backend> = OnceLock::new();

fn default_backend() -> &'static Backend {
    DEFAULT_BACKEND.get_or_init(|| {
        DEFAULT_BACKEND_SPEC
            .parse()
            .expect("default backend spec must parse")
    })
}

fn default_exact_backend() -> &'static Backend {
    DEFAULT_EXACT_BACKEND.get_or_init(|| {
        DEFAULT_EXACT_BACKEND_SPEC
            .parse()
            .expect("default exact backend spec must parse")
    })
}

pub struct howzat_backend_t;

pub struct howzat_result_t;

pub struct howzat_dense_graph_t;

pub struct howzat_adjacency_list_t;

#[repr(C)]
pub struct howzat_error_t {
    message: *mut c_char,
}

#[repr(C)]
pub struct howzat_usize_slice_t {
    pub ptr: *const usize,
    pub len: usize,
}

#[repr(C)]
pub struct howzat_f64_slice_t {
    pub ptr: *const f64,
    pub len: usize,
}

#[repr(C)]
pub struct howzat_i64_slice_t {
    pub ptr: *const i64,
    pub len: usize,
}

#[repr(C)]
pub struct howzat_str_slice_t {
    pub ptr: *const *const c_char,
    pub len: usize,
}

#[allow(non_camel_case_types)]
/// cbindgen:no-export
pub struct __mpq_struct;

#[allow(non_camel_case_types)]
/// cbindgen:no-export
pub type mpq_srcptr = *const __mpq_struct;

#[repr(C)]
pub struct howzat_mpq_slice_t {
    pub ptr: *const mpq_srcptr,
    pub len: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum howzat_graph_kind_t {
    HOWZAT_GRAPH_DENSE = 0,
    HOWZAT_GRAPH_SPARSE = 1,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum howzat_representation_t {
    HOWZAT_REPR_EUCLIDEAN_VERTICES = 0,
    HOWZAT_REPR_INEQUALITY = 1,
    HOWZAT_REPR_HOMOGENEOUS_GENERATORS = 2,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum howzat_coeff_kind_t {
    HOWZAT_COEFF_F64 = 0,
    HOWZAT_COEFF_GMPRAT = 1,
}

struct BackendHandle {
    inner: Backend,
    spec: CString,
}

struct DenseGraphHandle {
    inner: SetFamily,
}

struct AdjacencyListHandle {
    inner: AdjacencyList,
}

enum GraphHandle {
    Dense(DenseGraphHandle),
    Sparse(AdjacencyListHandle),
}

struct ResultHandle {
    spec: CString,
    stats: Stats,
    total_seconds: f64,
    fails: usize,
    fallbacks: usize,
    vertex_positions: Option<Vec<f64>>,
    vertex_positions_count: usize,
    vertex_positions_dim: usize,
    vertex_adjacency: GraphHandle,
    facet_adjacency: GraphHandle,
    facets_to_vertices_offsets: Vec<usize>,
    facets_to_vertices_data: Vec<usize>,
    coefficients: CoefficientsHandle,
}

struct CoeffMatrixF64Handle {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

struct CoeffMatrixI64Handle {
    data: Vec<i64>,
}

struct CoeffMatrixStrHandle {
    #[allow(dead_code)]
    data: Vec<CString>,
    ptrs: Vec<*const c_char>,
}

struct CoeffMatrixRugRatHandle {
    rows: usize,
    cols: usize,
    data: Vec<calculo::num::RugRat>,
    ptrs: Vec<mpq_srcptr>,
    strings: OnceLock<CoeffMatrixStrHandle>,
    i64s: OnceLock<Option<CoeffMatrixI64Handle>>,
}

enum CoefficientsHandle {
    F64 {
        generators: CoeffMatrixF64Handle,
        inequalities: CoeffMatrixF64Handle,
    },
    RugRat {
        generators: CoeffMatrixRugRatHandle,
        inequalities: CoeffMatrixRugRatHandle,
    },
}

fn cstring_lossy(s: &str) -> CString {
    if !s.as_bytes().contains(&b'\0') {
        return CString::new(s).expect("string must not contain NUL");
    }
    let sanitized: String = s.chars().map(|c| if c == '\0' { '�' } else { c }).collect();
    CString::new(sanitized).expect("sanitized string must not contain NUL")
}

fn set_error(out: *mut *mut howzat_error_t, msg: &str) {
    if out.is_null() {
        return;
    }
    let message = cstring_lossy(msg).into_raw();
    let err = Box::new(howzat_error_t { message });
    unsafe {
        *out = Box::into_raw(err);
    }
}

fn clear_error(out: *mut *mut howzat_error_t) {
    if out.is_null() {
        return;
    }
    unsafe {
        let prev = *out;
        *out = ptr::null_mut();
        if !prev.is_null() {
            let prev = Box::from_raw(prev);
            if !prev.message.is_null() {
                drop(CString::from_raw(prev.message));
            }
        }
    }
}

unsafe fn ref_backend<'a>(backend: *const howzat_backend_t) -> Option<&'a BackendHandle> {
    if backend.is_null() {
        return None;
    }
    Some(unsafe { &*(backend.cast::<BackendHandle>()) })
}

unsafe fn ref_result<'a>(result: *const howzat_result_t) -> Option<&'a ResultHandle> {
    if result.is_null() {
        return None;
    }
    Some(unsafe { &*(result.cast::<ResultHandle>()) })
}

unsafe fn ref_dense_graph<'a>(graph: *const howzat_dense_graph_t) -> Option<&'a DenseGraphHandle> {
    if graph.is_null() {
        return None;
    }
    Some(unsafe { &*(graph.cast::<DenseGraphHandle>()) })
}

unsafe fn ref_sparse_graph<'a>(
    graph: *const howzat_adjacency_list_t,
) -> Option<&'a AdjacencyListHandle> {
    if graph.is_null() {
        return None;
    }
    Some(unsafe { &*(graph.cast::<AdjacencyListHandle>()) })
}

fn flatten_set_family(family: &SetFamily) -> (Vec<usize>, Vec<usize>) {
    let sets = family.sets();
    let mut offsets = Vec::with_capacity(sets.len() + 1);
    offsets.push(0);

    let mut total = 0usize;
    for set in sets {
        total = total.saturating_add(set.cardinality());
        offsets.push(total);
    }

    let mut data = Vec::with_capacity(total);
    for set in sets {
        data.extend(set.iter().raw());
    }
    (offsets, data)
}

fn flatten_list_family(family: &ListFamily) -> (Vec<usize>, Vec<usize>) {
    let sets = family.sets();
    let mut offsets = Vec::with_capacity(sets.len() + 1);
    offsets.push(0);

    let mut total = 0usize;
    for set in sets {
        total = total.saturating_add(set.len());
        offsets.push(total);
    }

    let mut data = Vec::with_capacity(total);
    for set in sets {
        data.extend_from_slice(set);
    }
    (offsets, data)
}

fn build_coefficients_handle(
    coefficients: Option<AnyPolytopeCoefficients>,
) -> Result<CoefficientsHandle, String> {
    let coefficients = coefficients.ok_or_else(|| "backend did not return coefficients".to_string())?;

    fn build_rugrat_matrix(m: RowMajorMatrix<calculo::num::RugRat>) -> CoeffMatrixRugRatHandle {
        let RowMajorMatrix { rows, cols, data } = m;
        let ptrs = data
            .iter()
            .map(|v| v.0.as_raw().cast::<__mpq_struct>())
            .collect();
        CoeffMatrixRugRatHandle {
            rows,
            cols,
            data,
            ptrs,
            strings: OnceLock::new(),
            i64s: OnceLock::new(),
        }
    }

    fn is_rational(matrix: &CoefficientMatrix) -> bool {
        matches!(
            matrix,
            CoefficientMatrix::RugRat(_) | CoefficientMatrix::DashuRat(_)
        )
    }

    match (coefficients.generators, coefficients.inequalities) {
        (g, h) if !is_rational(&g) && !is_rational(&h) => {
            let g = g
                .coerce::<f64>()
                .map_err(|_| "coefficient matrix could not be coerced to f64".to_string())?;
            let h = h
                .coerce::<f64>()
                .map_err(|_| "coefficient matrix could not be coerced to f64".to_string())?;
            Ok(CoefficientsHandle::F64 {
                generators: CoeffMatrixF64Handle {
                    rows: g.rows,
                    cols: g.cols,
                    data: g.data,
                },
                inequalities: CoeffMatrixF64Handle {
                    rows: h.rows,
                    cols: h.cols,
                    data: h.data,
                },
            })
        }
        (g, h) if is_rational(&g) && is_rational(&h) => {
            let g = g.coerce::<calculo::num::RugRat>().map_err(|_| {
                "coefficient matrix could not be coerced to rug::Rational".to_string()
            })?;
            let h = h.coerce::<calculo::num::RugRat>().map_err(|_| {
                "coefficient matrix could not be coerced to rug::Rational".to_string()
            })?;
            Ok(CoefficientsHandle::RugRat {
                generators: build_rugrat_matrix(g),
                inequalities: build_rugrat_matrix(h),
            })
        }
        _ => Err("mixed coefficient kinds are not supported".to_string()),
    }
}

fn build_result_any(run: BackendRunAny) -> Result<ResultHandle, String> {
    match run {
        BackendRunAny::Dense(run) => {
            let howzat_kit::BackendRun {
                spec,
                stats,
                timing,
                geometry,
                coefficients,
                fails,
                fallbacks,
                error,
                ..
            } = run;

            if let Some(err) = error {
                return Err(err);
            }

            let vertex_positions_count = stats.vertices;
            let vertex_positions_dim = stats.dimension;

            let (vertex_positions, vertex_adjacency, facets_to_vertices, facet_adjacency) =
                match geometry {
                    BackendGeometry::Baseline(b) => (
                        Some(b.vertex_positions),
                        b.vertex_adjacency,
                        b.facets_to_vertices,
                        b.facet_adjacency,
                    ),
                    BackendGeometry::Input(g) => (
                        None,
                        g.vertex_adjacency,
                        g.facets_to_vertices,
                        g.facet_adjacency,
                    ),
                };

            let (facets_to_vertices_offsets, facets_to_vertices_data) =
                flatten_set_family(&facets_to_vertices);

            let coefficients = build_coefficients_handle(coefficients)?;

            let vertex_positions = match vertex_positions {
                None => None,
                Some(m) => Some(
                    m.coerce::<f64>()
                        .map_err(|_| "vertex positions could not be coerced to f64".to_string())?
                        .data,
                ),
            };

            Ok(ResultHandle {
                spec: cstring_lossy(&spec.to_string()),
                stats,
                total_seconds: timing.total.as_secs_f64(),
                fails,
                fallbacks,
                vertex_positions,
                vertex_positions_count,
                vertex_positions_dim,
                vertex_adjacency: GraphHandle::Dense(DenseGraphHandle {
                    inner: vertex_adjacency,
                }),
                facet_adjacency: GraphHandle::Dense(DenseGraphHandle {
                    inner: facet_adjacency,
                }),
                facets_to_vertices_offsets,
                facets_to_vertices_data,
                coefficients,
            })
        }
        BackendRunAny::Sparse(run) => {
            let howzat_kit::BackendRun {
                spec,
                stats,
                timing,
                geometry,
                coefficients,
                fails,
                fallbacks,
                error,
                ..
            } = run;

            if let Some(err) = error {
                return Err(err);
            }

            let vertex_positions_count = stats.vertices;
            let vertex_positions_dim = stats.dimension;

            let (vertex_positions, vertex_adjacency, facets_to_vertices, facet_adjacency) =
                match geometry {
                    BackendGeometry::Baseline(b) => (
                        Some(b.vertex_positions),
                        b.vertex_adjacency,
                        b.facets_to_vertices,
                        b.facet_adjacency,
                    ),
                    BackendGeometry::Input(g) => (
                        None,
                        g.vertex_adjacency,
                        g.facets_to_vertices,
                        g.facet_adjacency,
                    ),
                };

            let (facets_to_vertices_offsets, facets_to_vertices_data) =
                flatten_list_family(&facets_to_vertices);

            let coefficients = build_coefficients_handle(coefficients)?;

            let vertex_positions = match vertex_positions {
                None => None,
                Some(m) => Some(
                    m.coerce::<f64>()
                        .map_err(|_| "vertex positions could not be coerced to f64".to_string())?
                        .data,
                ),
            };

            Ok(ResultHandle {
                spec: cstring_lossy(&spec.to_string()),
                stats,
                total_seconds: timing.total.as_secs_f64(),
                fails,
                fallbacks,
                vertex_positions,
                vertex_positions_count,
                vertex_positions_dim,
                vertex_adjacency: GraphHandle::Sparse(AdjacencyListHandle {
                    inner: vertex_adjacency,
                }),
                facet_adjacency: GraphHandle::Sparse(AdjacencyListHandle {
                    inner: facet_adjacency,
                }),
                facets_to_vertices_offsets,
                facets_to_vertices_data,
                coefficients,
            })
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_error_message(err: *const howzat_error_t) -> *const c_char {
    if err.is_null() {
        return ptr::null();
    }
    unsafe { (*err).message }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_error_free(err: *mut howzat_error_t) {
    if err.is_null() {
        return;
    }
    unsafe {
        let err = Box::from_raw(err);
        if !err.message.is_null() {
            drop(CString::from_raw(err.message));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_backend_new(
    spec: *const c_char,
    out_err: *mut *mut howzat_error_t,
) -> *mut howzat_backend_t {
    clear_error(out_err);

    let make = || -> Result<BackendHandle, String> {
        let backend = if spec.is_null() {
            default_backend().clone()
        } else {
            let raw = unsafe { CStr::from_ptr(spec) };
            let raw = raw
                .to_str()
                .map_err(|_| "backend spec must be valid UTF-8".to_string())?;
            Backend::parse(raw)?
        };
        Ok(BackendHandle {
            spec: cstring_lossy(&backend.to_string()),
            inner: backend,
        })
    };

    let handle = match std::panic::catch_unwind(AssertUnwindSafe(make)) {
        Ok(Ok(handle)) => handle,
        Ok(Err(msg)) => {
            set_error(out_err, &msg);
            return ptr::null_mut();
        }
        Err(_) => {
            set_error(out_err, "panic while constructing backend");
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(handle)).cast::<howzat_backend_t>()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_backend_free(backend: *mut howzat_backend_t) {
    if backend.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(backend.cast::<BackendHandle>()));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_backend_spec(backend: *const howzat_backend_t) -> *const c_char {
    let Some(backend) = (unsafe { ref_backend(backend) }) else {
        return ptr::null();
    };
    backend.spec.as_ptr()
}

fn solve_row_major_f64(
    backend: &Backend,
    data: *const f64,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
) -> Result<BackendRunAny, String> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| "rows * cols overflow".to_string())?;
    if data.is_null() {
        return Err("data must not be NULL".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("rows and cols must be non-zero".to_string());
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let config = BackendRunConfig {
        output_coefficients: true,
        ..BackendRunConfig::default()
    };

    let repr = match repr {
        howzat_representation_t::HOWZAT_REPR_EUCLIDEAN_VERTICES => Representation::EuclideanVertices,
        howzat_representation_t::HOWZAT_REPR_INEQUALITY => Representation::Inequality,
        howzat_representation_t::HOWZAT_REPR_HOMOGENEOUS_GENERATORS => {
            Representation::HomogeneousGenerators
        }
    };
    if repr == Representation::Inequality && cols < 2 {
        return Err("inequality matrix must have at least 2 columns".to_string());
    }
    if repr == Representation::HomogeneousGenerators && cols < 2 {
        return Err("generator matrix must have at least 2 columns".to_string());
    }
    backend
        .solve_row_major(repr, slice, rows, cols, &config)
        .map_err(|e| e.to_string())
}

fn solve_row_major_i64(
    backend: &Backend,
    data: *const i64,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
) -> Result<BackendRunAny, String> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| "rows * cols overflow".to_string())?;
    if data.is_null() {
        return Err("data must not be NULL".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("rows and cols must be non-zero".to_string());
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let config = BackendRunConfig {
        output_coefficients: true,
        ..BackendRunConfig::default()
    };

    let repr = match repr {
        howzat_representation_t::HOWZAT_REPR_EUCLIDEAN_VERTICES => Representation::EuclideanVertices,
        howzat_representation_t::HOWZAT_REPR_INEQUALITY => Representation::Inequality,
        howzat_representation_t::HOWZAT_REPR_HOMOGENEOUS_GENERATORS => {
            Representation::HomogeneousGenerators
        }
    };
    if repr == Representation::Inequality && cols < 2 {
        return Err("inequality matrix must have at least 2 columns".to_string());
    }
    if repr == Representation::HomogeneousGenerators && cols < 2 {
        return Err("generator matrix must have at least 2 columns".to_string());
    }
    backend
        .solve_row_major_exact(repr, slice, rows, cols, &config)
        .map_err(|e| e.to_string())
}

fn solve_row_major_gmprat(
    backend: &Backend,
    data: *const mpq_srcptr,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
) -> Result<BackendRunAny, String> {
    use gmp_mpfr_sys::gmp::mpq_t;
    use rug::rational::BorrowRational;

    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| "rows * cols overflow".to_string())?;
    if data.is_null() {
        return Err("data must not be NULL".to_string());
    }
    if rows == 0 || cols == 0 {
        return Err("rows and cols must be non-zero".to_string());
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let mut values: Vec<calculo::num::RugRat> = Vec::with_capacity(len);
    for &ptr in slice {
        if ptr.is_null() {
            return Err("gmprat data entries must not be NULL".to_string());
        }
        let raw = unsafe { *(ptr.cast::<mpq_t>()) };
        let borrow = unsafe { BorrowRational::from_raw(raw) };
        values.push(calculo::num::RugRat((&*borrow).clone()));
    }

    let config = BackendRunConfig {
        output_coefficients: true,
        ..BackendRunConfig::default()
    };

    let repr = match repr {
        howzat_representation_t::HOWZAT_REPR_EUCLIDEAN_VERTICES => Representation::EuclideanVertices,
        howzat_representation_t::HOWZAT_REPR_INEQUALITY => Representation::Inequality,
        howzat_representation_t::HOWZAT_REPR_HOMOGENEOUS_GENERATORS => {
            Representation::HomogeneousGenerators
        }
    };
    if repr == Representation::Inequality && cols < 2 {
        return Err("inequality matrix must have at least 2 columns".to_string());
    }
    if repr == Representation::HomogeneousGenerators && cols < 2 {
        return Err("generator matrix must have at least 2 columns".to_string());
    }
    backend
        .solve_row_major_exact_gmprat(repr, values, rows, cols, &config)
        .map_err(|e| e.to_string())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_solve(
    data: *const f64,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
    out_err: *mut *mut howzat_error_t,
) -> *mut howzat_result_t {
    clear_error(out_err);

    let make = || -> Result<ResultHandle, String> {
        let run = solve_row_major_f64(default_backend(), data, rows, cols, repr)?;
        build_result_any(run)
    };

    let result = match std::panic::catch_unwind(AssertUnwindSafe(make)) {
        Ok(Ok(result)) => result,
        Ok(Err(msg)) => {
            set_error(out_err, &msg);
            return ptr::null_mut();
        }
        Err(_) => {
            set_error(out_err, "panic while solving");
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(result)).cast::<howzat_result_t>()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_solve_exact(
    data: *const i64,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
    out_err: *mut *mut howzat_error_t,
) -> *mut howzat_result_t {
    clear_error(out_err);

    let make = || -> Result<ResultHandle, String> {
        let run = solve_row_major_i64(default_exact_backend(), data, rows, cols, repr)?;
        build_result_any(run)
    };

    let result = match std::panic::catch_unwind(AssertUnwindSafe(make)) {
        Ok(Ok(result)) => result,
        Ok(Err(msg)) => {
            set_error(out_err, &msg);
            return ptr::null_mut();
        }
        Err(_) => {
            set_error(out_err, "panic while solving");
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(result)).cast::<howzat_result_t>()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_solve_exact_gmprat(
    data: *const mpq_srcptr,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
    out_err: *mut *mut howzat_error_t,
) -> *mut howzat_result_t {
    clear_error(out_err);

    let make = || -> Result<ResultHandle, String> {
        let run = solve_row_major_gmprat(default_exact_backend(), data, rows, cols, repr)?;
        build_result_any(run)
    };

    let result = match std::panic::catch_unwind(AssertUnwindSafe(make)) {
        Ok(Ok(result)) => result,
        Ok(Err(msg)) => {
            set_error(out_err, &msg);
            return ptr::null_mut();
        }
        Err(_) => {
            set_error(out_err, "panic while solving");
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(result)).cast::<howzat_result_t>()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_backend_solve(
    backend: *const howzat_backend_t,
    data: *const f64,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
    out_err: *mut *mut howzat_error_t,
) -> *mut howzat_result_t {
    clear_error(out_err);

    let Some(backend_ref) = unsafe { ref_backend(backend) }.map(|b| &b.inner) else {
        set_error(out_err, "backend must not be NULL");
        return ptr::null_mut();
    };

    let make = || -> Result<ResultHandle, String> {
        let run = solve_row_major_f64(backend_ref, data, rows, cols, repr)?;
        build_result_any(run)
    };

    let result = match std::panic::catch_unwind(AssertUnwindSafe(make)) {
        Ok(Ok(result)) => result,
        Ok(Err(msg)) => {
            set_error(out_err, &msg);
            return ptr::null_mut();
        }
        Err(_) => {
            set_error(out_err, "panic while solving");
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(result)).cast::<howzat_result_t>()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_backend_solve_exact(
    backend: *const howzat_backend_t,
    data: *const i64,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
    out_err: *mut *mut howzat_error_t,
) -> *mut howzat_result_t {
    clear_error(out_err);

    let Some(backend_ref) = unsafe { ref_backend(backend) }.map(|b| &b.inner) else {
        set_error(out_err, "backend must not be NULL");
        return ptr::null_mut();
    };

    let make = || -> Result<ResultHandle, String> {
        let run = solve_row_major_i64(backend_ref, data, rows, cols, repr)?;
        build_result_any(run)
    };

    let result = match std::panic::catch_unwind(AssertUnwindSafe(make)) {
        Ok(Ok(result)) => result,
        Ok(Err(msg)) => {
            set_error(out_err, &msg);
            return ptr::null_mut();
        }
        Err(_) => {
            set_error(out_err, "panic while solving");
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(result)).cast::<howzat_result_t>()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_backend_solve_exact_gmprat(
    backend: *const howzat_backend_t,
    data: *const mpq_srcptr,
    rows: usize,
    cols: usize,
    repr: howzat_representation_t,
    out_err: *mut *mut howzat_error_t,
) -> *mut howzat_result_t {
    clear_error(out_err);

    let Some(backend_ref) = unsafe { ref_backend(backend) }.map(|b| &b.inner) else {
        set_error(out_err, "backend must not be NULL");
        return ptr::null_mut();
    };

    let make = || -> Result<ResultHandle, String> {
        let run = solve_row_major_gmprat(backend_ref, data, rows, cols, repr)?;
        build_result_any(run)
    };

    let result = match std::panic::catch_unwind(AssertUnwindSafe(make)) {
        Ok(Ok(result)) => result,
        Ok(Err(msg)) => {
            set_error(out_err, &msg);
            return ptr::null_mut();
        }
        Err(_) => {
            set_error(out_err, "panic while solving");
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(result)).cast::<howzat_result_t>()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_free(result: *mut howzat_result_t) {
    if result.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(result.cast::<ResultHandle>()));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_spec(result: *const howzat_result_t) -> *const c_char {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return ptr::null();
    };
    result.spec.as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_dimension(result: *const howzat_result_t) -> usize {
    unsafe { ref_result(result) }
        .map(|r| r.stats.dimension)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_vertices(result: *const howzat_result_t) -> usize {
    unsafe { ref_result(result) }
        .map(|r| r.stats.vertices)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_facets(result: *const howzat_result_t) -> usize {
    unsafe { ref_result(result) }
        .map(|r| r.stats.facets)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_ridges(result: *const howzat_result_t) -> usize {
    unsafe { ref_result(result) }
        .map(|r| r.stats.ridges)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_total_seconds(result: *const howzat_result_t) -> f64 {
    unsafe { ref_result(result) }
        .map(|r| r.total_seconds)
        .unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_fails(result: *const howzat_result_t) -> usize {
    unsafe { ref_result(result) }.map(|r| r.fails).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_fallbacks(result: *const howzat_result_t) -> usize {
    unsafe { ref_result(result) }
        .map(|r| r.fallbacks)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_coeff_kind(
    result: *const howzat_result_t,
) -> howzat_coeff_kind_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_coeff_kind_t::HOWZAT_COEFF_F64;
    };
    match result.coefficients {
        CoefficientsHandle::F64 { .. } => howzat_coeff_kind_t::HOWZAT_COEFF_F64,
        CoefficientsHandle::RugRat { .. } => howzat_coeff_kind_t::HOWZAT_COEFF_GMPRAT,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_generators_shape(
    result: *const howzat_result_t,
    out_rows: *mut usize,
    out_cols: *mut usize,
) {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return;
    };
    let (rows, cols) = match &result.coefficients {
        CoefficientsHandle::F64 { generators, .. } => (generators.rows, generators.cols),
        CoefficientsHandle::RugRat { generators, .. } => (generators.rows, generators.cols),
    };
    if !out_rows.is_null() {
        unsafe { *out_rows = rows };
    }
    if !out_cols.is_null() {
        unsafe { *out_cols = cols };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_inequalities_shape(
    result: *const howzat_result_t,
    out_rows: *mut usize,
    out_cols: *mut usize,
) {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return;
    };
    let (rows, cols) = match &result.coefficients {
        CoefficientsHandle::F64 { inequalities, .. } => (inequalities.rows, inequalities.cols),
        CoefficientsHandle::RugRat { inequalities, .. } => (inequalities.rows, inequalities.cols),
    };
    if !out_rows.is_null() {
        unsafe { *out_rows = rows };
    }
    if !out_cols.is_null() {
        unsafe { *out_cols = cols };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_generators_f64(
    result: *const howzat_result_t,
) -> howzat_f64_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_f64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::F64 { generators, .. } = &result.coefficients else {
        return howzat_f64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_f64_slice_t {
        ptr: generators.data.as_ptr(),
        len: generators.data.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_inequalities_f64(
    result: *const howzat_result_t,
) -> howzat_f64_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_f64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::F64 { inequalities, .. } = &result.coefficients else {
        return howzat_f64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_f64_slice_t {
        ptr: inequalities.data.as_ptr(),
        len: inequalities.data.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_generators_i64(
    result: *const howzat_result_t,
) -> howzat_i64_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_i64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::RugRat { generators, .. } = &result.coefficients else {
        return howzat_i64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let i64_matrix = generators.i64s.get_or_init(|| {
        let mut data = Vec::with_capacity(generators.data.len());
        for value in &generators.data {
            if !value.0.is_integer() {
                return None;
            }
            let Some(value) = value.0.numer().to_i64() else {
                return None;
            };
            data.push(value);
        }
        Some(CoeffMatrixI64Handle {
            data,
        })
    });
    let Some(i64_matrix) = i64_matrix else {
        return howzat_i64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_i64_slice_t {
        ptr: i64_matrix.data.as_ptr(),
        len: i64_matrix.data.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_inequalities_i64(
    result: *const howzat_result_t,
) -> howzat_i64_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_i64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::RugRat { inequalities, .. } = &result.coefficients else {
        return howzat_i64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let i64_matrix = inequalities.i64s.get_or_init(|| {
        let mut data = Vec::with_capacity(inequalities.data.len());
        for value in &inequalities.data {
            if !value.0.is_integer() {
                return None;
            }
            let Some(value) = value.0.numer().to_i64() else {
                return None;
            };
            data.push(value);
        }
        Some(CoeffMatrixI64Handle {
            data,
        })
    });
    let Some(i64_matrix) = i64_matrix else {
        return howzat_i64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_i64_slice_t {
        ptr: i64_matrix.data.as_ptr(),
        len: i64_matrix.data.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_generators_gmprat(
    result: *const howzat_result_t,
) -> howzat_mpq_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_mpq_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::RugRat { generators, .. } = &result.coefficients else {
        return howzat_mpq_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_mpq_slice_t {
        ptr: generators.ptrs.as_ptr(),
        len: generators.ptrs.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_inequalities_gmprat(
    result: *const howzat_result_t,
) -> howzat_mpq_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_mpq_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::RugRat { inequalities, .. } = &result.coefficients else {
        return howzat_mpq_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_mpq_slice_t {
        ptr: inequalities.ptrs.as_ptr(),
        len: inequalities.ptrs.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_generators_str(
    result: *const howzat_result_t,
) -> howzat_str_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_str_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::RugRat { generators, .. } = &result.coefficients else {
        return howzat_str_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let str_matrix = generators.strings.get_or_init(|| {
        let data: Vec<CString> = generators
            .data
            .iter()
            .map(|v| cstring_lossy(&v.to_string()))
            .collect();
        let ptrs: Vec<*const c_char> = data.iter().map(|s| s.as_ptr()).collect();
        CoeffMatrixStrHandle {
            data,
            ptrs,
        }
    });
    howzat_str_slice_t {
        ptr: str_matrix.ptrs.as_ptr(),
        len: str_matrix.ptrs.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_inequalities_str(
    result: *const howzat_result_t,
) -> howzat_str_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_str_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let CoefficientsHandle::RugRat { inequalities, .. } = &result.coefficients else {
        return howzat_str_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let str_matrix = inequalities.strings.get_or_init(|| {
        let data: Vec<CString> = inequalities
            .data
            .iter()
            .map(|v| cstring_lossy(&v.to_string()))
            .collect();
        let ptrs: Vec<*const c_char> = data.iter().map(|s| s.as_ptr()).collect();
        CoeffMatrixStrHandle {
            data,
            ptrs,
        }
    });
    howzat_str_slice_t {
        ptr: str_matrix.ptrs.as_ptr(),
        len: str_matrix.ptrs.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_vertex_positions(
    result: *const howzat_result_t,
) -> howzat_f64_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_f64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    let Some(buf) = result.vertex_positions.as_ref() else {
        return howzat_f64_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_f64_slice_t {
        ptr: buf.as_ptr(),
        len: buf.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_vertex_positions_shape(
    result: *const howzat_result_t,
    out_vertex_count: *mut usize,
    out_dim: *mut usize,
) {
    if let Some(result) = unsafe { ref_result(result) } {
        if !out_vertex_count.is_null() {
            unsafe {
                *out_vertex_count = result.vertex_positions_count;
            }
        }
        if !out_dim.is_null() {
            unsafe {
                *out_dim = result.vertex_positions_dim;
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_facets_to_vertices_offsets(
    result: *const howzat_result_t,
) -> howzat_usize_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_usize_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_usize_slice_t {
        ptr: result.facets_to_vertices_offsets.as_ptr(),
        len: result.facets_to_vertices_offsets.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_facets_to_vertices_data(
    result: *const howzat_result_t,
) -> howzat_usize_slice_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_usize_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    howzat_usize_slice_t {
        ptr: result.facets_to_vertices_data.as_ptr(),
        len: result.facets_to_vertices_data.len(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_vertex_adjacency_kind(
    result: *const howzat_result_t,
) -> howzat_graph_kind_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_graph_kind_t::HOWZAT_GRAPH_SPARSE;
    };
    match &result.vertex_adjacency {
        GraphHandle::Dense(_) => howzat_graph_kind_t::HOWZAT_GRAPH_DENSE,
        GraphHandle::Sparse(_) => howzat_graph_kind_t::HOWZAT_GRAPH_SPARSE,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_facet_adjacency_kind(
    result: *const howzat_result_t,
) -> howzat_graph_kind_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return howzat_graph_kind_t::HOWZAT_GRAPH_SPARSE;
    };
    match &result.facet_adjacency {
        GraphHandle::Dense(_) => howzat_graph_kind_t::HOWZAT_GRAPH_DENSE,
        GraphHandle::Sparse(_) => howzat_graph_kind_t::HOWZAT_GRAPH_SPARSE,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_vertex_adjacency_dense(
    result: *const howzat_result_t,
) -> *const howzat_dense_graph_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return ptr::null();
    };
    match &result.vertex_adjacency {
        GraphHandle::Dense(g) => g as *const DenseGraphHandle as *const howzat_dense_graph_t,
        GraphHandle::Sparse(_) => ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_vertex_adjacency_sparse(
    result: *const howzat_result_t,
) -> *const howzat_adjacency_list_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return ptr::null();
    };
    match &result.vertex_adjacency {
        GraphHandle::Sparse(g) => g as *const AdjacencyListHandle as *const howzat_adjacency_list_t,
        GraphHandle::Dense(_) => ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_facet_adjacency_dense(
    result: *const howzat_result_t,
) -> *const howzat_dense_graph_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return ptr::null();
    };
    match &result.facet_adjacency {
        GraphHandle::Dense(g) => g as *const DenseGraphHandle as *const howzat_dense_graph_t,
        GraphHandle::Sparse(_) => ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_result_facet_adjacency_sparse(
    result: *const howzat_result_t,
) -> *const howzat_adjacency_list_t {
    let Some(result) = (unsafe { ref_result(result) }) else {
        return ptr::null();
    };
    match &result.facet_adjacency {
        GraphHandle::Sparse(g) => g as *const AdjacencyListHandle as *const howzat_adjacency_list_t,
        GraphHandle::Dense(_) => ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_dense_graph_node_count(
    graph: *const howzat_dense_graph_t,
) -> usize {
    unsafe { ref_dense_graph(graph) }
        .map(|g| g.inner.family_size())
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_dense_graph_degree(
    graph: *const howzat_dense_graph_t,
    node: usize,
) -> usize {
    let Some(graph) = (unsafe { ref_dense_graph(graph) }) else {
        return 0;
    };
    if node >= graph.inner.family_size() {
        return 0;
    }
    graph.inner.sets()[node].cardinality()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_dense_graph_contains(
    graph: *const howzat_dense_graph_t,
    node: usize,
    neighbor: usize,
) -> bool {
    let Some(graph) = (unsafe { ref_dense_graph(graph) }) else {
        return false;
    };
    let node_count = graph.inner.family_size();
    if node >= node_count || neighbor >= node_count {
        return false;
    }
    graph.inner.sets()[node].contains(neighbor)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_dense_graph_neighbors(
    graph: *const howzat_dense_graph_t,
    node: usize,
    out: *mut usize,
    out_cap: usize,
) -> usize {
    let Some(graph) = (unsafe { ref_dense_graph(graph) }) else {
        return 0;
    };
    let node_count = graph.inner.family_size();
    if node >= node_count {
        return 0;
    }

    let row = &graph.inner.sets()[node];
    let degree = row.cardinality();
    if out.is_null() || out_cap == 0 {
        return degree;
    }

    let mut written = 0usize;
    for idx in row.iter().raw() {
        if written >= out_cap {
            break;
        }
        unsafe {
            *out.add(written) = idx;
        }
        written += 1;
    }
    degree
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_adjacency_list_node_count(
    graph: *const howzat_adjacency_list_t,
) -> usize {
    unsafe { ref_sparse_graph(graph) }
        .map(|g| g.inner.num_vertices())
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_adjacency_list_degree(
    graph: *const howzat_adjacency_list_t,
    node: usize,
) -> usize {
    let Some(graph) = (unsafe { ref_sparse_graph(graph) }) else {
        return 0;
    };
    if node >= graph.inner.num_vertices() {
        return 0;
    }
    graph.inner.degree(node)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_adjacency_list_contains(
    graph: *const howzat_adjacency_list_t,
    node: usize,
    neighbor: usize,
) -> bool {
    let Some(graph) = (unsafe { ref_sparse_graph(graph) }) else {
        return false;
    };
    let node_count = graph.inner.num_vertices();
    if node >= node_count || neighbor >= node_count {
        return false;
    }
    graph.inner.contains(node, neighbor)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn howzat_adjacency_list_neighbors(
    graph: *const howzat_adjacency_list_t,
    node: usize,
) -> howzat_usize_slice_t {
    let Some(graph) = (unsafe { ref_sparse_graph(graph) }) else {
        return howzat_usize_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    };
    if node >= graph.inner.num_vertices() {
        return howzat_usize_slice_t {
            ptr: ptr::null(),
            len: 0,
        };
    }
    let neighbors = graph.inner.neighbors(node);
    howzat_usize_slice_t {
        ptr: neighbors.as_ptr(),
        len: neighbors.len(),
    }
}
