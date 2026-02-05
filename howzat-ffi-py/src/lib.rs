use std::sync::OnceLock;

use howzat_kit::{BackendGeometry, BackendRunAny, BackendRunConfig};
use howzat_kit::backend::{AnyPolytopeCoefficients, CoefficientMatrix};
use hullabaloo::AdjacencyList;
use hullabaloo::set_family::{ListFamily, SetFamily};
use numpy::IntoPyArray;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyInt};
use rug::integer::Order;

const DEFAULT_BACKEND_SPEC: &str = "howzat-dd[purify[snap]]:f64[eps[1e-12]]";
static DEFAULT_BACKEND: OnceLock<howzat_kit::Backend> = OnceLock::new();

const DEFAULT_EXACT_BACKEND_SPEC: &str = "howzat-dd:gmprat";
static DEFAULT_EXACT_BACKEND: OnceLock<howzat_kit::Backend> = OnceLock::new();

fn default_backend() -> &'static howzat_kit::Backend {
    DEFAULT_BACKEND.get_or_init(|| {
        DEFAULT_BACKEND_SPEC
            .parse()
            .expect("default backend spec must parse")
    })
}

fn default_exact_backend() -> &'static howzat_kit::Backend {
    DEFAULT_EXACT_BACKEND.get_or_init(|| {
        DEFAULT_EXACT_BACKEND_SPEC
            .parse()
            .expect("default exact backend spec must parse")
    })
}

fn check_index_lt(index: usize, len: usize, name: &str) -> PyResult<()> {
    if index < len {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{name} index out of range: {index} >= {len}"
    )))
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SolveRepresentation {
    EuclideanVertices,
    HomogeneousGenerators,
    Inequality,
}

#[pyclass(name = "Representation", module = "howzat")]
#[derive(Copy, Clone)]
pub struct PyRepresentation {
    repr: SolveRepresentation,
}

#[pymethods]
impl PyRepresentation {
    fn __repr__(&self) -> &'static str {
        match self.repr {
            SolveRepresentation::EuclideanVertices => "howzat.Representation.EuclideanVertices",
            SolveRepresentation::HomogeneousGenerators => {
                "howzat.Representation.HomogeneousGenerators"
            }
            SolveRepresentation::Inequality => "howzat.Representation.Inequality",
        }
    }
}

trait IntoPyGraph {
    fn into_py_graph(self, py: Python<'_>) -> PyResult<Py<PyAny>>;
}

trait IntoPyIncidence {
    fn into_facet_incidence(self) -> Vec<Vec<usize>>;
}

impl IntoPyIncidence for SetFamily {
    fn into_facet_incidence(self) -> Vec<Vec<usize>> {
        self.sets().iter().map(|set| set.to_indices()).collect()
    }
}

impl IntoPyIncidence for ListFamily {
    fn into_facet_incidence(self) -> Vec<Vec<usize>> {
        self.into_adjacency_lists()
    }
}

/// Dense undirected graph backed by a `SetFamily` of bitsets.
#[pyclass(name = "DenseGraph", module = "howzat")]
pub struct PyDenseGraph {
    inner: SetFamily,
}

#[pymethods]
impl PyDenseGraph {
    fn node_count(&self) -> usize {
        self.inner.family_size()
    }

    fn degree(&self, node: usize) -> PyResult<usize> {
        check_index_lt(node, self.inner.family_size(), "node")?;
        Ok(self.inner.sets()[node].cardinality())
    }

    fn contains(&self, node: usize, neighbor: usize) -> PyResult<bool> {
        let node_count = self.inner.family_size();
        check_index_lt(node, node_count, "node")?;
        check_index_lt(neighbor, node_count, "neighbor")?;
        Ok(self.inner.sets()[node].contains(neighbor))
    }

    fn neighbors(&self, node: usize) -> PyResult<Vec<usize>> {
        check_index_lt(node, self.inner.family_size(), "node")?;
        let row = &self.inner.sets()[node];
        let mut out = Vec::with_capacity(row.cardinality());
        out.extend(row.iter().raw());
        Ok(out)
    }

    fn __len__(&self) -> usize {
        self.inner.family_size()
    }

    fn __repr__(&self) -> String {
        format!("howzat.DenseGraph(nodes={})", self.inner.family_size())
    }
}

impl IntoPyGraph for SetFamily {
    fn into_py_graph(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, PyDenseGraph { inner: self })?.into_any())
    }
}

/// Sparse undirected graph backed by adjacency lists.
#[pyclass(name = "AdjacencyList", module = "howzat")]
pub struct PyAdjacencyList {
    inner: AdjacencyList,
}

#[pymethods]
impl PyAdjacencyList {
    fn node_count(&self) -> usize {
        self.inner.num_vertices()
    }

    fn degree(&self, node: usize) -> PyResult<usize> {
        check_index_lt(node, self.inner.num_vertices(), "node")?;
        Ok(self.inner.degree(node))
    }

    fn contains(&self, node: usize, neighbor: usize) -> PyResult<bool> {
        let node_count = self.inner.num_vertices();
        check_index_lt(node, node_count, "node")?;
        check_index_lt(neighbor, node_count, "neighbor")?;
        Ok(self.inner.contains(node, neighbor))
    }

    fn neighbors(&self, node: usize) -> PyResult<Vec<usize>> {
        check_index_lt(node, self.inner.num_vertices(), "node")?;
        Ok(self.inner.neighbors(node).to_vec())
    }

    fn __len__(&self) -> usize {
        self.inner.num_vertices()
    }

    fn __repr__(&self) -> String {
        format!("howzat.AdjacencyList(nodes={})", self.inner.num_vertices())
    }
}

impl IntoPyGraph for AdjacencyList {
    fn into_py_graph(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, PyAdjacencyList { inner: self })?.into_any())
    }
}

/// Result of a single backend solve.
///
/// Fields:
/// - `spec`: Backend spec actually used.
/// - `dimension`: Ambient dimension `d`.
/// - `vertices`: Number of vertices `n`.
/// - `facets`: Number of facets.
/// - `ridges`: Number of ridges (edges in the facet adjacency / FR graph).
/// - `total_seconds`: Time spent inside the backend (seconds).
/// - `vertex_positions`: Optional vertex coordinates if the backend returned baseline geometry.
/// - `vertex_adjacency`: Vertex adjacency graph (dense or sparse, depending on solve mode).
/// - `facets_to_vertices`: For each facet, the incident vertex indices.
/// - `facet_adjacency`: Facet adjacency graph (dense or sparse, depending on solve mode).
/// - `fails`: Backend-specific failure count (pipeline dependent).
/// - `fallbacks`: Backend-specific fallback count (pipeline dependent).
#[pyclass(name = "SolveResult", module = "howzat")]
pub struct SolveResult {
    #[pyo3(get)]
    spec: String,
    #[pyo3(get)]
    dimension: usize,
    #[pyo3(get)]
    vertices: usize,
    #[pyo3(get)]
    facets: usize,
    #[pyo3(get)]
    ridges: usize,
    #[pyo3(get)]
    total_seconds: f64,
    #[pyo3(get)]
    vertex_positions: Option<Vec<Vec<f64>>>,
    vertex_adjacency: Py<PyAny>,
    facets_to_vertices: Vec<Vec<usize>>,
    facet_adjacency: Py<PyAny>,
    #[pyo3(get)]
    generators: Py<PyAny>,
    #[pyo3(get)]
    inequalities: Py<PyAny>,
    #[pyo3(get)]
    fails: usize,
    #[pyo3(get)]
    fallbacks: usize,
}

#[pymethods]
impl SolveResult {
    #[getter]
    fn vertex_adjacency(&self, py: Python<'_>) -> Py<PyAny> {
        self.vertex_adjacency.clone_ref(py)
    }

    #[getter]
    fn facets_to_vertices(&self) -> &Vec<Vec<usize>> {
        &self.facets_to_vertices
    }

    #[getter]
    fn facet_adjacency(&self, py: Python<'_>) -> Py<PyAny> {
        self.facet_adjacency.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "howzat.SolveResult(spec={:?}, facets={}, ridges={})",
            self.spec, self.facets, self.ridges
        )
    }
}

/// Opaque backend parsed from a backend spec string.
///
/// Backend spec syntax matches `hirsch sandbox bench --backend` (without `^` / `%` prefixes).
/// The default is `howzat-dd[purify[snap]]:f64[eps[1e-12]]`.
#[pyclass(name = "Backend", module = "howzat")]
pub struct Backend {
    inner: howzat_kit::Backend,
}

fn coefficients_to_py(
    py: Python<'_>,
    coefficients: Option<AnyPolytopeCoefficients>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    fn matrix_to_py(py: Python<'_>, matrix: CoefficientMatrix) -> PyResult<Py<PyAny>> {
        match matrix {
            CoefficientMatrix::F64(m) => {
                let a = numpy::ndarray::Array2::from_shape_vec((m.rows, m.cols), m.data)
                    .map_err(|e| PyRuntimeError::new_err(format!("internal: coefficient reshape failed: {e}")))?;
                Ok(a.into_pyarray(py).to_owned().into_any().into())
            }
            CoefficientMatrix::RugRat(_) | CoefficientMatrix::DashuRat(_) => {
                let m = matrix.stringify().map_err(|_| {
                    PyRuntimeError::new_err("coefficient matrix could not be coerced to rug::Rational")
                })?;
                let mut out: Vec<Vec<String>> = Vec::with_capacity(m.rows);
                let mut iter = m.data.into_iter();
                for _ in 0..m.rows {
                    let mut row = Vec::with_capacity(m.cols);
                    row.extend(iter.by_ref().take(m.cols));
                    out.push(row);
                }
                Ok(out.into_pyobject(py)?.into_any().unbind())
            }
            _ => {
                let m = matrix.coerce::<f64>().map_err(|_| {
                    PyRuntimeError::new_err("coefficient matrix could not be coerced to float64")
                })?;
                let a = numpy::ndarray::Array2::from_shape_vec((m.rows, m.cols), m.data).map_err(|e| {
                    PyRuntimeError::new_err(format!("internal: coefficient reshape failed: {e}"))
                })?;
                Ok(a.into_pyarray(py).to_owned().into_any().into())
            }
        }
    }

    let Some(coefficients) = coefficients else {
        let none = py.None();
        return Ok((none.clone_ref(py), none));
    };

    Ok((
        matrix_to_py(py, coefficients.generators)?,
        matrix_to_py(py, coefficients.inequalities)?,
    ))
}

fn build_solve_result<Inc: IntoPyIncidence, Adj: IntoPyGraph>(
    py: Python<'_>,
    run: howzat_kit::BackendRun<Inc, Adj>,
) -> PyResult<SolveResult> {
    let howzat_kit::BackendRun {
        spec,
        stats,
        timing,
        coefficients,
        geometry,
        fails,
        fallbacks,
        error,
        ..
    } = run;

    if let Some(err) = error {
        return Err(PyRuntimeError::new_err(err));
    }

    fn vertex_positions_to_py(matrix: CoefficientMatrix) -> PyResult<Vec<Vec<f64>>> {
        let m = match matrix {
            CoefficientMatrix::F64(m) => m,
            other => other.coerce::<f64>().map_err(|_| {
                PyRuntimeError::new_err("vertex positions could not be coerced to float64")
            })?,
        };
        let mut out = Vec::with_capacity(m.rows);
        let mut iter = m.data.into_iter();
        for _ in 0..m.rows {
            let mut row = Vec::with_capacity(m.cols);
            row.extend(iter.by_ref().take(m.cols));
            out.push(row);
        }
        Ok(out)
    }

    let (vertex_positions, vertex_adjacency, facets_to_vertices, facet_adjacency) = match geometry {
        BackendGeometry::Baseline(b) => (
            Some(vertex_positions_to_py(b.vertex_positions)?),
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

    let facets_to_vertices = facets_to_vertices.into_facet_incidence();
    let (generators, inequalities) = coefficients_to_py(py, coefficients)?;

    Ok(SolveResult {
        spec: spec.to_string(),
        dimension: stats.dimension,
        vertices: stats.vertices,
        facets: stats.facets,
        ridges: stats.ridges,
        total_seconds: timing.total.as_secs_f64(),
        vertex_positions,
        vertex_adjacency: vertex_adjacency.into_py_graph(py)?,
        facets_to_vertices,
        facet_adjacency: facet_adjacency.into_py_graph(py)?,
        generators,
        inequalities,
        fails,
        fallbacks,
    })
}

fn build_solve_result_any(py: Python<'_>, run: BackendRunAny) -> PyResult<SolveResult> {
    match run {
        BackendRunAny::Dense(run) => build_solve_result(py, run),
        BackendRunAny::Sparse(run) => build_solve_result(py, run),
    }
}

trait SolveInput: numpy::Element + Copy {
    const EMPTY_ERROR_GEN: &'static str;
    const EMPTY_ERROR_INEQ: &'static str;
    const CONTIG_ERROR: &'static str;

    fn solve_row_major(
        backend: &howzat_kit::Backend,
        repr: howzat_kit::Representation,
        data: &[Self],
        rows: usize,
        cols: usize,
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, String>;
}

impl SolveInput for f64 {
    const EMPTY_ERROR_GEN: &'static str = "input must be a non-empty 2D float64 array";
    const EMPTY_ERROR_INEQ: &'static str = "input must be a non-empty 2D float64 array";
    const CONTIG_ERROR: &'static str =
        "input must be a contiguous (C-order) 2D float64 numpy array";

    fn solve_row_major(
        backend: &howzat_kit::Backend,
        repr: howzat_kit::Representation,
        data: &[Self],
        rows: usize,
        cols: usize,
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, String> {
        backend
            .solve_row_major(repr, data, rows, cols, config)
            .map_err(|err| err.to_string())
    }
}

impl SolveInput for i64 {
    const EMPTY_ERROR_GEN: &'static str = "input must be a non-empty 2D int64 array";
    const EMPTY_ERROR_INEQ: &'static str = "input must be a non-empty 2D int64 array";
    const CONTIG_ERROR: &'static str =
        "input must be a contiguous (C-order) 2D int64 numpy array";

    fn solve_row_major(
        backend: &howzat_kit::Backend,
        repr: howzat_kit::Representation,
        data: &[Self],
        rows: usize,
        cols: usize,
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, String> {
        backend
            .solve_row_major_exact(repr, data, rows, cols, config)
            .map_err(|err| err.to_string())
    }
}

fn solve_backend<T: SolveInput>(
    py: Python<'_>,
    backend: &howzat_kit::Backend,
    input: PyReadonlyArray2<'_, T>,
    repr: Option<PyRef<'_, PyRepresentation>>,
) -> PyResult<SolveResult> {
    let repr = repr.map_or(SolveRepresentation::EuclideanVertices, |r| r.repr);

    let input = input.as_array();
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err(match repr {
            SolveRepresentation::EuclideanVertices | SolveRepresentation::HomogeneousGenerators => {
                T::EMPTY_ERROR_GEN
            }
            SolveRepresentation::Inequality => T::EMPTY_ERROR_INEQ,
        }));
    }

    let slice = input
        .as_slice()
        .ok_or_else(|| PyValueError::new_err(T::CONTIG_ERROR))?;

    let config = BackendRunConfig {
        output_coefficients: true,
        ..BackendRunConfig::default()
    };

    let run = py
        .detach(|| match repr {
            SolveRepresentation::EuclideanVertices => {
                T::solve_row_major(
                    backend,
                    howzat_kit::Representation::EuclideanVertices,
                    slice,
                    rows,
                    cols,
                    &config,
                )
            }
            SolveRepresentation::HomogeneousGenerators => {
                if cols < 2 {
                    return Err("generator matrix must have at least 2 columns".to_string());
                }
                T::solve_row_major(
                    backend,
                    howzat_kit::Representation::HomogeneousGenerators,
                    slice,
                    rows,
                    cols,
                    &config,
                )
            }
            SolveRepresentation::Inequality => {
                if cols < 2 {
                    return Err("inequality matrix must have at least 2 columns".to_string());
                }
                T::solve_row_major(
                    backend,
                    howzat_kit::Representation::Inequality,
                    slice,
                    rows,
                    cols,
                    &config,
                )
            }
        })
        .map_err(PyRuntimeError::new_err)?;
    build_solve_result_any(py, run)
}

fn py_any_to_int_u32_vec(long: &Bound<'_, PyInt>) -> PyResult<Vec<u32>> {
    let py = long.py();

    let n_bits = unsafe { pyo3::ffi::_PyLong_NumBits(long.as_ptr()) };
    if n_bits == (-1isize as usize) {
        return Err(pyo3::PyErr::fetch(py));
    }
    if n_bits == 0 {
        return Ok(Vec::new());
    }

    let n_digits = (n_bits + 32) / 32;
    let mut buffer: Vec<u32> = Vec::with_capacity(n_digits);
    let status = unsafe {
        pyo3::ffi::_PyLong_AsByteArray(
            long.as_ptr().cast(),
            buffer.as_mut_ptr().cast::<u8>(),
            n_digits * 4,
            1,
            1,
        )
    };
    if status == -1 {
        return Err(pyo3::PyErr::fetch(py));
    }
    unsafe { buffer.set_len(n_digits) };
    buffer.iter_mut().for_each(|chunk| *chunk = u32::from_le(*chunk));
    Ok(buffer)
}

fn py_int_to_rug_integer(long: &Bound<'_, PyInt>) -> PyResult<rug::Integer> {
    let mut buffer = py_any_to_int_u32_vec(long)?;
    if buffer.last().copied().is_some_and(|last| last >> 31 != 0) {
        let mut elements = buffer.iter_mut();
        for element in elements.by_ref() {
            *element = (!*element).wrapping_add(1);
            if *element != 0 {
                break;
            }
        }
        for element in elements {
            *element = !*element;
        }

        let mut out = rug::Integer::from_digits(&buffer, Order::Lsf);
        out = -out;
        Ok(out)
    } else {
        Ok(rug::Integer::from_digits(&buffer, Order::Lsf))
    }
}

fn py_any_to_rug_integer(ob: &Bound<'_, PyAny>) -> PyResult<rug::Integer> {
    let py = ob.py();
    if let Ok(long) = ob.cast::<PyInt>() {
        return py_int_to_rug_integer(long);
    }

    let owned: Bound<'_, PyInt> = unsafe {
        Py::from_owned_ptr_or_err(py, pyo3::ffi::PyNumber_Index(ob.as_ptr()))?
            .into_bound(py)
    };
    py_int_to_rug_integer(&owned)
}

fn getattr_call0_if_needed<'py>(ob: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let attr = ob.getattr(name)?;
    if attr.is_callable() {
        attr.call0()
    } else {
        Ok(attr)
    }
}

fn py_any_to_rug_rat(ob: &Bound<'_, PyAny>) -> PyResult<calculo::num::RugRat> {
    let numer = getattr_call0_if_needed(ob, "numerator")?;
    let denom = getattr_call0_if_needed(ob, "denominator")?;
    let mut numer = py_any_to_rug_integer(&numer)?;
    let mut denom = py_any_to_rug_integer(&denom)?;
    if denom == 0 {
        return Err(PyValueError::new_err("invalid rational: denominator is zero"));
    }
    if denom < 0 {
        denom = -denom;
        numer = -numer;
    }
    Ok(calculo::num::RugRat(rug::Rational::from((numer, denom))))
}

fn solve_backend_exact_gmprat(
    py: Python<'_>,
    backend: &howzat_kit::Backend,
    input: PyReadonlyArray2<'_, Py<PyAny>>,
    repr: Option<PyRef<'_, PyRepresentation>>,
) -> PyResult<SolveResult> {
    let repr = repr.map_or(SolveRepresentation::EuclideanVertices, |r| r.repr);

    let input = input.as_array();
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err(match repr {
            SolveRepresentation::EuclideanVertices | SolveRepresentation::HomogeneousGenerators => {
                "input must be a non-empty 2D object array"
            }
            SolveRepresentation::Inequality => "input must be a non-empty 2D object array",
        }));
    }

    let slice = input
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("input must be a contiguous (C-order) 2D object numpy array"))?;

    let mut values: Vec<calculo::num::RugRat> = Vec::with_capacity(slice.len());
    for obj in slice {
        values.push(py_any_to_rug_rat(obj.bind(py))?);
    }

    let config = BackendRunConfig {
        output_coefficients: true,
        ..BackendRunConfig::default()
    };

    let repr = match repr {
        SolveRepresentation::EuclideanVertices => howzat_kit::Representation::EuclideanVertices,
        SolveRepresentation::HomogeneousGenerators => howzat_kit::Representation::HomogeneousGenerators,
        SolveRepresentation::Inequality => howzat_kit::Representation::Inequality,
    };

    let run = py
        .detach(|| backend.solve_row_major_exact_gmprat(repr, values, rows, cols, &config))
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    build_solve_result_any(py, run)
}

/// Solve with the default backend (`howzat-dd[purify[snap]]:f64[eps[1e-12]]`).
///
/// `vertices` must be a contiguous (C-order) `float64` NumPy array with shape `(n, d)`.
#[pyfunction]
#[pyo3(signature = (input, repr=None))]
fn solve(
    py: Python<'_>,
    input: PyReadonlyArray2<'_, f64>,
    repr: Option<PyRef<'_, PyRepresentation>>,
) -> PyResult<SolveResult> {
    solve_backend(py, default_backend(), input, repr)
}

/// Solve exactly with the default exact backend (`howzat-dd:gmprat`).
///
/// `input` must be a contiguous (C-order) NumPy array with shape `(n, d)` (int64) or
/// an object array holding exact rationals.
#[pyfunction]
#[pyo3(signature = (input, repr=None))]
fn solve_exact(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    repr: Option<PyRef<'_, PyRepresentation>>,
) -> PyResult<SolveResult> {
    if let Ok(input) = input.extract::<PyReadonlyArray2<i64>>() {
        return solve_backend(py, default_exact_backend(), input, repr);
    }
    let input: PyReadonlyArray2<Py<PyAny>> = input
        .extract()
        .map_err(|_| PyValueError::new_err("input must be a 2D int64 array or a 2D object array"))?;
    solve_backend_exact_gmprat(py, default_exact_backend(), input, repr)
}

#[pymethods]
impl Backend {
    #[new]
    #[pyo3(signature = (spec=None))]
    /// Create a backend from a backend spec string.
    ///
    /// If `spec` is `None`, uses the cached default backend (`howzat-dd[purify[snap]]:f64[eps[1e-12]]`).
    fn new(spec: Option<&str>) -> PyResult<Self> {
        let inner = match spec {
            Some(spec) => spec
                .parse()
                .map_err(|err: String| PyValueError::new_err(err))?,
            None => default_backend().clone(),
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("howzat.Backend({:?})", self.inner.to_string())
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    /// Solve with this backend.
    #[pyo3(signature = (input, repr=None))]
    fn solve(
        &self,
        py: Python<'_>,
        input: PyReadonlyArray2<'_, f64>,
        repr: Option<PyRef<'_, PyRepresentation>>,
    ) -> PyResult<SolveResult> {
        solve_backend(py, &self.inner, input, repr)
    }

    /// Solve exactly with this backend.
    #[pyo3(signature = (input, repr=None))]
    fn solve_exact(
        &self,
        py: Python<'_>,
        input: &Bound<'_, PyAny>,
        repr: Option<PyRef<'_, PyRepresentation>>,
    ) -> PyResult<SolveResult> {
        if let Ok(input) = input.extract::<PyReadonlyArray2<i64>>() {
            return solve_backend(py, &self.inner, input, repr);
        }
        let input: PyReadonlyArray2<Py<PyAny>> = input
            .extract()
            .map_err(|_| PyValueError::new_err("input must be a 2D int64 array or a 2D object array"))?;
        solve_backend_exact_gmprat(py, &self.inner, input, repr)
    }
}

/// High-performance polytope backend runner bindings (PyO3).
///
/// Entry points:
/// - `howzat.solve(vertices)` runs with a cached default backend (`howzat-dd[purify[snap]]:f64[eps[1e-12]]`).
/// - `howzat.solve_exact(vertices)` runs with a cached default exact backend (`howzat-dd:gmprat`).
/// - `howzat.Backend(spec).solve(vertices)` / `.solve_exact(vertices)` run with an explicit backend.
#[pymodule]
fn howzat(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = default_backend();
    let _ = default_exact_backend();
    m.add_class::<Backend>()?;
    m.add_class::<SolveResult>()?;
    m.add_class::<PyAdjacencyList>()?;
    m.add_class::<PyDenseGraph>()?;
    m.add_class::<PyRepresentation>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_exact, m)?)?;

    let rep_cls = m.getattr("Representation")?;
    rep_cls.setattr(
        "EuclideanVertices",
        Py::new(
            _py,
            PyRepresentation {
                repr: SolveRepresentation::EuclideanVertices,
            },
        )?,
    )?;
    rep_cls.setattr(
        "Inequality",
        Py::new(
            _py,
            PyRepresentation {
                repr: SolveRepresentation::Inequality,
            },
        )?,
    )?;
    rep_cls.setattr(
        "HomogeneousGenerators",
        Py::new(
            _py,
            PyRepresentation {
                repr: SolveRepresentation::HomogeneousGenerators,
            },
        )?,
    )?;
    Ok(())
}
