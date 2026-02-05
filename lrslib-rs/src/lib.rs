//! Safe, idiomatic Rust bindings on top of the raw `lrslib-sys` FFI.
//!
//! The primary goal (for now) is H↔V conversion:
//! - enumerate vertices/rays from an H-representation
//! - enumerate facets from a V-representation (convex hull)

#[cfg(all(not(feature = "gmp"), not(target_pointer_width = "64")))]
compile_error!(
    "lrslib-rs without `gmp` currently requires a 64-bit target (LRSLONG+B128). \
Enable the `gmp` feature for other targets."
);

mod raw;

use std::collections::BTreeSet;

use calculo::num::{f64_row_to_scaled_ints, f64_to_num_den};
use rug::Complete;
use rug::Integer;
use thiserror::Error;

use lrslib_sys as sys;
use raw::mp;

#[derive(Debug, Error)]
pub enum LrsError {
    #[error("lrslib returned a null pointer")]
    NullPointer,

    #[error("invalid matrix dimensions (rows={rows}, cols={cols})")]
    InvalidMatrix { rows: usize, cols: usize },

    #[error("matrix rows are ragged (inconsistent column counts)")]
    RaggedMatrix,

    #[error("non-finite value {value} at row={row} col={col}")]
    NonFinite { row: usize, col: usize, value: f64 },

    #[error(
        "generator matrix first column must be 0 (ray) or 1 (vertex), got {value} at row={row}"
    )]
    BadGeneratorType { row: usize, value: f64 },

    #[error("lrslib failed to find an initial basis (empty or invalid input)")]
    NoInitialBasis,

    #[error("failed to parse lrslib integer '{text}' as f64")]
    ParseF64 { text: String },

    #[error("failed to parse lrslib integer '{text}' as Integer")]
    ParseBigInt { text: String },

    #[error("operation unsupported: {0}")]
    Unsupported(&'static str),
}

pub type LrsResult<T> = Result<T, LrsError>;

use hullabaloo::set_family::ListFamily;
use hullabaloo::types::RepresentationKind;

#[derive(Debug, Clone)]
pub struct Matrix {
    repr: RepresentationKind,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, repr: RepresentationKind) -> LrsResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(LrsError::InvalidMatrix { rows, cols });
        }
        Ok(Self {
            repr,
            rows,
            cols,
            data: vec![0.0; rows * cols],
        })
    }

    pub fn from_rows(rows: &[Vec<f64>], repr: RepresentationKind) -> LrsResult<Self> {
        let Some(first) = rows.first() else {
            return Err(LrsError::InvalidMatrix { rows: 0, cols: 0 });
        };
        if first.is_empty() {
            return Err(LrsError::InvalidMatrix {
                rows: rows.len(),
                cols: 0,
            });
        }
        let cols = first.len();
        if rows.iter().skip(1).any(|r| r.len() != cols) {
            return Err(LrsError::RaggedMatrix);
        }

        let mut out = Self::new(rows.len(), cols, repr)?;
        for (r, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                out.set(r, c, v);
            }
        }
        Ok(out)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn representation(&self) -> RepresentationKind {
        self.repr
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.rows);
        assert!(col < self.cols);
        self.data[row * self.cols + col] = value;
    }

    pub fn to_rows_vec(&self) -> Vec<Vec<f64>> {
        (0..self.rows)
            .map(|r| {
                let start = r * self.cols;
                self.data[start..start + self.cols].to_vec()
            })
            .collect()
    }

    pub fn from_vertices<const D: usize>(vertices: &[[f64; D]]) -> LrsResult<Self> {
        if vertices.is_empty() {
            return Err(LrsError::InvalidMatrix {
                rows: 0,
                cols: D + 1,
            });
        }

        let mut m = Matrix::new(vertices.len(), D + 1, RepresentationKind::Generator)?;
        for (r, v) in vertices.iter().enumerate() {
            m.set(r, 0, 1.0);
            for (c, &x) in v.iter().enumerate() {
                m.set(r, c + 1, x);
            }
        }
        Ok(m)
    }

    /// Build a generator matrix from a dynamic list of vertices.
    ///
    /// The ambient dimension is determined from the first vertex. All vertices
    /// must have the same length.
    pub fn from_vertex_rows(vertices: &[Vec<f64>]) -> LrsResult<Self> {
        let Some(first) = vertices.first() else {
            return Err(LrsError::InvalidMatrix { rows: 0, cols: 0 });
        };
        let dim = first.len();
        if dim == 0 {
            return Err(LrsError::InvalidMatrix {
                rows: vertices.len(),
                cols: 0,
            });
        }
        if vertices.iter().skip(1).any(|v| v.len() != dim) {
            return Err(LrsError::RaggedMatrix);
        }

        let mut m = Matrix::new(vertices.len(), dim + 1, RepresentationKind::Generator)?;
        for (r, coords) in vertices.iter().enumerate() {
            m.set(r, 0, 1.0);
            for (c, &x) in coords.iter().enumerate() {
                m.set(r, c + 1, x);
            }
        }
        Ok(m)
    }
}

#[derive(Debug, Clone)]
pub struct Polyhedron {
    input: Matrix,
}

impl Polyhedron {
    pub fn from_matrix(m: Matrix) -> Self {
        Self { input: m }
    }

    pub fn from_vertices(vertices: &[Vec<f64>]) -> LrsResult<Self> {
        Ok(Self::from_matrix(Matrix::from_vertex_rows(vertices)?))
    }

    /// Return an H-representation (facets) for this polyhedron.
    pub fn facets(&self) -> LrsResult<Matrix> {
        match self.input.representation() {
            RepresentationKind::Inequality => Ok(self.input.clone()),
            RepresentationKind::Generator => facets_from_generators(&self.input),
        }
    }

    /// Return a V-representation (vertices/rays) for this polyhedron.
    pub fn generators(&self) -> LrsResult<Matrix> {
        match self.input.representation() {
            RepresentationKind::Generator => Ok(self.input.clone()),
            RepresentationKind::Inequality => generators_from_inequalities(&self.input),
        }
    }

    /// Output→input incidence.
    ///
    /// - If this polyhedron was constructed from a V-representation, this returns
    ///   (facet → vertices/rays).
    /// - If constructed from an H-representation, this returns
    ///   (vertex/ray → inequalities).
    pub fn incidence(&self) -> LrsResult<ListFamily> {
        match self.input.representation() {
            RepresentationKind::Generator => {
                let (_, facet_to_vertex) = facets_with_incidence_from_generators(&self.input)?;
                Ok(ListFamily::from_sorted_sets(
                    facet_to_vertex,
                    self.input.rows(),
                ))
            }
            RepresentationKind::Inequality => {
                let (_, gen_to_ineq) = generators_with_incidence_from_inequalities(&self.input)?;
                Ok(ListFamily::from_sorted_sets(gen_to_ineq, self.input.rows()))
            }
        }
    }

    /// Input→output incidence (the reverse direction of [`Self::incidence`]).
    pub fn input_incidence(&self) -> LrsResult<ListFamily> {
        match self.input.representation() {
            RepresentationKind::Generator => {
                let (_facets, facet_to_vertex) =
                    facets_with_incidence_from_generators(&self.input)?;
                let incidence = ListFamily::from_sorted_sets(facet_to_vertex, self.input.rows());
                Ok(incidence.transpose())
            }
            RepresentationKind::Inequality => {
                let (_gens, gen_to_ineq) =
                    generators_with_incidence_from_inequalities(&self.input)?;
                let incidence = ListFamily::from_sorted_sets(gen_to_ineq, self.input.rows());
                Ok(incidence.transpose())
            }
        }
    }

    /// Compute the "other" representation plus incidence in both directions.
    ///
    /// This is the preferred way to amortize expensive conversion work across multiple queries
    /// (e.g. incidence + downstream combinatorial queries), without internal caching: callers should keep the returned
    /// value if they need to re-use results.
    pub fn solve(&self) -> LrsResult<PolyhedronSolved> {
        match self.input.representation() {
            RepresentationKind::Generator => {
                let (facets, facet_to_vertex) = facets_with_incidence_from_generators(&self.input)?;
                let incidence = ListFamily::from_sorted_sets(facet_to_vertex, self.input.rows());
                let input_incidence = incidence.transpose();
                Ok(PolyhedronSolved {
                    input: self.input.clone(),
                    output: facets,
                    incidence,
                    input_incidence,
                })
            }
            RepresentationKind::Inequality => {
                let (generators, gen_to_ineq) =
                    generators_with_incidence_from_inequalities(&self.input)?;
                let incidence = ListFamily::from_sorted_sets(gen_to_ineq, self.input.rows());
                let input_incidence = incidence.transpose();
                Ok(PolyhedronSolved {
                    input: self.input.clone(),
                    output: generators,
                    incidence,
                    input_incidence,
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PolyhedronSolved {
    input: Matrix,
    output: Matrix,
    /// Output→input incidence (matching [`Polyhedron::incidence`]).
    incidence: ListFamily,
    /// Input→output incidence (matching [`Polyhedron::input_incidence`]).
    input_incidence: ListFamily,
}

impl PolyhedronSolved {
    pub fn input(&self) -> &Matrix {
        &self.input
    }

    pub fn output(&self) -> &Matrix {
        &self.output
    }

    pub fn incidence(&self) -> &ListFamily {
        &self.incidence
    }

    pub fn input_incidence(&self) -> &ListFamily {
        &self.input_incidence
    }

    pub fn into_parts(self) -> (Matrix, Matrix, ListFamily, ListFamily) {
        (
            self.input,
            self.output,
            self.incidence,
            self.input_incidence,
        )
    }
}

#[inline(always)]
fn set_matrix_rows(handle: &mut raw::Handle, matrix: &Matrix) -> LrsResult<()> {
    let cols = matrix.cols();
    for r in 0..matrix.rows() {
        let start = r * cols;
        let row = &matrix.data[start..start + cols];
        handle.set_row(r + 1, row)?;
    }
    Ok(())
}

struct ExtractedSolution<K, C> {
    key: K,
    incidence: Vec<usize>,
    ctx: C,
}

trait SolutionExtractor {
    type Key: Ord;
    type Ctx;
    type Row;

    fn extract_solution(
        &mut self,
        handle: &mut raw::Handle,
        col: i64,
    ) -> LrsResult<ExtractedSolution<Self::Key, Self::Ctx>>;

    fn build_row(&mut self, handle: &mut raw::Handle, ctx: Self::Ctx) -> LrsResult<Vec<Self::Row>>;
}

#[inline(always)]
fn enumerate_unique_solutions<E: SolutionExtractor>(
    handle: &mut raw::Handle,
    extractor: &mut E,
) -> LrsResult<(Vec<Vec<E::Row>>, Vec<Vec<usize>>)> {
    handle.get_first_basis()?;

    let mut rows = Vec::<Vec<E::Row>>::new();
    let mut incidence = Vec::<Vec<usize>>::new();
    let mut seen = BTreeSet::<E::Key>::new();
    loop {
        let d = handle.solution_cols();
        for col in 0..=d {
            if !handle.get_solution(col) {
                continue;
            }

            let extracted = extractor.extract_solution(handle, col)?;
            if !seen.insert(extracted.key) {
                continue;
            }

            rows.push(extractor.build_row(handle, extracted.ctx)?);
            incidence.push(extracted.incidence);
        }

        if !handle.next_basis() {
            break;
        }
    }

    Ok((rows, incidence))
}

fn facets_with_incidence_from_generators(
    generators: &Matrix,
) -> LrsResult<(Matrix, Vec<Vec<usize>>)> {
    if generators.representation() != RepresentationKind::Generator {
        return Err(LrsError::InvalidMatrix {
            rows: generators.rows(),
            cols: generators.cols(),
        });
    }

    // First column must be 0/1 (ray/vertex) in lrslib's expected V-rep format.
    for r in 0..generators.rows() {
        let t = generators.get(r, 0);
        if t != 0.0 && t != 1.0 {
            return Err(LrsError::BadGeneratorType { row: r, value: t });
        }
    }

    let polytope = is_vertex_only_generator_matrix(generators);
    let mut handle = raw::Handle::new(generators.rows(), generators.cols(), true, polytope)?;
    let n = handle.n();

    // Precompute exact rational input rows so we can derive facet→vertex incidence
    // by evaluating each output inequality on the original generators.
    //
    // The incidence data embedded in the lrs dictionary for hull computations refers
    // to the dual system and does not match the "facet contains vertex" relation.
    let gen_rows = precompute_generator_rows(generators);
    set_matrix_rows(&mut handle, generators)?;

    let mut extractor = HullFacetExtractor {
        n,
        generators: &gen_rows,
        coeffs: Vec::with_capacity(n),
    };
    let (rows, incidence) = enumerate_unique_solutions(&mut handle, &mut extractor)?;

    Ok((
        Matrix::from_rows(&rows, RepresentationKind::Inequality)?,
        incidence,
    ))
}

pub fn facets_with_incidence_from_integer_vertices(
    vertices: &[&[i64]],
) -> LrsResult<(Matrix, Vec<Vec<usize>>)> {
    let Some(first) = vertices.first() else {
        return Err(LrsError::InvalidMatrix { rows: 0, cols: 0 });
    };
    let dim = first.len();
    if dim == 0 {
        return Err(LrsError::InvalidMatrix {
            rows: vertices.len(),
            cols: 0,
        });
    }
    if vertices.iter().skip(1).any(|v| v.len() != dim) {
        return Err(LrsError::RaggedMatrix);
    }

    let mut handle = raw::Handle::new(vertices.len(), dim.saturating_add(1), true, true)?;
    let n = handle.n();
    for (row, coords) in vertices.iter().enumerate() {
        handle.set_vertex_row_i64(row + 1, coords)?;
    }

    let mut extractor = HullFacetExtractorI64 {
        n,
        vertices,
        coeffs: Vec::with_capacity(n),
    };
    let (rows, incidence) = enumerate_unique_solutions(&mut handle, &mut extractor)?;

    Ok((
        Matrix::from_rows(&rows, RepresentationKind::Inequality)?,
        incidence,
    ))
}

pub fn facet_incidence_from_integer_vertices(vertices: &[&[i64]]) -> LrsResult<Vec<Vec<usize>>> {
    let Some(first) = vertices.first() else {
        return Err(LrsError::InvalidMatrix { rows: 0, cols: 0 });
    };
    let dim = first.len();
    if dim == 0 {
        return Err(LrsError::InvalidMatrix {
            rows: vertices.len(),
            cols: 0,
        });
    }
    if vertices.iter().skip(1).any(|v| v.len() != dim) {
        return Err(LrsError::RaggedMatrix);
    }

    let mut handle = raw::Handle::new(vertices.len(), dim.saturating_add(1), true, true)?;
    let n = handle.n();
    for (row, coords) in vertices.iter().enumerate() {
        handle.set_vertex_row_i64(row + 1, coords)?;
    }

    let mut extractor = HullFacetIncidenceExtractorI64 {
        n,
        vertices,
        coeffs: Vec::with_capacity(n),
    };
    let (_rows, incidence) = enumerate_unique_solutions(&mut handle, &mut extractor)?;
    Ok(incidence)
}

pub fn facets_with_incidence_from_integer_vertices_exact(
    vertices: &[&[i64]],
) -> LrsResult<(Vec<Vec<Integer>>, Vec<Vec<usize>>)> {
    let Some(first) = vertices.first() else {
        return Err(LrsError::InvalidMatrix { rows: 0, cols: 0 });
    };
    let dim = first.len();
    if dim == 0 {
        return Err(LrsError::InvalidMatrix {
            rows: vertices.len(),
            cols: 0,
        });
    }
    if vertices.iter().skip(1).any(|v| v.len() != dim) {
        return Err(LrsError::RaggedMatrix);
    }

    let mut handle = raw::Handle::new(vertices.len(), dim.saturating_add(1), true, true)?;
    let n = handle.n();
    for (row, coords) in vertices.iter().enumerate() {
        handle.set_vertex_row_i64(row + 1, coords)?;
    }

    let mut extractor = HullFacetExtractorI64Exact {
        n,
        vertices,
        coeffs: Vec::with_capacity(n),
    };
    enumerate_unique_solutions(&mut handle, &mut extractor)
}

#[derive(Clone)]
struct GeneratorRowExact {
    t: Integer,
    denom: Integer,
    coords_scaled: Vec<Integer>,
}

fn precompute_generator_rows(generators: &Matrix) -> Vec<GeneratorRowExact> {
    let n = generators.cols();
    let dim = n.saturating_sub(1);
    let mut out = Vec::with_capacity(generators.rows());

    for r in 0..generators.rows() {
        let start = r * n;
        let t = Integer::from(generators.data[start] as i64);
        let coords = &generators.data[start + 1..start + n];
        debug_assert_eq!(coords.len(), dim, "generator dimension mismatch");
        let (denom, coords_scaled) = f64_row_to_scaled_ints::<Integer>(coords);

        out.push(GeneratorRowExact {
            t,
            denom,
            coords_scaled,
        })
    }

    out
}

fn facet_incidence_from_output_mp(
    output: sys::lrs_mp_vector,
    n: usize,
    generators: &[GeneratorRowExact],
    coeffs: &mut Vec<Integer>,
) -> LrsResult<Vec<usize>> {
    coeffs.clear();
    if coeffs.capacity() < n {
        coeffs.reserve_exact(n - coeffs.capacity());
    }
    for i in 0..n {
        let mp = mp::mp_ptr_from_vec(output, i);
        coeffs.push(mp::mp_int_to_integer(mp)?);
    }

    let mut out = Vec::new();
    for (idx, generator) in generators.iter().enumerate() {
        // Evaluate (b, a) · (t, x) exactly as an integer numerator over `generator.denom`.
        let mut num = (&coeffs[0] * &generator.t).complete();
        num *= &generator.denom;
        for (coeff, coord) in coeffs.iter().skip(1).zip(&generator.coords_scaled) {
            num += coeff * coord;
        }
        if num == 0 {
            out.push(idx);
        }
    }

    Ok(out)
}

fn facet_incidence_from_output_mp_i64(
    output: sys::lrs_mp_vector,
    n: usize,
    vertices: &[&[i64]],
    coeffs: &mut Vec<Integer>,
) -> LrsResult<Vec<usize>> {
    coeffs.clear();
    if coeffs.capacity() < n {
        coeffs.reserve_exact(n - coeffs.capacity());
    }
    for i in 0..n {
        let mp = mp::mp_ptr_from_vec(output, i);
        coeffs.push(mp::mp_int_to_integer(mp)?);
    }

    let mut out = Vec::new();
    for (idx, coords) in vertices.iter().enumerate() {
        let mut num = coeffs[0].clone();
        for (coeff, &coord) in coeffs.iter().skip(1).zip(coords.iter()) {
            if coord != 0 {
                num += coeff * coord;
            }
        }
        if num == 0 {
            out.push(idx);
        }
    }

    Ok(out)
}

struct HullFacetExtractor<'a> {
    n: usize,
    generators: &'a [GeneratorRowExact],
    coeffs: Vec<Integer>,
}

impl SolutionExtractor for HullFacetExtractor<'_> {
    type Key = Vec<usize>;
    type Ctx = ();
    type Row = f64;

    #[inline(always)]
    fn extract_solution(
        &mut self,
        handle: &mut raw::Handle,
        _col: i64,
    ) -> LrsResult<ExtractedSolution<Self::Key, Self::Ctx>> {
        let output = handle.output_vector();
        let incidence =
            facet_incidence_from_output_mp(output, self.n, self.generators, &mut self.coeffs)?;
        Ok(ExtractedSolution {
            key: incidence.clone(),
            incidence,
            ctx: (),
        })
    }

    #[inline(always)]
    fn build_row(&mut self, handle: &mut raw::Handle, _ctx: Self::Ctx) -> LrsResult<Vec<f64>> {
        let output = handle.output_vector();
        let mut out_row = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mp = mp::mp_ptr_from_vec(output, i);
            out_row.push(mp::mp_int_to_f64(mp)?);
        }
        Ok(out_row)
    }
}

struct HullFacetExtractorI64<'a> {
    n: usize,
    vertices: &'a [&'a [i64]],
    coeffs: Vec<Integer>,
}

impl SolutionExtractor for HullFacetExtractorI64<'_> {
    type Key = Vec<usize>;
    type Ctx = ();
    type Row = f64;

    #[inline(always)]
    fn extract_solution(
        &mut self,
        handle: &mut raw::Handle,
        _col: i64,
    ) -> LrsResult<ExtractedSolution<Self::Key, Self::Ctx>> {
        let output = handle.output_vector();
        let incidence = facet_incidence_from_output_mp_i64(
            output,
            self.n,
            self.vertices,
            &mut self.coeffs,
        )?;
        Ok(ExtractedSolution {
            key: incidence.clone(),
            incidence,
            ctx: (),
        })
    }

    #[inline(always)]
    fn build_row(&mut self, handle: &mut raw::Handle, _ctx: Self::Ctx) -> LrsResult<Vec<f64>> {
        let output = handle.output_vector();
        let mut out_row = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mp = mp::mp_ptr_from_vec(output, i);
            out_row.push(mp::mp_int_to_f64(mp)?);
        }
        Ok(out_row)
    }
}

struct HullFacetIncidenceExtractorI64<'a> {
    n: usize,
    vertices: &'a [&'a [i64]],
    coeffs: Vec<Integer>,
}

impl SolutionExtractor for HullFacetIncidenceExtractorI64<'_> {
    type Key = Vec<usize>;
    type Ctx = ();
    type Row = ();

    #[inline(always)]
    fn extract_solution(
        &mut self,
        handle: &mut raw::Handle,
        _col: i64,
    ) -> LrsResult<ExtractedSolution<Self::Key, Self::Ctx>> {
        let output = handle.output_vector();
        let incidence = facet_incidence_from_output_mp_i64(
            output,
            self.n,
            self.vertices,
            &mut self.coeffs,
        )?;
        Ok(ExtractedSolution {
            key: incidence.clone(),
            incidence,
            ctx: (),
        })
    }

    #[inline(always)]
    fn build_row(&mut self, _handle: &mut raw::Handle, _ctx: Self::Ctx) -> LrsResult<Vec<()>> {
        Ok(Vec::new())
    }
}

struct HullFacetExtractorI64Exact<'a> {
    n: usize,
    vertices: &'a [&'a [i64]],
    coeffs: Vec<Integer>,
}

impl SolutionExtractor for HullFacetExtractorI64Exact<'_> {
    type Key = Vec<usize>;
    type Ctx = ();
    type Row = Integer;

    #[inline(always)]
    fn extract_solution(
        &mut self,
        handle: &mut raw::Handle,
        _col: i64,
    ) -> LrsResult<ExtractedSolution<Self::Key, Self::Ctx>> {
        let output = handle.output_vector();
        let incidence = facet_incidence_from_output_mp_i64(
            output,
            self.n,
            self.vertices,
            &mut self.coeffs,
        )?;
        Ok(ExtractedSolution {
            key: incidence.clone(),
            incidence,
            ctx: (),
        })
    }

    #[inline(always)]
    fn build_row(&mut self, _handle: &mut raw::Handle, _ctx: Self::Ctx) -> LrsResult<Vec<Integer>> {
        Ok(self.coeffs.clone())
    }
}

struct HrepGeneratorExtractor {
    n: usize,
}

impl SolutionExtractor for HrepGeneratorExtractor {
    type Key = (bool, Vec<usize>);
    type Ctx = bool;
    type Row = f64;

    #[inline(always)]
    fn extract_solution(
        &mut self,
        handle: &mut raw::Handle,
        col: i64,
    ) -> LrsResult<ExtractedSolution<Self::Key, Self::Ctx>> {
        let output = handle.output_vector();
        let denom = mp::mp_ptr_from_vec(output, 0);
        let is_vertex = !mp::mp_is_zero(denom);
        let incidence = handle.solution_incidence(col);
        Ok(ExtractedSolution {
            key: (is_vertex, incidence.clone()),
            incidence,
            ctx: is_vertex,
        })
    }

    #[inline(always)]
    fn build_row(&mut self, handle: &mut raw::Handle, is_vertex: bool) -> LrsResult<Vec<f64>> {
        let output = handle.output_vector();
        let denom = mp::mp_ptr_from_vec(output, 0);

        let mut out_row = Vec::with_capacity(self.n);
        out_row.push(if is_vertex { 1.0 } else { 0.0 });

        for i in 1..self.n {
            let num = mp::mp_ptr_from_vec(output, i);
            if is_vertex {
                out_row.push(mp::mp_rat_to_f64(num, denom));
            } else {
                out_row.push(mp::mp_int_to_f64(num)?);
            }
        }

        Ok(out_row)
    }
}

fn generators_with_incidence_from_inequalities(
    ineq: &Matrix,
) -> LrsResult<(Matrix, Vec<Vec<usize>>)> {
    if ineq.representation() != RepresentationKind::Inequality {
        return Err(LrsError::InvalidMatrix {
            rows: ineq.rows(),
            cols: ineq.cols(),
        });
    }

    let mut handle = raw::Handle::new(ineq.rows(), ineq.cols(), false, false)?;
    let n = handle.n();

    set_matrix_rows(&mut handle, ineq)?;

    let mut extractor = HrepGeneratorExtractor { n };
    let (rows, incidence) = enumerate_unique_solutions(&mut handle, &mut extractor)?;

    Ok((
        Matrix::from_rows(&rows, RepresentationKind::Generator)?,
        incidence,
    ))
}

fn facets_from_generators(generators: &Matrix) -> LrsResult<Matrix> {
    facets_with_incidence_from_generators(generators).map(|(m, _)| m)
}

fn generators_from_inequalities(ineq: &Matrix) -> LrsResult<Matrix> {
    generators_with_incidence_from_inequalities(ineq).map(|(m, _)| m)
}

fn is_vertex_only_generator_matrix(m: &Matrix) -> bool {
    m.representation() == RepresentationKind::Generator
        && (0..m.rows()).all(|r| m.get(r, 0) == 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Mutex;

    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().expect("test serial lock poisoned")
    }

    fn assert_symmetric_no_self_loops(adjacency: &[Vec<usize>]) {
        for (i, neigh) in adjacency.iter().enumerate() {
            for &j in neigh {
                assert_ne!(i, j, "self-loop at {i}");
                assert!(
                    adjacency
                        .get(j)
                        .is_some_and(|back| back.binary_search(&i).is_ok()),
                    "missing symmetric edge {j} -> {i}"
                );
            }
        }
    }

    #[test]
    fn square_facets_count() {
        let _serial = serial();

        let verts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let poly = Polyhedron::from_vertices(&verts).expect("poly");
        let facets = poly.facets().expect("facets");
        assert_eq!(facets.representation(), RepresentationKind::Inequality);
        assert_eq!(facets.cols(), 3);
        assert_eq!(facets.rows(), 4);
    }

    #[test]
    fn square_vertices_roundtrip() {
        let _serial = serial();

        // Unit square: x>=0, y>=0, x<=1, y<=1.
        // In lrslib H-rep format: b + a x >= 0.
        let rows = vec![
            vec![0.0, 1.0, 0.0],  // x >= 0
            vec![0.0, 0.0, 1.0],  // y >= 0
            vec![1.0, -1.0, 0.0], // 1 - x >= 0
            vec![1.0, 0.0, -1.0], // 1 - y >= 0
        ];
        let h = Matrix::from_rows(&rows, RepresentationKind::Inequality).expect("h");
        let poly = Polyhedron::from_matrix(h);
        let gens = poly.generators().expect("gens");
        assert_eq!(gens.representation(), RepresentationKind::Generator);
        assert_eq!(gens.cols(), 3);
        assert_eq!(gens.rows(), 4);

        // All should be vertices (bounded polytope).
        for r in 0..gens.rows() {
            assert_eq!(gens.get(r, 0), 1.0);
        }
    }

    #[test]
    fn square_incidence_and_adjacency_from_vertices() {
        let _serial = serial();

        let verts = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let poly = Polyhedron::from_vertices(&verts).expect("poly");

        // V-input: incidence is facet -> vertex.
        let inc = poly.incidence().expect("incidence");
        assert_eq!(inc.len(), 4);
        assert_eq!(inc.universe_size(), 4);
        for f in 0..inc.len() {
            let s = inc.set(f).expect("set");
            assert_eq!(s.len(), 2, "facet {f} incidence {s:?}");
        }

        let input_inc = poly.input_incidence().expect("input_incidence");
        assert_eq!(input_inc.len(), 4);
        assert_eq!(input_inc.universe_size(), 4);
        for v in 0..input_inc.len() {
            let s = input_inc.set(v).expect("set");
            assert_eq!(s.len(), 2);
        }
        for f in 0..inc.len() {
            for &v in inc.set(f).unwrap() {
                assert!(input_inc.set(v).unwrap().binary_search(&f).is_ok());
            }
        }

        let adj_dim = verts.first().map_or(0, |v| v.len()) + 1;

        // V-input: facet adjacency is computed from facet->vertex incidence;
        // vertex adjacency is computed from vertex->facet incidence.
        let facet_graph = hullabaloo::adjacency::adjacency_from_incidence(
            inc.sets(),
            inc.universe_size(),
            adj_dim,
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        assert_eq!(facet_graph.adjacency_lists().len(), 4);
        assert_symmetric_no_self_loops(facet_graph.adjacency_lists());
        for neigh in facet_graph.adjacency_lists() {
            assert_eq!(neigh.len(), 2);
        }

        let vertex_graph = hullabaloo::adjacency::adjacency_from_incidence(
            input_inc.sets(),
            input_inc.universe_size(),
            adj_dim,
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        assert_eq!(vertex_graph.adjacency_lists().len(), 4);
        assert_symmetric_no_self_loops(vertex_graph.adjacency_lists());
        for neigh in vertex_graph.adjacency_lists() {
            assert_eq!(neigh.len(), 2);
        }
    }

    #[test]
    fn square_incidence_and_adjacency_from_inequalities() {
        let _serial = serial();

        // Unit square: x>=0, y>=0, x<=1, y<=1.
        let rows = vec![
            vec![0.0, 1.0, 0.0],  // x >= 0
            vec![0.0, 0.0, 1.0],  // y >= 0
            vec![1.0, -1.0, 0.0], // 1 - x >= 0
            vec![1.0, 0.0, -1.0], // 1 - y >= 0
        ];
        let h = Matrix::from_rows(&rows, RepresentationKind::Inequality).expect("h");
        let poly = Polyhedron::from_matrix(h);

        // H-input: incidence is (vertex -> inequality).
        let inc = poly.incidence().expect("incidence");
        assert_eq!(inc.len(), 4);
        assert_eq!(inc.universe_size(), 4);
        for v in 0..inc.len() {
            let s = inc.set(v).expect("set");
            assert_eq!(s.len(), 2);
        }

        let input_inc = poly.input_incidence().expect("input_incidence");
        assert_eq!(input_inc.len(), 4);
        assert_eq!(input_inc.universe_size(), 4);
        for f in 0..input_inc.len() {
            let s = input_inc.set(f).expect("set");
            assert_eq!(s.len(), 2);
        }
        for v in 0..inc.len() {
            for &f in inc.set(v).unwrap() {
                assert!(input_inc.set(f).unwrap().binary_search(&v).is_ok());
            }
        }

        let adj_dim = rows.first().map_or(0, |r| r.len());

        // H-input: vertex adjacency is computed from vertex->facet incidence;
        // facet adjacency is computed from facet->vertex incidence.
        let vertex_graph = hullabaloo::adjacency::adjacency_from_incidence(
            inc.sets(),
            inc.universe_size(),
            adj_dim,
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        assert_eq!(vertex_graph.adjacency_lists().len(), 4);
        assert_symmetric_no_self_loops(vertex_graph.adjacency_lists());
        for neigh in vertex_graph.adjacency_lists() {
            assert_eq!(neigh.len(), 2);
        }

        let facet_graph = hullabaloo::adjacency::adjacency_from_incidence(
            input_inc.sets(),
            input_inc.universe_size(),
            adj_dim,
            hullabaloo::adjacency::IncidenceAdjacencyOptions::default(),
        );
        assert_eq!(facet_graph.adjacency_lists().len(), 4);
        assert_symmetric_no_self_loops(facet_graph.adjacency_lists());
        for neigh in facet_graph.adjacency_lists() {
            assert_eq!(neigh.len(), 2);
        }
    }
}
