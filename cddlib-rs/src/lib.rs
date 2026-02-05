//! Safe, idiomatic Rust bindings on top of the raw `cddlib-sys` FFI.
//!
//! This crate owns the unsafe glue for working with cddlib matrices, polyhedra,
//! and LP solutions. Callers should be able to use these types without touching
//! raw pointers or the `cddlib-sys` bindings directly.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::rc::Rc;

use hullabaloo::set_family::SetFamily as HbSetFamily;
use hullabaloo::types::{
    Representation as RepresentationMarker, RepresentationKind, RowSet as HbRowSet,
};
use thiserror::Error;
pub use hullabaloo::types::{Generator, Inequality};

#[cfg(all(
    not(feature = "gmprational"),
    not(feature = "gmp"),
    not(feature = "f64")
))]
compile_error!("cddlib-rs requires at least one backend feature: f64, gmp, or gmprational");

mod backend;
mod raw;

pub use backend::{CddNumber, DefaultNumber};

#[cfg(feature = "gmp")]
pub use backend::CddFloat;

#[cfg(feature = "gmprational")]
pub use backend::CddRational;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CddErrorCode {
    DimensionTooLarge,
    ImproperInputFormat,
    NegativeMatrixSize,
    EmptyVRepresentation,
    EmptyHRepresentation,
    EmptyRepresentation,
    InputFileNotFound,
    OutputFileNotOpen,
    NoLpObjective,
    NoRealNumberSupport,
    NotAvailForH,
    NotAvailForV,
    CannotHandleLinearity,
    RowIndexOutOfRange,
    ColIndexOutOfRange,
    LpCycling,
    NumericallyInconsistent,
    NoError,
}

impl CddErrorCode {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::DimensionTooLarge,
            1 => Self::ImproperInputFormat,
            2 => Self::NegativeMatrixSize,
            3 => Self::EmptyVRepresentation,
            4 => Self::EmptyHRepresentation,
            5 => Self::EmptyRepresentation,
            6 => Self::InputFileNotFound,
            7 => Self::OutputFileNotOpen,
            8 => Self::NoLpObjective,
            9 => Self::NoRealNumberSupport,
            10 => Self::NotAvailForH,
            11 => Self::NotAvailForV,
            12 => Self::CannotHandleLinearity,
            13 => Self::RowIndexOutOfRange,
            14 => Self::ColIndexOutOfRange,
            15 => Self::LpCycling,
            16 => Self::NumericallyInconsistent,
            17 => Self::NoError,
            other => panic!("unknown dd_ErrorType value {other}"),
        }
    }

    pub fn as_raw(self) -> u32 {
        match self {
            Self::DimensionTooLarge => 0,
            Self::ImproperInputFormat => 1,
            Self::NegativeMatrixSize => 2,
            Self::EmptyVRepresentation => 3,
            Self::EmptyHRepresentation => 4,
            Self::EmptyRepresentation => 5,
            Self::InputFileNotFound => 6,
            Self::OutputFileNotOpen => 7,
            Self::NoLpObjective => 8,
            Self::NoRealNumberSupport => 9,
            Self::NotAvailForH => 10,
            Self::NotAvailForV => 11,
            Self::CannotHandleLinearity => 12,
            Self::RowIndexOutOfRange => 13,
            Self::ColIndexOutOfRange => 14,
            Self::LpCycling => 15,
            Self::NumericallyInconsistent => 16,
            Self::NoError => 17,
        }
    }

    fn cddlib_name(self) -> &'static str {
        match self {
            Self::DimensionTooLarge => "dd_DimensionTooLarge",
            Self::ImproperInputFormat => "dd_ImproperInputFormat",
            Self::NegativeMatrixSize => "dd_NegativeMatrixSize",
            Self::EmptyVRepresentation => "dd_EmptyVrepresentation",
            Self::EmptyHRepresentation => "dd_EmptyHrepresentation",
            Self::EmptyRepresentation => "dd_EmptyRepresentation",
            Self::InputFileNotFound => "dd_IFileNotFound",
            Self::OutputFileNotOpen => "dd_OFileNotOpen",
            Self::NoLpObjective => "dd_NoLPObjective",
            Self::NoRealNumberSupport => "dd_NoRealNumberSupport",
            Self::NotAvailForH => "dd_NotAvailForH",
            Self::NotAvailForV => "dd_NotAvailForV",
            Self::CannotHandleLinearity => "dd_CannotHandleLinearity",
            Self::RowIndexOutOfRange => "dd_RowIndexOutOfRange",
            Self::ColIndexOutOfRange => "dd_ColIndexOutOfRange",
            Self::LpCycling => "dd_LPCycling",
            Self::NumericallyInconsistent => "dd_NumericallyInconsistent",
            Self::NoError => "dd_NoError",
        }
    }
}

impl std::fmt::Display for CddErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code={})", self.cddlib_name(), self.as_raw())
    }
}

#[derive(Debug, Error)]
pub enum CddError {
    #[error("cddlib returned {0}")]
    Cdd(CddErrorCode),
    #[error("cddlib returned a null pointer")]
    NullPointer,
    #[error("LP error")]
    LpError,
    #[error("LP did not have an optimal solution (status={0:?})")]
    LpStatus(LpStatus),
    #[error("cddlib operation returned failure")]
    OpFailed,
}

#[derive(Debug, Error)]
pub enum CddWrapperError {
    #[error(transparent)]
    Cdd(#[from] CddError),
    #[error("invalid matrix dimensions (rows={rows}, cols={cols})")]
    InvalidMatrix { rows: usize, cols: usize },
}

pub type CddResult<T> = std::result::Result<T, CddWrapperError>;

fn representation_to_raw(repr: RepresentationKind) -> u32 {
    match repr {
        RepresentationKind::Inequality => 1,
        RepresentationKind::Generator => 2,
    }
}

fn representation_from_raw(raw: u32) -> RepresentationKind {
    match raw {
        1 => RepresentationKind::Inequality,
        2 => RepresentationKind::Generator,
        other => panic!("unknown dd_RepresentationType value {other}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberType {
    Real,
    Rational,
}

impl NumberType {
    fn to_raw(self) -> u32 {
        match self {
            NumberType::Real => 1,
            NumberType::Rational => 2,
        }
    }

    fn from_raw(raw: u32) -> Self {
        match raw {
            1 => NumberType::Real,
            2 => NumberType::Rational,
            other => panic!("unknown dd_NumberType value {other}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpObjective {
    None,
    Maximize,
    Minimize,
}

impl LpObjective {
    fn to_raw(self) -> u32 {
        match self {
            LpObjective::None => 0,
            LpObjective::Maximize => 1,
            LpObjective::Minimize => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    Undecided,
    Optimal,
    Inconsistent,
    DualInconsistent,
    StructuralInconsistent,
    StructuralDualInconsistent,
    Unbounded,
    DualUnbounded,
}

impl LpStatus {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => LpStatus::Undecided,
            1 => LpStatus::Optimal,
            2 => LpStatus::Inconsistent,
            3 => LpStatus::DualInconsistent,
            4 => LpStatus::StructuralInconsistent,
            5 => LpStatus::StructuralDualInconsistent,
            6 => LpStatus::Unbounded,
            7 => LpStatus::DualUnbounded,
            other => panic!("unknown dd_LPStatusType value {other}"),
        }
    }
}

#[derive(Debug)]
pub struct Matrix<N: CddNumber = DefaultNumber, R: RepresentationMarker = Inequality> {
    raw: raw::MatrixData<N>,
    _no_send_sync: PhantomData<Rc<()>>,
    _repr: PhantomData<R>,
}

#[derive(Debug)]
pub struct CanonicalForm<N: CddNumber = DefaultNumber, R: RepresentationMarker = Inequality> {
    pub matrix: Matrix<N, R>,
    pub implicit_linearity: Vec<usize>,
    pub redundant_rows: Vec<usize>,
    pub positions: Vec<isize>,
}

impl<N: CddNumber, R: RepresentationMarker> Matrix<N, R> {
    pub fn new(rows: usize, cols: usize, num_type: NumberType) -> CddResult<Self> {
        Ok(Matrix {
            raw: raw::MatrixData::new(rows, cols, R::KIND, num_type)?,
            _no_send_sync: PhantomData,
            _repr: PhantomData,
        })
    }

    pub fn rows(&self) -> usize {
        self.raw.rows()
    }

    pub fn cols(&self) -> usize {
        self.raw.cols()
    }

    pub fn representation(&self) -> RepresentationKind {
        R::KIND
    }

    pub fn number_type(&self) -> NumberType {
        self.raw.number_type()
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    /// # Safety
    ///
    /// - `ptr` must be a non-null `dd_MatrixPtr` compatible with the backend `N`.
    /// - The pointer must be valid for the lifetime of this wrapper and must be
    ///   owned by this wrapper (it will be freed on drop).
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Matrix {
            raw: raw::MatrixData::from_raw(ptr),
            _no_send_sync: PhantomData,
            _repr: PhantomData,
        }
    }

    pub fn clone_cdd(&self) -> CddResult<Self> {
        Ok(Self {
            raw: self.raw.clone_cdd()?,
            _no_send_sync: PhantomData,
            _repr: PhantomData,
        })
    }

    pub fn set(&mut self, row: usize, col: usize, value: &N) {
        self.raw.set_mytype(row, col, value);
    }

    pub fn get(&self, row: usize, col: usize) -> N {
        self.raw.get_mytype(row, col)
    }

    pub fn set_real(&mut self, row: usize, col: usize, value: f64) {
        self.raw.set_real(row, col, value);
    }

    pub fn set_int(&mut self, row: usize, col: usize, value: std::os::raw::c_long) {
        self.raw.set_int(row, col, value);
    }

    pub fn get_real(&self, row: usize, col: usize) -> f64 {
        self.raw.get_real(row, col)
    }

    pub fn set_objective_real(&mut self, coeffs: &[f64]) {
        self.raw.set_objective_real(coeffs);
    }

    pub fn append_rows_in_place(&mut self, rows: &Matrix<N, R>) -> CddResult<()> {
        self.raw.append_rows_in_place(&rows.raw)
    }

    pub fn append_rows(&self, rows: &Matrix<N, R>) -> CddResult<Self> {
        Ok(Self {
            raw: self.raw.append_rows(&rows.raw)?,
            _no_send_sync: PhantomData,
            _repr: PhantomData,
        })
    }

    pub fn remove_row(&mut self, row: usize) -> CddResult<()> {
        self.raw.remove_row(row)
    }

    pub fn is_row_redundant(&self, row: usize) -> CddResult<bool> {
        self.raw.is_row_redundant(row)
    }

    pub fn redundant_rows(&self) -> CddResult<Vec<usize>> {
        self.raw.redundant_rows()
    }

    pub fn canonicalize(&self) -> CddResult<CanonicalForm<N, R>> {
        let (canon_raw, implicit, redundant, positions) = self.raw.canonicalize()?;
        let canon = Matrix {
            raw: canon_raw,
            _no_send_sync: PhantomData,
            _repr: PhantomData,
        };
        Ok(CanonicalForm {
            matrix: canon,
            implicit_linearity: implicit,
            redundant_rows: redundant,
            positions,
        })
    }
}

impl<N: CddNumber> Matrix<N, Generator> {
    pub fn set_generator_type(&mut self, row: usize, is_vertex: bool) {
        self.raw.set_generator_type(row, is_vertex);
    }

    pub fn from_vertices<const D: usize>(vertices: &[[N; D]]) -> CddResult<Self> {
        if vertices.is_empty() {
            return Err(CddWrapperError::InvalidMatrix {
                rows: 0,
                cols: D + 1,
            });
        }

        let mut m = Self::new(vertices.len(), D + 1, N::DEFAULT_NUMBER_TYPE)?;
        for (i, v) in vertices.iter().enumerate() {
            m.set_generator_type(i, true);
            for (j, coord) in v.iter().enumerate() {
                m.set(i, j + 1, coord);
            }
        }
        Ok(m)
    }

    /// Build a generator matrix from a dynamic list of vertices.
    ///
    /// The ambient dimension is determined from the first vertex. All vertices
    /// must have the same length.
    pub fn from_vertex_rows(vertices: &[Vec<N>]) -> CddResult<Self> {
        let Some(first) = vertices.first() else {
            return Err(CddWrapperError::InvalidMatrix { rows: 0, cols: 0 });
        };

        let dim = first.len();
        if dim == 0 {
            return Err(CddWrapperError::InvalidMatrix {
                rows: vertices.len(),
                cols: 0,
            });
        }

        if vertices.iter().skip(1).any(|v| v.len() != dim) {
            return Err(CddWrapperError::InvalidMatrix {
                rows: vertices.len(),
                cols: dim + 1,
            });
        }

        let mut m = Self::new(vertices.len(), dim + 1, N::DEFAULT_NUMBER_TYPE)?;
        for (row, coords) in vertices.iter().enumerate() {
            m.set_generator_type(row, true);
            for (col, coord) in coords.iter().enumerate() {
                m.set(row, col + 1, coord);
            }
        }
        Ok(m)
    }

    pub fn append_row(&self, coords: &[N], is_vertex: bool) -> CddResult<Self> {
        if coords.len() + 1 != self.cols() {
            return Err(CddWrapperError::InvalidMatrix {
                rows: self.rows() + 1,
                cols: coords.len() + 1,
            });
        }

        let mut row = Self::new(1, self.cols(), self.number_type())?;
        row.set_generator_type(0, is_vertex);
        for (col, val) in coords.iter().enumerate() {
            row.set(0, col + 1, val);
        }

        self.append_rows(&row)
    }
}

#[derive(Debug)]
pub struct SetFamily<N: CddNumber = DefaultNumber> {
    raw: raw::SetFamilyData<N>,
    _no_send_sync: PhantomData<Rc<()>>,
}

impl<N: CddNumber> SetFamily<N> {
    /// # Safety
    ///
    /// - `ptr` must be a non-null `dd_SetFamilyPtr` compatible with the backend `N`.
    /// - The pointer must be valid for the lifetime of this wrapper and must be
    ///   owned by this wrapper (it will be freed on drop).
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        SetFamily {
            raw: raw::SetFamilyData::from_raw(ptr),
            _no_send_sync: PhantomData,
        }
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.raw.as_raw()
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn universe_size(&self) -> usize {
        self.raw.universe_size()
    }

    pub fn to_adjacency_lists(&self) -> Vec<Vec<usize>> {
        self.raw.to_adjacency_lists()
    }
}

impl<N: CddNumber> From<SetFamily<N>> for HbSetFamily {
    fn from(value: SetFamily<N>) -> Self {
        let universe = value.universe_size();
        let n = value.len();

        let mut sets: Vec<HbRowSet> = Vec::with_capacity(n);
        for idx in 0..n {
            let members = value.raw.members(idx);
            sets.push(HbRowSet::from_indices(universe, &members));
        }

        HbSetFamily::from_sets(universe, sets)
    }
}

#[derive(Debug)]
pub struct Polyhedron<N: CddNumber = DefaultNumber> {
    raw: raw::PolyhedronData<N>,
    _no_send_sync: PhantomData<Rc<()>>,
}

impl<N: CddNumber> Polyhedron<N> {
    pub fn from_matrix<R: RepresentationMarker>(m: &Matrix<N, R>) -> CddResult<Self> {
        Ok(Self {
            raw: raw::PolyhedronData::from_matrix(&m.raw)?,
            _no_send_sync: PhantomData,
        })
    }

    pub fn from_vertices<const D: usize>(vertices: &[[N; D]]) -> CddResult<Self> {
        let m = Matrix::<N, Generator>::from_vertices(vertices)?;
        Self::from_matrix(&m)
    }

    pub fn from_vertex_rows(vertices: &[Vec<N>]) -> CddResult<Self> {
        let m = Matrix::<N, Generator>::from_vertex_rows(vertices)?;
        Self::from_matrix(&m)
    }

    pub fn facets(&self) -> CddResult<Matrix<N, Inequality>> {
        Ok(Matrix {
            raw: self.raw.facets()?,
            _no_send_sync: PhantomData,
            _repr: PhantomData,
        })
    }

    pub fn generators(&self) -> CddResult<Matrix<N, Generator>> {
        Ok(Matrix {
            raw: self.raw.generators()?,
            _no_send_sync: PhantomData,
            _repr: PhantomData,
        })
    }

    #[deprecated(
        note = "Uses cddlib's dd_CopyAdjacency, which has a serious performance-limiting bug that \
causes an approximately superexponential slowdown for \"large\" numbers of generators. Prefer \
hullabaloo's adjacency construction APIs instead."
    )]
    pub fn adjacency(&self) -> CddResult<SetFamily<N>> {
        Ok(SetFamily {
            raw: self.raw.adjacency()?,
            _no_send_sync: PhantomData,
        })
    }

    pub fn input_adjacency(&self) -> CddResult<SetFamily<N>> {
        Ok(SetFamily {
            raw: self.raw.input_adjacency()?,
            _no_send_sync: PhantomData,
        })
    }

    pub fn incidence(&self) -> CddResult<SetFamily<N>> {
        Ok(SetFamily {
            raw: self.raw.incidence()?,
            _no_send_sync: PhantomData,
        })
    }

    pub fn input_incidence(&self) -> CddResult<SetFamily<N>> {
        Ok(SetFamily {
            raw: self.raw.input_incidence()?,
            _no_send_sync: PhantomData,
        })
    }

    pub fn append_input_rows<R: RepresentationMarker>(
        &mut self,
        rows: &Matrix<N, R>,
    ) -> CddResult<()> {
        self.raw.append_input_rows(&rows.raw)
    }

    pub fn width_in_direction_real(&self, direction: &[f64]) -> CddResult<f64> {
        let mut h_min = self.facets()?;
        let mut h_max = h_min.clone_cdd()?;

        if direction.len() + 1 != h_max.cols() {
            return Err(CddWrapperError::InvalidMatrix {
                rows: h_max.rows(),
                cols: h_max.cols(),
            });
        }

        let cols = h_max.cols();
        let mut coeffs = vec![0.0f64; cols];

        for (i, &u_i) in direction.iter().enumerate() {
            coeffs[i + 1] = u_i;
        }

        h_max.set_objective_real(&coeffs);
        let lp_max = Lp::<N>::from_matrix(&mut h_max, LpObjective::Maximize)?;
        let sol_max = lp_max.solve()?;
        let max_val = sol_max.opt_value_real();

        h_min.set_objective_real(&coeffs);
        let lp_min = Lp::<N>::from_matrix(&mut h_min, LpObjective::Minimize)?;
        let sol_min = lp_min.solve()?;
        let min_val = sol_min.opt_value_real();

        Ok(max_val - min_val)
    }
}

#[derive(Debug)]
pub struct Lp<N: CddNumber = DefaultNumber> {
    raw: raw::LpData<N>,
    _no_send_sync: PhantomData<Rc<()>>,
}

#[derive(Debug)]
pub struct LpSolution<N: CddNumber = DefaultNumber> {
    raw: raw::LpSolutionData<N>,
    _no_send_sync: PhantomData<Rc<()>>,
}

impl<N: CddNumber> Lp<N> {
    pub fn from_matrix(matrix: &mut Matrix<N>, objective: LpObjective) -> CddResult<Self> {
        Ok(Lp {
            raw: raw::LpData::from_matrix(&mut matrix.raw, objective)?,
            _no_send_sync: PhantomData,
        })
    }

    pub fn solve(&self) -> CddResult<LpSolution<N>> {
        Ok(LpSolution {
            raw: self.raw.solve()?,
            _no_send_sync: PhantomData,
        })
    }
}

impl<N: CddNumber> LpSolution<N> {
    pub fn opt_value_real(&self) -> f64 {
        self.raw.opt_value_real()
    }
}

#[cfg(feature = "f64")]
pub type MatrixF64 = Matrix<f64>;
#[cfg(feature = "f64")]
pub type PolyhedronF64 = Polyhedron<f64>;

#[cfg(feature = "gmp")]
pub type MatrixGmpFloat = Matrix<CddFloat>;
#[cfg(feature = "gmp")]
pub type PolyhedronGmpFloat = Polyhedron<CddFloat>;

#[cfg(feature = "gmprational")]
pub type MatrixGmpRational = Matrix<CddRational>;
#[cfg(feature = "gmprational")]
pub type PolyhedronGmpRational = Polyhedron<CddRational>;

impl<N: CddNumber, R: RepresentationMarker> Matrix<N, R> {
    pub fn convert<M: CddNumber>(&self) -> CddResult<Matrix<M, R>> {
        convert_matrix_via_real::<N, M, R>(self)
    }
}

#[cfg(all(feature = "f64", feature = "gmp"))]
impl<R: RepresentationMarker> From<Matrix<f64, R>> for Matrix<CddFloat, R> {
    fn from(value: Matrix<f64, R>) -> Self {
        value
            .convert()
            .expect("Matrix<f64> -> Matrix<CddFloat> conversion failed")
    }
}

#[cfg(all(feature = "f64", feature = "gmprational"))]
impl<R: RepresentationMarker> From<Matrix<f64, R>> for Matrix<CddRational, R> {
    fn from(value: Matrix<f64, R>) -> Self {
        value
            .convert()
            .expect("Matrix<f64> -> Matrix<CddRational> conversion failed")
    }
}

#[cfg(all(feature = "f64", feature = "gmp"))]
impl<R: RepresentationMarker> From<Matrix<CddFloat, R>> for Matrix<f64, R> {
    fn from(value: Matrix<CddFloat, R>) -> Self {
        value
            .convert()
            .expect("Matrix<CddFloat> -> Matrix<f64> conversion failed")
    }
}

#[cfg(all(feature = "f64", feature = "gmprational"))]
impl<R: RepresentationMarker> From<Matrix<CddRational, R>> for Matrix<f64, R> {
    fn from(value: Matrix<CddRational, R>) -> Self {
        value
            .convert()
            .expect("Matrix<CddRational> -> Matrix<f64> conversion failed")
    }
}

#[cfg(all(feature = "gmp", feature = "gmprational"))]
impl<R: RepresentationMarker> From<Matrix<CddFloat, R>> for Matrix<CddRational, R> {
    fn from(value: Matrix<CddFloat, R>) -> Self {
        value
            .convert()
            .expect("Matrix<CddFloat> -> Matrix<CddRational> conversion failed")
    }
}

#[cfg(all(feature = "gmp", feature = "gmprational"))]
impl<R: RepresentationMarker> From<Matrix<CddRational, R>> for Matrix<CddFloat, R> {
    fn from(value: Matrix<CddRational, R>) -> Self {
        value
            .convert()
            .expect("Matrix<CddRational> -> Matrix<CddFloat> conversion failed")
    }
}

fn convert_matrix_via_real<Src: CddNumber, Dst: CddNumber, R: RepresentationMarker>(
    src: &Matrix<Src, R>,
) -> CddResult<Matrix<Dst, R>> {
    let rows = src.rows();
    let cols = src.cols();
    let mut out = Matrix::<Dst, R>::new(rows, cols, Dst::DEFAULT_NUMBER_TYPE)?;

    for i in 0..rows {
        for j in 0..cols {
            out.set_real(i, j, src.get_real(i, j));
        }
    }

    out.raw.copy_objective_from(&src.raw);

    if let Some(coeffs) = src.raw.objective_row_coeffs_real() {
        out.set_objective_real(&coeffs);
    }

    Ok(out)
}
