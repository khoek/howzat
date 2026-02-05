use crate::HowzatError as Error;
use crate::dd::state::{ConeBasisPrep, ConeOutput};
use crate::dd::{DefaultNormalizer, SinglePrecisionUmpire, Umpire};
use crate::matrix::LpMatrix;
use calculo::num::{Epsilon, Normalizer, Num};
use hullabaloo::types::{InequalityKind, Representation, RowSet};

type SinglePrecisionOutput<N, R, E, NM> = ConeOutput<N, R, SinglePrecisionUmpire<N, E, NM>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnumerationMode {
    Exact,
    AssumeNondegenerate,
    Relaxed,
    RelaxedNondegenerate,
}

impl EnumerationMode {
    pub(crate) fn relaxed(self) -> bool {
        matches!(
            self,
            EnumerationMode::Relaxed | EnumerationMode::RelaxedNondegenerate
        )
    }

    pub(crate) fn assumes_nondegeneracy(self) -> bool {
        matches!(
            self,
            EnumerationMode::AssumeNondegenerate | EnumerationMode::RelaxedNondegenerate
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BasisInitialization {
    Top,
    Bot,
}

impl BasisInitialization {
    pub(crate) fn at_bot(self) -> bool {
        matches!(self, BasisInitialization::Bot)
    }
}

#[derive(Clone, Debug)]
pub struct ConeOptions {
    pub enumeration_mode: EnumerationMode,
    pub basis_initialization: BasisInitialization,
}

#[derive(Clone, Debug, Default)]
pub struct ConeOptionsBuilder {
    pub(crate) options: ConeOptions,
}

impl Default for ConeOptions {
    fn default() -> Self {
        Self {
            enumeration_mode: EnumerationMode::Exact,
            basis_initialization: BasisInitialization::Top,
        }
    }
}

impl ConeOptions {
    pub fn builder() -> ConeOptionsBuilder {
        ConeOptionsBuilder {
            options: Self::default(),
        }
    }

    pub fn enumeration_mode(&self) -> EnumerationMode {
        self.enumeration_mode
    }

    pub fn basis_initialization(&self) -> BasisInitialization {
        self.basis_initialization
    }

    pub fn assumes_nondegeneracy(&self) -> bool {
        self.enumeration_mode.assumes_nondegeneracy()
    }

    pub fn relaxed_enumeration(&self) -> bool {
        self.enumeration_mode.relaxed()
    }

    pub fn init_basis_at_bot(&self) -> bool {
        self.basis_initialization.at_bot()
    }
}

impl ConeOptionsBuilder {
    pub fn enumeration_mode(&mut self, enumeration_mode: EnumerationMode) -> &mut Self {
        self.options.enumeration_mode = enumeration_mode;
        self
    }

    pub fn basis_initialization(&mut self, basis_initialization: BasisInitialization) -> &mut Self {
        self.options.basis_initialization = basis_initialization;
        self
    }

    pub fn finish(&mut self) -> Result<ConeOptions, Error> {
        Ok(std::mem::take(&mut self.options))
    }
}

#[derive(Clone, Debug)]
pub struct Cone<N: Num, R: Representation> {
    pub(crate) matrix: LpMatrix<N, R>,
    pub(crate) equality_kinds: Vec<InequalityKind>,
    pub(crate) options: ConeOptions,
    pub(crate) ground_set: RowSet,
    pub(crate) equality_set: RowSet,
    pub(crate) _strict_inequality_set: RowSet,
}

#[derive(Clone, Debug)]
pub struct ConeBuilder<N: Num, R: Representation> {
    matrix: LpMatrix<N, R>,
    equality_kinds: Vec<InequalityKind>,
    options: ConeOptions,
}

impl<N: Num, R: Representation> Cone<N, R> {
    pub fn new(
        matrix: LpMatrix<N, R>,
        equality_kinds: Vec<InequalityKind>,
        options: ConeOptions,
    ) -> Result<Self, Error> {
        assert_eq!(
            equality_kinds.len(),
            matrix.row_count(),
            "equality_kinds length mismatch (kinds={} rows={})",
            equality_kinds.len(),
            matrix.row_count()
        );

        let m = matrix.row_count();

        let ground_set = RowSet::all(m);
        let mut strict_inequality_set = RowSet::new(m);
        let mut equality_set = RowSet::new(m);
        for (i, kind) in equality_kinds.iter().enumerate() {
            match kind {
                InequalityKind::Equality => {
                    equality_set.insert(i);
                }
                InequalityKind::StrictInequality => {
                    strict_inequality_set.insert(i);
                }
                InequalityKind::Inequality => {}
            }
        }

        Ok(Self {
            matrix,
            equality_kinds,
            options,
            ground_set,
            equality_set,
            _strict_inequality_set: strict_inequality_set,
        })
    }

    pub fn into_basis_prep_with_normalizer<E: Epsilon<N>, NM: Normalizer<N>>(
        self,
        eps: E,
        normalizer: NM,
    ) -> ConeBasisPrep<N, R, SinglePrecisionUmpire<N, E, NM>>
    {
        let umpire = SinglePrecisionUmpire::with_normalizer(eps, normalizer);
        ConeBasisPrep::new(crate::dd::state::ConeEngine::new_with_umpire(self, umpire))
    }

    pub fn into_basis_prep_with_umpire<U: Umpire<N>>(self, umpire: U) -> ConeBasisPrep<N, R, U> {
        ConeBasisPrep::new(crate::dd::state::ConeEngine::new_with_umpire(self, umpire))
    }

    /// Select the zero-set representation at compile time.
    ///
    /// ```no_run
    /// use howzat::dd::{Cone, RowRepr, SatRepr, SinglePrecisionUmpire};
    /// use howzat::matrix::LpMatrix;
    /// use calculo::num::Num;
    /// use hullabaloo::types::{Inequality, InequalityKind};
    ///
    /// let matrix = LpMatrix::<f64, Inequality>::from_rows(vec![vec![1.0], vec![-1.0]]);
    /// let kinds = vec![InequalityKind::Inequality; 2];
    /// let cone = Cone::new(matrix, kinds, Default::default()).unwrap();
    ///
    /// let umpire = SinglePrecisionUmpire::new(f64::default_eps());
    /// let _sat = cone
    ///     .clone()
    ///     .into_basis_prep_with_umpire_and_zero_repr::<_, SatRepr>(umpire.clone());
    /// let _row = cone.into_basis_prep_with_umpire_and_zero_repr::<_, RowRepr>(umpire);
    /// ```
    pub fn into_basis_prep_with_umpire_and_zero_repr<U, ZR>(
        self,
        umpire: U,
    ) -> ConeBasisPrep<N, R, U, ZR>
    where
        U: Umpire<N, ZR>,
        ZR: crate::dd::ZeroRepr,
    {
        ConeBasisPrep::new(crate::dd::state::ConeEngine::<N, R, U, ZR>::new_with_umpire(
            self, umpire,
        ))
    }

    pub fn run_dd_with_normalizer<E: Epsilon<N>, NM: Normalizer<N>>(
        self,
        eps: E,
        normalizer: NM,
    ) -> Result<SinglePrecisionOutput<N, R, E, NM>, Error>
    {
        self.into_basis_prep_with_normalizer(eps, normalizer)
            .run_dd()
    }

    pub fn options(&self) -> &ConeOptions {
        &self.options
    }

    pub fn matrix(&self) -> &LpMatrix<N, R> {
        &self.matrix
    }

    pub fn ground_set(&self) -> &RowSet {
        &self.ground_set
    }

    pub fn equality_set(&self) -> &RowSet {
        &self.equality_set
    }

    pub fn equality_kinds(&self) -> &[InequalityKind] {
        &self.equality_kinds
    }
}

impl<N: DefaultNormalizer, R: Representation> Cone<N, R> {
    pub fn into_basis_prep<E: Epsilon<N>>(
        self,
        eps: E,
    ) -> ConeBasisPrep<N, R, SinglePrecisionUmpire<N, E>> {
        let umpire = SinglePrecisionUmpire::new(eps);
        ConeBasisPrep::new(crate::dd::state::ConeEngine::new_with_umpire(self, umpire))
    }

    pub fn run_dd<E: Epsilon<N>>(
        self,
        eps: E,
    ) -> Result<ConeOutput<N, R, SinglePrecisionUmpire<N, E>>, Error> {
        self.into_basis_prep(eps).run_dd()
    }
}

impl<N: Num, R: Representation> ConeBuilder<N, R> {
    pub fn new(matrix: LpMatrix<N, R>, equality_kinds: Vec<InequalityKind>) -> Self {
        Self {
            matrix,
            equality_kinds,
            options: ConeOptions::default(),
        }
    }

    pub fn options(&mut self, options: ConeOptions) -> &mut Self {
        self.options = options;
        self
    }

    pub fn configure_options(
        &mut self,
        configure: impl FnOnce(&mut ConeOptionsBuilder) -> Result<(), Error>,
    ) -> Result<&mut Self, Error> {
        let mut builder = ConeOptions::builder();
        builder.options = self.options.clone();
        configure(&mut builder)?;
        self.options = builder.finish()?;
        Ok(self)
    }

    pub fn finish(&mut self) -> Result<Cone<N, R>, Error> {
        let matrix = std::mem::replace(&mut self.matrix, LpMatrix::<N, R>::new(0, 0));
        let equality_kinds = std::mem::take(&mut self.equality_kinds);
        let options = self.options.clone();
        Cone::new(matrix, equality_kinds, options)
    }
}
