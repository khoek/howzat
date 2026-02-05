use std::{
    fmt,
    time::{Duration, Instant},
};

mod cddlib;
mod howzat_common;
mod howzat_dd;
mod howzat_lrs;
mod lrslib;
mod ppl;

use anyhow::{anyhow, ensure};
use calculo::num::Num;
use howzat::dd::ConeOptions;
use hullabaloo::AdjacencyList;
use hullabaloo::set_family::{ListFamily, SetFamily};
use hullabaloo::types::AdjacencyOutput;
use serde::{Deserialize, Serialize};

use crate::inequalities::{
    HowzatInequalities, RowMajorInequalities, RowMajorInequalitiesI64, InequalitiesF64,
    InequalitiesI64,
};
use crate::vertices::{
    HomogeneousGeneratorRowsF64, HomogeneousGeneratorRowsI64, HowzatVertices,
    RowMajorHomogeneousGenerators, RowMajorHomogeneousGeneratorsI64, RowMajorVertices,
    RowMajorVerticesI64, VerticesF64, VerticesI64,
};

use howzat_dd::{
    DEFAULT_HOWZAT_DD_PIPELINE, HowzatDdPipelineSpec, HowzatDdPurifierSpec, HowzatDdUmpire,
    parse_howzat_dd_pipeline, parse_howzat_dd_purifier,
};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum BackendSpec {
    CddlibF64,
    CddlibGmpFloat,
    CddlibGmpRational,
    CddlibHlblF64,
    CddlibHlblGmpFloat,
    CddlibHlblGmpRational,
    HowzatDd {
        umpire: HowzatDdUmpire,
        purifier: Option<HowzatDdPurifierSpec>,
        pipeline: HowzatDdPipelineSpec,
    },
    HowzatLrsRug,
    HowzatLrsDashu,
    LrslibHlblGmpInt,
    PplHlblGmpInt,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum RequestedAdjacency {
    Default,
    Dense,
    Sparse,
}

impl RequestedAdjacency {
    fn token(self) -> Option<&'static str> {
        match self {
            Self::Default => None,
            Self::Dense => Some("dense"),
            Self::Sparse => Some("sparse"),
        }
    }
}

impl BackendSpec {
    fn supports_dense_adjacency(&self) -> bool {
        !matches!(
            self,
            Self::CddlibF64
                | Self::CddlibGmpFloat
                | Self::CddlibGmpRational
                | Self::CddlibHlblF64
                | Self::CddlibHlblGmpFloat
                | Self::CddlibHlblGmpRational
        )
    }

    fn supports_sparse_adjacency(&self) -> bool {
        true
    }

    fn fmt_kind(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CddlibF64 | Self::CddlibGmpFloat | Self::CddlibGmpRational => f.write_str("cddlib"),
            Self::CddlibHlblF64 | Self::CddlibHlblGmpFloat | Self::CddlibHlblGmpRational => {
                f.write_str("cddlib+hlbl")
            }
            Self::HowzatDd { umpire, .. } => {
                f.write_str("howzat-dd")?;
                if let Some(token) = umpire.canonical_token() {
                    write!(f, "@{token}")?;
                }
                Ok(())
            }
            Self::HowzatLrsRug | Self::HowzatLrsDashu => f.write_str("howzat-lrs"),
            Self::LrslibHlblGmpInt => f.write_str("lrslib+hlbl"),
            Self::PplHlblGmpInt => f.write_str("ppl+hlbl"),
        }
    }

    fn fmt_num(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CddlibF64 | Self::CddlibHlblF64 => f.write_str("f64"),
            Self::CddlibGmpFloat | Self::CddlibHlblGmpFloat => f.write_str("gmpfloat"),
            Self::CddlibGmpRational | Self::CddlibHlblGmpRational => f.write_str("gmprational"),
            Self::HowzatDd { pipeline, .. } => f.write_str(&pipeline.canonical()),
            Self::HowzatLrsRug => f.write_str("rug"),
            Self::HowzatLrsDashu => f.write_str("dashu"),
            Self::LrslibHlblGmpInt => f.write_str("gmpint"),
            Self::PplHlblGmpInt => f.write_str("gmpint"),
        }
    }
}

impl fmt::Display for BackendSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_kind(f)?;
        f.write_str(":")?;
        self.fmt_num(f)
    }
}

impl std::str::FromStr for BackendSpec {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let raw = value.trim();
        if raw.is_empty() {
            return Err("backend spec cannot be empty".to_string());
        }

        let raw = raw.to_ascii_lowercase();

        let (kind, num) = raw
            .split_once(':')
            .map(|(k, n)| (k.trim(), Some(n.trim())))
            .unwrap_or((raw.trim(), None));

        let (kind, umpire) = if let Some((base, selector)) = kind.split_once('@') {
            if selector.contains('@') {
                return Err(format!(
                    "backend spec '{value}' contains multiple '@' selectors (expected e.g. howzat-dd@sp:...)"
                ));
            }
            let base = base.trim();
            let selector = selector.trim();
            if base.is_empty() || selector.is_empty() {
                return Err(format!(
                    "backend spec '{value}' has an invalid '@' selector (expected howzat-dd@sp or howzat-dd@int)"
                ));
            }
            let umpire = match selector {
                "int" => HowzatDdUmpire::Int,
                "sp" => HowzatDdUmpire::Sp,
                _ => {
                    return Err(format!(
                        "backend spec '{value}' has an unknown umpire selector '@{selector}' (expected '@int' or '@sp')"
                    ));
                }
            };
            (base, umpire)
        } else {
            (kind, HowzatDdUmpire::Default)
        };

        let spec = match (kind, num) {
            ("cddlib", None | Some("") | Some("gmprational")) => Self::CddlibGmpRational,
            ("cddlib", Some("f64")) => Self::CddlibF64,
            ("cddlib", Some("gmpfloat")) => Self::CddlibGmpFloat,
            ("cddlib+hlbl", None | Some("") | Some("gmprational")) => Self::CddlibHlblGmpRational,
            ("cddlib+hlbl", Some("f64")) => Self::CddlibHlblF64,
            ("cddlib+hlbl", Some("gmpfloat")) => Self::CddlibHlblGmpFloat,
            ("howzat-dd", None | Some("")) => Self::HowzatDd {
                umpire,
                purifier: None,
                pipeline: parse_howzat_dd_pipeline(DEFAULT_HOWZAT_DD_PIPELINE)?,
            },
            ("howzat-dd", Some(spec)) => Self::HowzatDd {
                umpire,
                purifier: None,
                pipeline: parse_howzat_dd_pipeline(spec)?,
            },
            ("howzat-lrs", None | Some("") | Some("rug")) => Self::HowzatLrsRug,
            ("howzat-lrs", Some("dashu")) => Self::HowzatLrsDashu,
            ("lrslib+hlbl", None | Some("") | Some("gmpint")) => Self::LrslibHlblGmpInt,
            ("ppl+hlbl", None | Some("") | Some("gmpint")) => Self::PplHlblGmpInt,
            _ => {
                return Err(format!(
                    "unknown backend spec '{value}' (see --help for supported values)"
                ));
            }
        };

        if umpire != HowzatDdUmpire::Default && !matches!(spec, Self::HowzatDd { .. }) {
            return Err(format!(
                "backend spec '{value}' does not support '@{token}' (only howzat-dd does)",
                token = umpire
                    .canonical_token()
                    .expect("selector is not Default")
            ));
        }

        Ok(spec)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Backend(BackendSpec, RequestedAdjacency);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BackendAdjacencyMode {
    Dense,
    Sparse,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Representation {
    EuclideanVertices,
    HomogeneousGenerators,
    Inequality,
}

impl serde::Serialize for Backend {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for Backend {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let spec = String::deserialize(deserializer)?;
        Self::parse(&spec).map_err(serde::de::Error::custom)
    }
}

impl Backend {
    pub fn parse(spec: &str) -> Result<Self, String> {
        spec.parse()
    }

    pub fn adjacency_mode(
        &self,
        vertex_count: usize,
        dim: usize,
        output_adjacency: bool,
    ) -> BackendAdjacencyMode {
        match self.choose_adjacency(vertex_count, dim, output_adjacency) {
            RequestedAdjacency::Dense => BackendAdjacencyMode::Dense,
            RequestedAdjacency::Sparse => BackendAdjacencyMode::Sparse,
            RequestedAdjacency::Default => unreachable!("choose_adjacency never returns Default"),
        }
    }

    pub fn minimum_output_level(&self) -> BackendOutputLevel {
        let (minimum, _) = match &self.0 {
            BackendSpec::HowzatDd { pipeline, .. } => {
                if pipeline.has_checks() {
                    (
                        BackendOutputLevel::Incidence,
                        Some("howzat-dd pipeline contains Check steps that require incidence"),
                    )
                } else {
                    (BackendOutputLevel::Representation, None)
                }
            }
            BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu => (
                BackendOutputLevel::Incidence,
                Some("howzat-lrs always computes an incidence certificate before exact resolution"),
            ),
            BackendSpec::LrslibHlblGmpInt => (
                BackendOutputLevel::Incidence,
                Some("lrslib solve() always produces incidence"),
            ),
            BackendSpec::PplHlblGmpInt => (
                BackendOutputLevel::Incidence,
                Some("ppl solve() always produces incidence"),
            ),
            _ => (BackendOutputLevel::Representation, None),
        };

        minimum
    }

    const DEFAULT_DENSE_ADJACENCY_LIMIT_BYTES: u128 = 128 * 1024 * 1024;
    const DEFAULT_DENSE_ADJACENCY_LIMIT_NODES: u128 = 32768;

    fn choose_adjacency(
        &self,
        vertex_count: usize,
        dim: usize,
        output_adjacency: bool,
    ) -> RequestedAdjacency {
        match self.1 {
            RequestedAdjacency::Dense => RequestedAdjacency::Dense,
            RequestedAdjacency::Sparse => RequestedAdjacency::Sparse,
            RequestedAdjacency::Default => {
                if !self.0.supports_dense_adjacency() {
                    return RequestedAdjacency::Sparse;
                }

                fn dense_graph_bytes(node_count: u128) -> u128 {
                    node_count.saturating_mul(node_count).saturating_add(7) / 8
                }

                fn binom_capped(n: u128, k: u128, cap: u128) -> u128 {
                    if k > n {
                        return 0;
                    }
                    let k = k.min(n - k);
                    if k == 0 {
                        return 1;
                    }

                    let mut result = 1u128;
                    for i in 1..=k {
                        let numerator = n - k + i;
                        let Some(product) = result.checked_mul(numerator) else {
                            return cap + 1;
                        };
                        result = product / i;
                        if result > cap {
                            return cap + 1;
                        }
                    }
                    result
                }

                fn cyclic_facets_capped(vertex_count: usize, dim: usize, cap: u128) -> u128 {
                    let Some(max_dim) = vertex_count.checked_sub(1) else {
                        return 0;
                    };
                    let d = dim.min(max_dim);
                    if d < 2 {
                        return 0;
                    }

                    let n = vertex_count as u128;
                    let d = d as u128;
                    let m = d / 2;
                    if m == 0 {
                        return 0;
                    }

                    if d % 2 == 0 {
                        let a = binom_capped(n.saturating_sub(m), m, cap);
                        if a > cap {
                            return cap + 1;
                        }
                        let b = binom_capped(n.saturating_sub(m + 1), m - 1, cap);
                        if b > cap {
                            return cap + 1;
                        }
                        a.saturating_add(b).min(cap + 1)
                    } else {
                        let a = binom_capped(n.saturating_sub(m + 1), m, cap);
                        if a > cap {
                            return cap + 1;
                        }
                        a.saturating_mul(2).min(cap + 1)
                    }
                }

                let n = vertex_count as u128;
                if dense_graph_bytes(n) > Self::DEFAULT_DENSE_ADJACENCY_LIMIT_BYTES {
                    return RequestedAdjacency::Sparse;
                }

                if output_adjacency {
                    let max_facets = cyclic_facets_capped(
                        vertex_count,
                        dim,
                        Self::DEFAULT_DENSE_ADJACENCY_LIMIT_NODES,
                    );
                    if max_facets > Self::DEFAULT_DENSE_ADJACENCY_LIMIT_NODES {
                        return RequestedAdjacency::Sparse;
                    }
                }

                RequestedAdjacency::Dense
            }
        }
    }

    pub fn solve(
        &self,
        repr: Representation,
        data: &[Vec<f64>],
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, anyhow::Error> {
        match repr {
            Representation::EuclideanVertices => {
                let dim = data.first().map_or(0, |v| v.len());
                match self.choose_adjacency(data.len(), dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self.solve_dense(data, config).map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => {
                        self.solve_sparse(data, config).map(BackendRunAny::Sparse)
                    }
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::HomogeneousGenerators => {
                self.ensure_generator_matrix_backend()?;
                let generators = HomogeneousGeneratorRowsF64::new(data)?;
                let dim = generators.dim().saturating_sub(1);
                match self.choose_adjacency(generators.vertex_count(), dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_dense(&generators, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_sparse(&generators, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::Inequality => {
                let dim = data.first().map_or(0, |row| row.len().saturating_sub(1));
                match self.choose_adjacency(data.len(), dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_dense_inequalities(data, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_sparse_inequalities(data, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
        }
    }

    pub fn solve_row_major(
        &self,
        repr: Representation,
        data: &[f64],
        rows: usize,
        cols: usize,
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, anyhow::Error> {
        match repr {
            Representation::EuclideanVertices => {
                let vertices = RowMajorVertices::new(data, rows, cols)?;
                match self.choose_adjacency(rows, cols, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_dense(&vertices, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_sparse(&vertices, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::HomogeneousGenerators => {
                self.ensure_generator_matrix_backend()?;
                let generators = RowMajorHomogeneousGenerators::new(data, rows, cols)?;
                let dim = cols.saturating_sub(1);
                match self.choose_adjacency(rows, dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_dense(&generators, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_sparse(&generators, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::Inequality => {
                let dim = cols.saturating_sub(1);
                let inequalities = RowMajorInequalities::new(data, rows, dim)?;
                match self.choose_adjacency(rows, dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_dense_inequalities(&inequalities, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_sparse_inequalities(&inequalities, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
        }
    }

    pub fn solve_exact(
        &self,
        repr: Representation,
        data: &[Vec<i64>],
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, anyhow::Error> {
        self.ensure_exact_backend()?;
        match repr {
            Representation::EuclideanVertices => {
                let dim = data.first().map_or(0, |v| v.len());
                match self.choose_adjacency(data.len(), dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_exact_dense(data, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_exact_sparse(data, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::HomogeneousGenerators => {
                self.ensure_generator_matrix_backend()?;
                let generators = HomogeneousGeneratorRowsI64::new(data)?;
                let dim = generators.dim().saturating_sub(1);
                match self.choose_adjacency(generators.vertex_count(), dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_exact_dense(&generators, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_exact_sparse(&generators, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::Inequality => {
                let dim = data.first().map_or(0, |row| row.len().saturating_sub(1));
                match self.choose_adjacency(data.len(), dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_exact_dense_inequalities(data, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_exact_sparse_inequalities(data, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
        }
    }

    pub fn solve_row_major_exact(
        &self,
        repr: Representation,
        data: &[i64],
        rows: usize,
        cols: usize,
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, anyhow::Error> {
        self.ensure_exact_backend()?;
        match repr {
            Representation::EuclideanVertices => {
                let vertices = RowMajorVerticesI64::new(data, rows, cols)?;
                match self.choose_adjacency(rows, cols, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_exact_dense(&vertices, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_exact_sparse(&vertices, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::HomogeneousGenerators => {
                self.ensure_generator_matrix_backend()?;
                let generators = RowMajorHomogeneousGeneratorsI64::new(data, rows, cols)?;
                let dim = cols.saturating_sub(1);
                match self.choose_adjacency(rows, dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_exact_dense(&generators, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_exact_sparse(&generators, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
            Representation::Inequality => {
                let dim = cols.saturating_sub(1);
                let inequalities = RowMajorInequalitiesI64::new(data, rows, dim)?;
                match self.choose_adjacency(rows, dim, config.output_adjacency) {
                    RequestedAdjacency::Dense => self
                        .solve_exact_dense_inequalities(&inequalities, config)
                        .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => self
                        .solve_exact_sparse_inequalities(&inequalities, config)
                        .map(BackendRunAny::Sparse),
                    RequestedAdjacency::Default => {
                        unreachable!("choose_adjacency never returns Default")
                    }
                }
            }
        }
    }

    pub fn solve_row_major_exact_gmprat(
        &self,
        repr: Representation,
        data: Vec<calculo::num::RugRat>,
        rows: usize,
        cols: usize,
        config: &BackendRunConfig,
    ) -> Result<BackendRunAny, anyhow::Error> {
        use howzat::matrix::LpMatrixBuilder;
        use howzat::dd::{DefaultNormalizer, IntUmpire, SinglePrecisionUmpire as SpUmpire, SnapPurifier as Snap};
        use hullabaloo::types::{Generator, IncidenceOutput, Inequality};

        self.ensure_exact_backend()?;
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let (dd_umpire, purifier, pipeline) = match &self.0 {
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                (*umpire, *purifier, pipeline)
            }
            _ => {
                return Err(anyhow!(
                    "backend '{self}' does not support exact gmprat input"
                ));
            }
        };

        ensure!(
            pipeline.canonical() == "gmprat",
            "{self} only supports exact gmprat input for howzat-dd:gmprat"
        );

        let start_total = Instant::now();
        let timing = config.timing_detail;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::Exact
        } else {
            CoefficientMode::Off
        };

        let howzat_output_adjacency = if output_adjacency {
            config.howzat_output_adjacency
        } else {
            AdjacencyOutput::Off
        };
        let howzat_output_incidence = if config.output_incidence {
            IncidenceOutput::Set
        } else {
            IncidenceOutput::Off
        };

        let dd = |output_adjacency| howzat::polyhedron::DdConfig {
            cone: config.howzat_options.clone(),
            poly: howzat::polyhedron::PolyhedronOptions {
                output_incidence: howzat_output_incidence,
                output_adjacency,
                input_incidence: IncidenceOutput::Off,
                input_adjacency: AdjacencyOutput::Off,
                save_basis_and_tableau: false,
                save_repair_hints: false,
                profile_adjacency: false,
            },
        };

        fn run_howzat_exact_dd<Inc, Adj, R>(
            spec: Backend,
            dd_umpire: HowzatDdUmpire,
            purifier: Option<HowzatDdPurifierSpec>,
            output_adjacency: bool,
            timing: bool,
            coeff_mode: CoefficientMode,
            dd: howzat::polyhedron::DdConfig,
            matrix: howzat::matrix::LpMatrix<calculo::num::RugRat, R>,
            time_matrix: Duration,
            start_total: Instant,
        ) -> Result<BackendRun<Inc, Adj>, anyhow::Error>
        where
            Inc: From<SetFamily>,
            Adj: hullabaloo::adjacency::AdjacencyStore,
            R: hullabaloo::types::DualRepresentation,
        {
            let start_dd = Instant::now();
            let poly = match dd_umpire {
                HowzatDdUmpire::Default | HowzatDdUmpire::Int => {
                    ensure!(
                        purifier.is_none(),
                        "{spec} does not support purification for exact gmprat input under @int"
                    );
                    let umpire = IntUmpire::new(calculo::num::RugRat::default_eps());
                    howzat::polyhedron::PolyhedronOutput::<calculo::num::RugRat, R>::from_matrix_dd_int_with_umpire(
                        matrix, dd, umpire,
                    )
                }
                HowzatDdUmpire::Sp => {
                    let eps = calculo::num::RugRat::default_eps();
                    let normalizer = <calculo::num::RugRat as DefaultNormalizer>::Norm::default();
                    match purifier {
                        None => {
                            let umpire = SpUmpire::with_normalizer(eps, normalizer);
                            howzat::polyhedron::PolyhedronOutput::<calculo::num::RugRat, R>::from_matrix_dd(
                                matrix, dd, umpire,
                            )
                        }
                        Some(HowzatDdPurifierSpec::Snap) => {
                            let umpire = SpUmpire::with_purifier(eps, normalizer, Snap::new());
                            howzat::polyhedron::PolyhedronOutput::<calculo::num::RugRat, R>::from_matrix_dd(
                                matrix, dd, umpire,
                            )
                        }
                        Some(HowzatDdPurifierSpec::UpSnap(_)) => {
                            return Err(anyhow!(
                                "{spec} does not support purify[upsnap[...]] under exact gmprat input"
                            ));
                        }
                    }
                }
            }
            .map_err(|err| anyhow!("howzat-dd failed: {err}"))?;
            let time_dd = start_dd.elapsed();

            let store_facet_row_indices = coeff_mode != CoefficientMode::Off;
            let (geometry, extract_detail) = howzat_common::summarize_howzat_geometry::<Inc, Adj, _, _>(
                &poly,
                output_adjacency,
                timing,
                store_facet_row_indices,
            )?;
            let coefficients = if coeff_mode == CoefficientMode::Off {
                None
            } else {
                let Some(facet_row_indices) = geometry.facet_row_indices.as_deref() else {
                    return Err(anyhow!("internal: facet_row_indices missing"));
                };
                howzat_common::extract_howzat_coefficients(&poly, facet_row_indices, coeff_mode)?
            };

            let total = start_total.elapsed();
            let detail = timing.then(|| {
                TimingDetail::HowzatDd(HowzatDdTimingDetail {
                    fast_matrix: Duration::ZERO,
                    fast_dd: Duration::ZERO,
                    cert: Duration::ZERO,
                    repair_partial: Duration::ZERO,
                    repair_graph: Duration::ZERO,
                    exact_matrix: time_matrix,
                    exact_dd: time_dd,
                    incidence: extract_detail.incidence,
                    vertex_adjacency: extract_detail.vertex_adjacency,
                    facet_adjacency: extract_detail.facet_adjacency,
                })
            });

            Ok(BackendRun {
                spec,
                stats: geometry.stats,
                timing: BackendTiming {
                    total,
                    fast: None,
                    resolve: None,
                    exact: Some(time_matrix + time_dd),
                },
                facets: None,
                coefficients,
                geometry: BackendGeometry::Input(InputGeometry {
                    vertex_adjacency: geometry.vertex_adjacency,
                    facets_to_vertices: geometry.facets_to_vertices,
                    facet_adjacency: geometry.facet_adjacency,
                }),
                fails: 0,
                fallbacks: 0,
                error: None,
                detail,
            })
        }

        match repr {
            Representation::EuclideanVertices => {
                ensure!(
                    rows
                        .checked_mul(cols)
                        .is_some_and(|len| len == data.len()),
                    "expected {rows}x{cols} coords but got {}",
                    data.len()
                );
                let Some(out_cols) = cols.checked_add(1) else {
                    return Err(anyhow!("howzat generator dimension too large"));
                };
                let mut input = data.into_iter();
                let start_matrix = Instant::now();
                let mut flat: Vec<calculo::num::RugRat> =
                    Vec::with_capacity(rows.saturating_mul(out_cols));
                for _ in 0..rows {
                    flat.push(calculo::num::RugRat(rug::Rational::from(1)));
                    flat.extend(input.by_ref().take(cols));
                }
                let time_matrix = start_matrix.elapsed();
                let matrix =
                    LpMatrixBuilder::<calculo::num::RugRat, Generator>::from_flat(rows, out_cols, flat)
                        .build();
                match self.choose_adjacency(rows, cols, output_adjacency) {
                    RequestedAdjacency::Dense => run_howzat_exact_dd::<SetFamily, SetFamily, Generator>(
                        self.clone(),
                        dd_umpire,
                        purifier,
                        output_adjacency,
                        timing,
                        coeff_mode,
                        dd(howzat_output_adjacency),
                        matrix,
                        time_matrix,
                        start_total,
                    )
                    .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => {
                        run_howzat_exact_dd::<ListFamily, AdjacencyList, Generator>(
                            self.clone(),
                            dd_umpire,
                            purifier,
                            output_adjacency,
                            timing,
                            coeff_mode,
                            dd(AdjacencyOutput::Off),
                            matrix,
                            time_matrix,
                            start_total,
                        )
                        .map(BackendRunAny::Sparse)
                    }
                    RequestedAdjacency::Default => unreachable!("choose_adjacency never returns Default"),
                }
            }
            Representation::HomogeneousGenerators => {
                self.ensure_generator_matrix_backend()?;
                ensure!(
                    cols > 1,
                    "generator matrix must have at least 2 columns"
                );
                ensure!(
                    rows
                        .checked_mul(cols)
                        .is_some_and(|len| len == data.len()),
                    "expected {rows}x{cols} generator entries but got {}",
                    data.len()
                );
                let start_matrix = Instant::now();
                let matrix =
                    LpMatrixBuilder::<calculo::num::RugRat, Generator>::from_flat(rows, cols, data)
                        .build();
                let time_matrix = start_matrix.elapsed();
                let dim = cols - 1;
                match self.choose_adjacency(rows, dim, output_adjacency) {
                    RequestedAdjacency::Dense => run_howzat_exact_dd::<SetFamily, SetFamily, Generator>(
                        self.clone(),
                        dd_umpire,
                        purifier,
                        output_adjacency,
                        timing,
                        coeff_mode,
                        dd(howzat_output_adjacency),
                        matrix,
                        time_matrix,
                        start_total,
                    )
                    .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => {
                        run_howzat_exact_dd::<ListFamily, AdjacencyList, Generator>(
                            self.clone(),
                            dd_umpire,
                            purifier,
                            output_adjacency,
                            timing,
                            coeff_mode,
                            dd(AdjacencyOutput::Off),
                            matrix,
                            time_matrix,
                            start_total,
                        )
                        .map(BackendRunAny::Sparse)
                    }
                    RequestedAdjacency::Default => unreachable!("choose_adjacency never returns Default"),
                }
            }
            Representation::Inequality => {
                ensure!(
                    cols > 1,
                    "inequality matrix must have at least 2 columns"
                );
                ensure!(
                    rows
                        .checked_mul(cols)
                        .is_some_and(|len| len == data.len()),
                    "expected {rows}x{cols} coeffs but got {}",
                    data.len()
                );
                let start_matrix = Instant::now();
                let matrix =
                    LpMatrixBuilder::<calculo::num::RugRat, Inequality>::from_flat(rows, cols, data)
                        .build();
                let time_matrix = start_matrix.elapsed();
                let dim = cols - 1;
                match self.choose_adjacency(rows, dim, output_adjacency) {
                    RequestedAdjacency::Dense => run_howzat_exact_dd::<SetFamily, SetFamily, Inequality>(
                        self.clone(),
                        dd_umpire,
                        purifier,
                        output_adjacency,
                        timing,
                        coeff_mode,
                        dd(howzat_output_adjacency),
                        matrix,
                        time_matrix,
                        start_total,
                    )
                    .map(BackendRunAny::Dense),
                    RequestedAdjacency::Sparse => {
                        run_howzat_exact_dd::<ListFamily, AdjacencyList, Inequality>(
                            self.clone(),
                            dd_umpire,
                            purifier,
                            output_adjacency,
                            timing,
                            coeff_mode,
                            dd(AdjacencyOutput::Off),
                            matrix,
                            time_matrix,
                            start_total,
                        )
                        .map(BackendRunAny::Sparse)
                    }
                    RequestedAdjacency::Default => unreachable!("choose_adjacency never returns Default"),
                }
            }
        }
    }

    fn ensure_exact_backend(&self) -> Result<(), anyhow::Error> {
        let is_exact = match &self.0 {
            BackendSpec::CddlibGmpRational | BackendSpec::CddlibHlblGmpRational => true,
            BackendSpec::HowzatDd { pipeline, .. } => pipeline.is_exact(),
            BackendSpec::LrslibHlblGmpInt | BackendSpec::PplHlblGmpInt => true,
            _ => false,
        };

        ensure!(
            is_exact,
            "backend '{self}' is not exact; choose an exact backend spec or call solve()/solve_row_major() instead"
        );
        Ok(())
    }

    fn ensure_generator_matrix_backend(&self) -> Result<(), anyhow::Error> {
        ensure!(
            matches!(
                &self.0,
                BackendSpec::HowzatDd { .. } | BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu
            ),
            "backend '{self}' does not support HomogeneousGenerators input; use a howzat backend or pass Euclidean vertices instead"
        );
        Ok(())
    }

    fn solve_exact_dense<V: VerticesI64 + HowzatVertices + ?Sized>(
        &self,
        vertices: &V,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<SetFamily, SetFamily>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::Exact
        } else {
            CoefficientMode::Off
        };
        let howzat_output_adjacency = if output_adjacency {
            config.howzat_output_adjacency
        } else {
            AdjacencyOutput::Off
        };

        let run = match &self.0 {
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend::<V, SetFamily, SetFamily>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    howzat_output_adjacency,
                    coeff_mode,
                    vertices,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::LrslibHlblGmpInt => {
                lrslib::run_lrslib_hlbl_backend_i64::<V, SetFamily, SetFamily>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::PplHlblGmpInt => ppl::run_ppl_hlbl_backend_i64::<V, SetFamily, SetFamily>(
                self.clone(),
                vertices,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibGmpRational | BackendSpec::CddlibHlblGmpRational => {
                Err(anyhow!("{self} does not support dense adjacency"))
            }
            _ => Err(anyhow!(
                "internal: solve_exact_dense called with non-exact backend"
            )),
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => Err(err),
        }
    }

    fn solve_exact_dense_inequalities<I: InequalitiesI64 + HowzatInequalities + ?Sized>(
        &self,
        inequalities: &I,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<SetFamily, SetFamily>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::Exact
        } else {
            CoefficientMode::Off
        };
        let howzat_output_adjacency = if output_adjacency {
            config.howzat_output_adjacency
        } else {
            AdjacencyOutput::Off
        };

        let run = match &self.0 {
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend_from_inequalities::<I, SetFamily, SetFamily>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    howzat_output_adjacency,
                    coeff_mode,
                    inequalities,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu => {
                howzat_lrs::run_howzat_lrs_backend_from_inequalities::<I, SetFamily, SetFamily>(
                    self.clone(),
                    inequalities,
                    output_incidence,
                    output_adjacency,
                    howzat_output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            _ => Err(anyhow!("{self} does not support inequality input")),
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => Err(err),
        }
    }

    fn solve_exact_sparse<V: VerticesI64 + HowzatVertices + ?Sized>(
        &self,
        vertices: &V,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let start_run = Instant::now();
        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::Exact
        } else {
            CoefficientMode::Off
        };

        let run = match &self.0 {
            BackendSpec::CddlibGmpRational => {
                cddlib::run_cddlib_backend_i64::<cddlib_rs::CddRational, V>(
                    self.clone(),
                    vertices,
                    cddlib_rs::NumberType::Rational,
                    false,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::CddlibHlblGmpRational => {
                cddlib::run_cddlib_backend_i64::<cddlib_rs::CddRational, V>(
                    self.clone(),
                    vertices,
                    cddlib_rs::NumberType::Rational,
                    true,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend::<V, ListFamily, AdjacencyList>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    AdjacencyOutput::Off,
                    coeff_mode,
                    vertices,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::LrslibHlblGmpInt => {
                lrslib::run_lrslib_hlbl_backend_i64::<V, ListFamily, AdjacencyList>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::PplHlblGmpInt => {
                ppl::run_ppl_hlbl_backend_i64::<V, ListFamily, AdjacencyList>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            _ => Err(anyhow!(
                "internal: solve_exact_sparse called with non-exact backend"
            )),
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => {
                if matches!(
                    self.0,
                    BackendSpec::CddlibF64
                        | BackendSpec::CddlibGmpFloat
                        | BackendSpec::CddlibGmpRational
                        | BackendSpec::CddlibHlblF64
                        | BackendSpec::CddlibHlblGmpFloat
                        | BackendSpec::CddlibHlblGmpRational
                ) && cddlib::is_cddlib_error_code(
                    &err,
                    cddlib_rs::CddErrorCode::NumericallyInconsistent,
                )
                {
                    Ok(cddlib::backend_error_run_sparse(
                        self.clone(),
                        vertices.dim(),
                        vertices.vertex_count(),
                        start_run.elapsed(),
                        err.to_string(),
                    ))
                } else {
                    Err(err)
                }
            }
        }
    }

    fn solve_exact_sparse_inequalities<I: InequalitiesI64 + HowzatInequalities + ?Sized>(
        &self,
        inequalities: &I,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        self.ensure_exact_backend()?;

        let start_run = Instant::now();
        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::Exact
        } else {
            CoefficientMode::Off
        };

        let run = match &self.0 {
            BackendSpec::CddlibGmpRational => cddlib::run_cddlib_backend_inequalities_i64::<
                cddlib_rs::CddRational,
                I,
            >(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Rational,
                false,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibHlblGmpRational => cddlib::run_cddlib_backend_inequalities_i64::<
                cddlib_rs::CddRational,
                I,
            >(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Rational,
                true,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend_from_inequalities::<I, ListFamily, AdjacencyList>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    AdjacencyOutput::Off,
                    coeff_mode,
                    inequalities,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu => {
                howzat_lrs::run_howzat_lrs_backend_from_inequalities::<I, ListFamily, AdjacencyList>(
                    self.clone(),
                    inequalities,
                    output_incidence,
                    output_adjacency,
                    AdjacencyOutput::Off,
                    coeff_mode,
                    timing,
                )
            }
            _ => Err(anyhow!("{self} does not support inequality input")),
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => {
                if matches!(
                    self.0,
                    BackendSpec::CddlibF64
                        | BackendSpec::CddlibGmpFloat
                        | BackendSpec::CddlibGmpRational
                        | BackendSpec::CddlibHlblF64
                        | BackendSpec::CddlibHlblGmpFloat
                        | BackendSpec::CddlibHlblGmpRational
                ) && cddlib::is_cddlib_error_code(
                    &err,
                    cddlib_rs::CddErrorCode::NumericallyInconsistent,
                )
                {
                    Ok(cddlib::backend_error_run_sparse(
                        self.clone(),
                        inequalities.dim(),
                        inequalities.facet_count(),
                        start_run.elapsed(),
                        err.to_string(),
                    ))
                } else {
                    Err(err)
                }
            }
        }
    }

    fn solve_dense<V: VerticesF64 + HowzatVertices + ?Sized>(
        &self,
        vertices: &V,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<SetFamily, SetFamily>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::F64
        } else {
            CoefficientMode::Off
        };
        let howzat_output_adjacency = if output_adjacency {
            config.howzat_output_adjacency
        } else {
            AdjacencyOutput::Off
        };

        let run = match &self.0 {
            BackendSpec::CddlibF64
            | BackendSpec::CddlibGmpFloat
            | BackendSpec::CddlibGmpRational
            | BackendSpec::CddlibHlblF64
            | BackendSpec::CddlibHlblGmpFloat
            | BackendSpec::CddlibHlblGmpRational => {
                Err(anyhow!("{self} does not support dense adjacency"))
            }
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend::<V, SetFamily, SetFamily>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    howzat_output_adjacency,
                    coeff_mode,
                    vertices,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu => {
                howzat_lrs::run_howzat_lrs_backend::<V, SetFamily, SetFamily>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    howzat_output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::LrslibHlblGmpInt => {
                lrslib::run_lrslib_hlbl_backend::<V, SetFamily, SetFamily>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::PplHlblGmpInt => ppl::run_ppl_hlbl_backend::<V, SetFamily, SetFamily>(
                self.clone(),
                vertices,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => Err(err),
        }
    }

    fn solve_dense_inequalities<I: InequalitiesF64 + HowzatInequalities + ?Sized>(
        &self,
        inequalities: &I,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<SetFamily, SetFamily>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::F64
        } else {
            CoefficientMode::Off
        };
        let howzat_output_adjacency = if output_adjacency {
            config.howzat_output_adjacency
        } else {
            AdjacencyOutput::Off
        };

        let run = match &self.0 {
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend_from_inequalities::<I, SetFamily, SetFamily>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    howzat_output_adjacency,
                    coeff_mode,
                    inequalities,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu => {
                howzat_lrs::run_howzat_lrs_backend_from_inequalities::<I, SetFamily, SetFamily>(
                    self.clone(),
                    inequalities,
                    output_incidence,
                    output_adjacency,
                    howzat_output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            _ => Err(anyhow!("{self} does not support inequality input")),
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => Err(err),
        }
    }

    fn solve_sparse<V: VerticesF64 + HowzatVertices + ?Sized>(
        &self,
        vertices: &V,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let start_run = Instant::now();
        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::F64
        } else {
            CoefficientMode::Off
        };

        let run = match &self.0 {
            BackendSpec::CddlibF64 => cddlib::run_cddlib_backend::<f64, V>(
                self.clone(),
                vertices,
                cddlib_rs::NumberType::Real,
                false,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibGmpFloat => cddlib::run_cddlib_backend::<cddlib_rs::CddFloat, V>(
                self.clone(),
                vertices,
                cddlib_rs::NumberType::Real,
                false,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibGmpRational => {
                cddlib::run_cddlib_backend::<cddlib_rs::CddRational, V>(
                    self.clone(),
                    vertices,
                    cddlib_rs::NumberType::Rational,
                    false,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::CddlibHlblF64 => cddlib::run_cddlib_backend::<f64, V>(
                self.clone(),
                vertices,
                cddlib_rs::NumberType::Real,
                true,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibHlblGmpFloat => {
                cddlib::run_cddlib_backend::<cddlib_rs::CddFloat, V>(
                    self.clone(),
                    vertices,
                    cddlib_rs::NumberType::Real,
                    true,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::CddlibHlblGmpRational => {
                cddlib::run_cddlib_backend::<cddlib_rs::CddRational, V>(
                    self.clone(),
                    vertices,
                    cddlib_rs::NumberType::Rational,
                    true,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend::<V, ListFamily, AdjacencyList>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    AdjacencyOutput::Off,
                    coeff_mode,
                    vertices,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu => {
                howzat_lrs::run_howzat_lrs_backend::<V, ListFamily, AdjacencyList>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    AdjacencyOutput::Off,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::LrslibHlblGmpInt => {
                lrslib::run_lrslib_hlbl_backend::<V, ListFamily, AdjacencyList>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
            BackendSpec::PplHlblGmpInt => {
                ppl::run_ppl_hlbl_backend::<V, ListFamily, AdjacencyList>(
                    self.clone(),
                    vertices,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                )
            }
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => {
                if matches!(
                    self.0,
                    BackendSpec::CddlibF64
                        | BackendSpec::CddlibGmpFloat
                        | BackendSpec::CddlibGmpRational
                        | BackendSpec::CddlibHlblF64
                        | BackendSpec::CddlibHlblGmpFloat
                        | BackendSpec::CddlibHlblGmpRational
                ) && cddlib::is_cddlib_error_code(
                    &err,
                    cddlib_rs::CddErrorCode::NumericallyInconsistent,
                )
                {
                    Ok(cddlib::backend_error_run_sparse(
                        self.clone(),
                        vertices.dim(),
                        vertices.vertex_count(),
                        start_run.elapsed(),
                        err.to_string(),
                    ))
                } else {
                    Err(err)
                }
            }
        }
    }

    fn solve_sparse_inequalities<I: InequalitiesF64 + HowzatInequalities + ?Sized>(
        &self,
        inequalities: &I,
        config: &BackendRunConfig,
    ) -> Result<BackendRun<ListFamily, AdjacencyList>, anyhow::Error> {
        ensure!(
            !config.output_adjacency || config.output_incidence,
            "output_adjacency requires output_incidence"
        );

        let start_run = Instant::now();
        let timing = config.timing_detail;
        let output_incidence = config.output_incidence;
        let output_adjacency = config.output_adjacency;
        let coeff_mode = if config.output_coefficients {
            CoefficientMode::F64
        } else {
            CoefficientMode::Off
        };

        let run = match &self.0 {
            BackendSpec::CddlibF64 => cddlib::run_cddlib_backend_inequalities::<f64, I>(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Real,
                false,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibGmpFloat => cddlib::run_cddlib_backend_inequalities::<
                cddlib_rs::CddFloat,
                I,
            >(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Real,
                false,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibGmpRational => cddlib::run_cddlib_backend_inequalities::<
                cddlib_rs::CddRational,
                I,
            >(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Rational,
                false,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibHlblF64 => cddlib::run_cddlib_backend_inequalities::<f64, I>(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Real,
                true,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibHlblGmpFloat => cddlib::run_cddlib_backend_inequalities::<
                cddlib_rs::CddFloat,
                I,
            >(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Real,
                true,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::CddlibHlblGmpRational => cddlib::run_cddlib_backend_inequalities::<
                cddlib_rs::CddRational,
                I,
            >(
                self.clone(),
                inequalities,
                cddlib_rs::NumberType::Rational,
                true,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
            ),
            BackendSpec::HowzatDd {
                umpire,
                purifier,
                pipeline,
            } => {
                howzat_dd::run_howzat_dd_backend_from_inequalities::<I, ListFamily, AdjacencyList>(
                    self.clone(),
                    *umpire,
                    *purifier,
                    pipeline,
                    output_incidence,
                    output_adjacency,
                    AdjacencyOutput::Off,
                    coeff_mode,
                    inequalities,
                    &config.howzat_options,
                    timing,
                )
            }
            BackendSpec::HowzatLrsRug | BackendSpec::HowzatLrsDashu => {
                howzat_lrs::run_howzat_lrs_backend_from_inequalities::<I, ListFamily, AdjacencyList>(
                    self.clone(),
                    inequalities,
                    output_incidence,
                    output_adjacency,
                    AdjacencyOutput::Off,
                    coeff_mode,
                    timing,
                )
            }
            _ => Err(anyhow!("{self} does not support inequality input")),
        };

        match run {
            Ok(run) => Ok(run),
            Err(err) => {
                if matches!(
                    self.0,
                    BackendSpec::CddlibF64
                        | BackendSpec::CddlibGmpFloat
                        | BackendSpec::CddlibGmpRational
                        | BackendSpec::CddlibHlblF64
                        | BackendSpec::CddlibHlblGmpFloat
                        | BackendSpec::CddlibHlblGmpRational
                ) && cddlib::is_cddlib_error_code(
                    &err,
                    cddlib_rs::CddErrorCode::NumericallyInconsistent,
                )
                {
                    Ok(cddlib::backend_error_run_sparse(
                        self.clone(),
                        inequalities.dim(),
                        inequalities.facet_count(),
                        start_run.elapsed(),
                        err.to_string(),
                    ))
                } else {
                    Err(err)
                }
            }
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_kind(f)?;
        let mut opened = false;

        if let BackendSpec::HowzatDd {
            purifier: Some(purifier),
            ..
        } = &self.0
        {
            f.write_str("[")?;
            opened = true;
            write!(f, "purify[{}]", purifier.canonical_token())?;
        }

        if let Some(token) = self.1.token() {
            if !opened {
                f.write_str("[")?;
                opened = true;
            } else {
                f.write_str(",")?;
            }
            write!(f, "adj[{token}]")?;
        }

        if opened {
            f.write_str("]")?;
        }

        f.write_str(":")?;
        self.0.fmt_num(f)?;
        Ok(())
    }
}

fn split_backend_bracket_group(raw: &str) -> Result<Option<(&str, &str)>, String> {
    let token = raw.trim_end();
    if !token.ends_with(']') {
        return Ok(None);
    }

    let mut depth = 0usize;
    for (idx, ch) in token.char_indices().rev() {
        match ch {
            ']' => depth += 1,
            '[' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let base = token.get(..idx).unwrap_or("");
                    let inner = token.get(idx + 1..token.len() - 1).unwrap_or("");
                    return Ok(Some((base, inner)));
                }
            }
            _ => {}
        }
    }

    Err(format!("backend spec '{raw}' has an unmatched ']'"))
}

fn split_backend_option_token(token: &str) -> Result<(&str, &str), String> {
    let token = token.trim();
    if token.is_empty() {
        return Err("backend option cannot be empty".to_string());
    }

    let Some(open_idx) = token.find('[') else {
        return Err(format!(
            "backend option '{token}' must be written as name[value], e.g. adj[sparse]"
        ));
    };
    let name = token.get(..open_idx).unwrap_or("").trim();
    if name.is_empty() {
        return Err(format!("backend option '{token}' is missing a name"));
    }

    let after_open = token.get(open_idx + 1..).unwrap_or("");
    let mut depth = 1usize;
    let mut close_rel = None;
    for (offset, ch) in after_open.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    close_rel = Some(offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(close_rel) = close_rel else {
        return Err(format!("backend option '{token}' has an unmatched '['"));
    };

    let arg = after_open.get(..close_rel).unwrap_or("").trim();
    let rest = after_open.get(close_rel + 1..).unwrap_or("").trim();
    if !rest.is_empty() {
        return Err(format!(
            "backend option '{token}' contains trailing characters after the closing ']'"
        ));
    }

    Ok((name, arg))
}

fn parse_backend_options(
    raw: &str,
) -> Result<(Option<HowzatDdPurifierSpec>, RequestedAdjacency), String> {
    let mut purifier = None;
    let mut adjacency = RequestedAdjacency::Default;

    let mut depth = 0usize;
    let mut start = 0usize;
    let mut tokens = Vec::new();
    for (idx, ch) in raw.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                tokens.push(raw.get(start..idx).unwrap_or("").trim());
                start = idx + 1;
            }
            _ => {}
        }
    }
    tokens.push(raw.get(start..).unwrap_or("").trim());

    for token in tokens {
        if token.is_empty() {
            return Err(format!(
                "backend options '{raw}' contains an empty entry (check comma placement)"
            ));
        }
        let (name, arg) = split_backend_option_token(token)?;

        if name.eq_ignore_ascii_case("purify") {
            if purifier.is_some() {
                return Err(format!(
                    "backend options '{raw}' contains multiple purify[...] entries"
                ));
            }
            purifier = Some(parse_howzat_dd_purifier(arg)?);
        } else if name.eq_ignore_ascii_case("adj") {
            if adjacency != RequestedAdjacency::Default {
                return Err(format!(
                    "backend options '{raw}' contains multiple adj[...] entries"
                ));
            }
            if arg.eq_ignore_ascii_case("dense") {
                adjacency = RequestedAdjacency::Dense;
            } else if arg.eq_ignore_ascii_case("sparse") {
                adjacency = RequestedAdjacency::Sparse;
            } else {
                return Err(format!(
                    "backend option adj[{arg}] is invalid; expected adj[dense] or adj[sparse]"
                ));
            }
        } else {
            return Err(format!(
                "backend option '{name}' is unknown; supported options: purify[...], adj[dense|sparse]"
            ));
        }
    }

    Ok((purifier, adjacency))
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let raw = value.trim();
        if raw.is_empty() {
            return Err("backend spec cannot be empty".to_string());
        }

        let (kind_part, num_part) = raw
            .split_once(':')
            .map(|(k, n)| (k.trim(), Some(n.trim())))
            .unwrap_or((raw.trim(), None));

        let mut kind_part = kind_part;
        let num_part = num_part;
        let mut purifier = None;
        let mut adjacency = RequestedAdjacency::Default;

        if let Some((candidate_kind, inner)) = split_backend_bracket_group(kind_part)? {
            let inner = inner.trim();
            if inner.eq_ignore_ascii_case("dense") || inner.eq_ignore_ascii_case("sparse") {
                return Err(format!(
                    "backend spec '{value}' uses legacy adjacency syntax '[{inner}]'; use '[adj[{inner}]]' instead"
                ));
            }

            let candidate_kind = candidate_kind.trim_end();
            if candidate_kind.is_empty() {
                return Err("backend spec cannot be empty".to_string());
            }

            (purifier, adjacency) = parse_backend_options(inner)?;
            kind_part = candidate_kind;
        }

        if let Some(num) = num_part {
            let lowered = num.to_ascii_lowercase();
            if lowered.contains("purify[") || lowered.contains("adj[") {
                return Err(format!(
                    "backend spec '{value}' contains backend options after ':'; \
write backend options immediately after the backend kind, before ':' (e.g. 'howzat-dd[purify[snap]]:f64')"
                ));
            }
            if !kind_part.to_ascii_lowercase().starts_with("howzat-dd") && num.contains('[') {
                return Err(format!(
                    "backend spec '{value}' contains an unexpected '[' after ':'; \
backend options must appear after the backend kind, before ':'"
                ));
            }
        }

        let spec_string = if let Some(num) = num_part {
            format!("{kind_part}:{num}")
        } else {
            kind_part.to_string()
        };

        let mut spec: BackendSpec = spec_string.parse()?;

        if let Some(purifier) = purifier {
            match &mut spec {
                BackendSpec::HowzatDd {
                    purifier: spec_purifier,
                    ..
                } => {
                    *spec_purifier = Some(purifier);
                }
                _ => {
                    return Err(format!(
                        "backend spec '{value}' does not support purify[...] (only howzat-dd does)"
                    ));
                }
            }
        }

        match adjacency {
            RequestedAdjacency::Default => {}
            RequestedAdjacency::Dense => {
                if !spec.supports_dense_adjacency() {
                    return Err(format!(
                        "backend spec '{value}' does not support adj[dense] adjacency"
                    ));
                }
            }
            RequestedAdjacency::Sparse => {
                if !spec.supports_sparse_adjacency() {
                    return Err(format!(
                        "backend spec '{value}' does not support adj[sparse] adjacency"
                    ));
                }
            }
        }

        Ok(Self(spec, adjacency))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendArg {
    pub spec: Backend,
    pub authoritative: bool,
    pub perf_baseline: bool,
}

impl std::str::FromStr for BackendArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let raw = value.trim();
        if raw.is_empty() {
            return Err("backend spec cannot be empty".to_string());
        }

        let mut authoritative = false;
        let mut perf_baseline = false;
        let mut rest = raw;

        loop {
            let Some(prefix) = rest.chars().next() else {
                break;
            };
            match prefix {
                '^' => {
                    if authoritative {
                        return Err(format!(
                            "backend spec '{value}' contains multiple '^' prefixes"
                        ));
                    }
                    authoritative = true;
                    rest = &rest['^'.len_utf8()..];
                }
                '%' => {
                    if perf_baseline {
                        return Err(format!(
                            "backend spec '{value}' contains multiple '%' prefixes"
                        ));
                    }
                    perf_baseline = true;
                    rest = &rest['%'.len_utf8()..];
                }
                _ => break,
            }
        }

        let spec: Backend = rest.trim().parse()?;
        Ok(Self {
            spec,
            authoritative,
            perf_baseline,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BackendRunConfig {
    pub howzat_options: ConeOptions,
    pub output_incidence: bool,
    pub output_adjacency: bool,
    pub output_coefficients: bool,
    pub howzat_output_adjacency: AdjacencyOutput,
    pub timing_detail: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum BackendOutputLevel {
    Representation,
    Incidence,
    Adjacency,
}

impl BackendOutputLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Representation => "representation",
            Self::Incidence => "incidence",
            Self::Adjacency => "adjacency",
        }
    }
}

impl Default for BackendRunConfig {
    fn default() -> Self {
        Self {
            howzat_options: ConeOptions::default(),
            output_incidence: true,
            output_adjacency: true,
            output_coefficients: false,
            howzat_output_adjacency: AdjacencyOutput::List,
            timing_detail: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CoefficientMode {
    Off,
    F64,
    Exact,
}

#[derive(Debug, Clone)]
pub struct RowMajorMatrix<N> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<N>,
}

impl<N> RowMajorMatrix<N> {
    pub fn row(&self, row: usize) -> &[N] {
        assert!(row < self.rows);
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RowMajorMatrixView<'a, N> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [N],
}

impl<'a, N> RowMajorMatrixView<'a, N> {
    pub fn row(&self, row: usize) -> &'a [N] {
        assert!(row < self.rows);
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }
}

#[derive(Debug, Clone)]
pub enum CoefficientMatrix {
    F64(RowMajorMatrix<f64>),
    RugFloat128(RowMajorMatrix<calculo::num::RugFloat<128>>),
    RugFloat256(RowMajorMatrix<calculo::num::RugFloat<256>>),
    RugFloat512(RowMajorMatrix<calculo::num::RugFloat<512>>),
    DashuFloat128(RowMajorMatrix<calculo::num::DashuFloat<128>>),
    DashuFloat256(RowMajorMatrix<calculo::num::DashuFloat<256>>),
    DashuFloat512(RowMajorMatrix<calculo::num::DashuFloat<512>>),
    RugRat(RowMajorMatrix<calculo::num::RugRat>),
    DashuRat(RowMajorMatrix<calculo::num::DashuRat>),
}

mod coefficient_scalar_sealed {
    pub trait Sealed {}
}

pub trait CoefficientScalar: calculo::num::Num + coefficient_scalar_sealed::Sealed + 'static {
    fn wrap(matrix: RowMajorMatrix<Self>) -> CoefficientMatrix;

    fn view(matrix: &CoefficientMatrix) -> Option<RowMajorMatrixView<'_, Self>>;

    fn coerce_from_matrix(
        matrix: &CoefficientMatrix,
    ) -> Result<RowMajorMatrix<Self>, calculo::num::ConversionError>;
}

fn coerce_matrix_via_f64<N: calculo::num::Num>(
    matrix: &CoefficientMatrix,
) -> Result<RowMajorMatrix<N>, calculo::num::ConversionError> {
    #[inline(always)]
    fn push<N: calculo::num::Num>(
        out: &mut Vec<N>,
        value: f64,
    ) -> Result<(), calculo::num::ConversionError> {
        if !value.is_finite() {
            return Err(calculo::num::ConversionError);
        }
        out.push(N::try_from_f64(value).ok_or(calculo::num::ConversionError)?);
        Ok(())
    }

    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut out = Vec::with_capacity(rows.saturating_mul(cols));
    match matrix {
        CoefficientMatrix::F64(m) => {
            for &v in &m.data {
                push(&mut out, v)?;
            }
        }
        CoefficientMatrix::RugFloat128(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
        CoefficientMatrix::RugFloat256(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
        CoefficientMatrix::RugFloat512(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
        CoefficientMatrix::DashuFloat128(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
        CoefficientMatrix::DashuFloat256(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
        CoefficientMatrix::DashuFloat512(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
        CoefficientMatrix::RugRat(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
        CoefficientMatrix::DashuRat(m) => {
            for v in &m.data {
                push(&mut out, v.to_f64())?;
            }
        }
    }

    Ok(RowMajorMatrix { rows, cols, data: out })
}

fn dashu_ibig_to_rug_integer(
    value: &<calculo::num::DashuRat as calculo::num::Rat>::Int,
) -> rug::Integer {
    use rug::integer::Order;

    let (sign, words) = value.as_sign_words();
    let mut out = rug::Integer::from_digits(words, Order::Lsf);
    let is_negative: bool = sign.into();
    if is_negative {
        out = -out;
    }
    out
}

fn coerce_matrix_to_rug_rat(
    matrix: &CoefficientMatrix,
) -> Result<RowMajorMatrix<calculo::num::RugRat>, calculo::num::ConversionError> {
    use calculo::num::CoerceFrom as _;
    use calculo::num::Rat as _;

    fn from_dashu_rat(
        value: &calculo::num::DashuRat,
    ) -> Result<calculo::num::RugRat, calculo::num::ConversionError> {
        let (numer, denom) = value.clone().into_parts();
        let numer = dashu_ibig_to_rug_integer(&numer);
        let denom = dashu_ibig_to_rug_integer(&denom);
        if denom == 0 {
            return Err(calculo::num::ConversionError);
        }
        Ok(calculo::num::RugRat(rug::Rational::from((numer, denom))))
    }

    let rows = matrix.rows();
    let cols = matrix.cols();
    let mut out = Vec::with_capacity(rows.saturating_mul(cols));

    match matrix {
        CoefficientMatrix::F64(m) => {
            for &v in &m.data {
                if !v.is_finite() {
                    return Err(calculo::num::ConversionError);
                }
                out.push(
                    calculo::num::RugRat::try_from_f64(v).ok_or(calculo::num::ConversionError)?,
                );
            }
        }
        CoefficientMatrix::RugFloat128(m) => {
            for v in &m.data {
                let rat = v.0.to_rational().ok_or(calculo::num::ConversionError)?;
                out.push(calculo::num::RugRat(rat));
            }
        }
        CoefficientMatrix::RugFloat256(m) => {
            for v in &m.data {
                let rat = v.0.to_rational().ok_or(calculo::num::ConversionError)?;
                out.push(calculo::num::RugRat(rat));
            }
        }
        CoefficientMatrix::RugFloat512(m) => {
            for v in &m.data {
                let rat = v.0.to_rational().ok_or(calculo::num::ConversionError)?;
                out.push(calculo::num::RugRat(rat));
            }
        }
        CoefficientMatrix::DashuFloat128(m) => {
            for v in &m.data {
                let dashu_rat = calculo::num::DashuRat::coerce_from(v)?;
                out.push(from_dashu_rat(&dashu_rat)?);
            }
        }
        CoefficientMatrix::DashuFloat256(m) => {
            for v in &m.data {
                let dashu_rat = calculo::num::DashuRat::coerce_from(v)?;
                out.push(from_dashu_rat(&dashu_rat)?);
            }
        }
        CoefficientMatrix::DashuFloat512(m) => {
            for v in &m.data {
                let dashu_rat = calculo::num::DashuRat::coerce_from(v)?;
                out.push(from_dashu_rat(&dashu_rat)?);
            }
        }
        CoefficientMatrix::RugRat(m) => out.extend(m.data.iter().cloned()),
        CoefficientMatrix::DashuRat(m) => {
            for v in &m.data {
                out.push(from_dashu_rat(v)?);
            }
        }
    }

    Ok(RowMajorMatrix { rows, cols, data: out })
}

macro_rules! impl_coeff_scalar_via_f64 {
    ($ty:ty, $variant:ident) => {
        impl coefficient_scalar_sealed::Sealed for $ty {}

        impl CoefficientScalar for $ty {
            #[inline(always)]
            fn wrap(matrix: RowMajorMatrix<Self>) -> CoefficientMatrix {
                CoefficientMatrix::$variant(matrix)
            }

            #[inline(always)]
            fn view(matrix: &CoefficientMatrix) -> Option<RowMajorMatrixView<'_, Self>> {
                let CoefficientMatrix::$variant(m) = matrix else {
                    return None;
                };
                Some(RowMajorMatrixView {
                    rows: m.rows,
                    cols: m.cols,
                    data: &m.data,
                })
            }

            #[inline(always)]
            fn coerce_from_matrix(
                matrix: &CoefficientMatrix,
            ) -> Result<RowMajorMatrix<Self>, calculo::num::ConversionError> {
                if let Some(view) = Self::view(matrix) {
                    return Ok(RowMajorMatrix {
                        rows: view.rows,
                        cols: view.cols,
                        data: view.data.to_vec(),
                    });
                }
                coerce_matrix_via_f64(matrix)
            }
        }
    };
}

impl_coeff_scalar_via_f64!(f64, F64);
impl_coeff_scalar_via_f64!(calculo::num::RugFloat<128>, RugFloat128);
impl_coeff_scalar_via_f64!(calculo::num::RugFloat<256>, RugFloat256);
impl_coeff_scalar_via_f64!(calculo::num::RugFloat<512>, RugFloat512);
impl_coeff_scalar_via_f64!(calculo::num::DashuFloat<128>, DashuFloat128);
impl_coeff_scalar_via_f64!(calculo::num::DashuFloat<256>, DashuFloat256);
impl_coeff_scalar_via_f64!(calculo::num::DashuFloat<512>, DashuFloat512);
impl_coeff_scalar_via_f64!(calculo::num::DashuRat, DashuRat);

impl coefficient_scalar_sealed::Sealed for calculo::num::RugRat {}

impl CoefficientScalar for calculo::num::RugRat {
    #[inline(always)]
    fn wrap(matrix: RowMajorMatrix<Self>) -> CoefficientMatrix {
        CoefficientMatrix::RugRat(matrix)
    }

    #[inline(always)]
    fn view(matrix: &CoefficientMatrix) -> Option<RowMajorMatrixView<'_, Self>> {
        let CoefficientMatrix::RugRat(m) = matrix else {
            return None;
        };
        Some(RowMajorMatrixView {
            rows: m.rows,
            cols: m.cols,
            data: &m.data,
        })
    }

    #[inline(always)]
    fn coerce_from_matrix(
        matrix: &CoefficientMatrix,
    ) -> Result<RowMajorMatrix<Self>, calculo::num::ConversionError> {
        coerce_matrix_to_rug_rat(matrix)
    }
}

impl CoefficientMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::F64(m) => m.rows,
            Self::RugFloat128(m) => m.rows,
            Self::RugFloat256(m) => m.rows,
            Self::RugFloat512(m) => m.rows,
            Self::DashuFloat128(m) => m.rows,
            Self::DashuFloat256(m) => m.rows,
            Self::DashuFloat512(m) => m.rows,
            Self::RugRat(m) => m.rows,
            Self::DashuRat(m) => m.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F64(m) => m.cols,
            Self::RugFloat128(m) => m.cols,
            Self::RugFloat256(m) => m.cols,
            Self::RugFloat512(m) => m.cols,
            Self::DashuFloat128(m) => m.cols,
            Self::DashuFloat256(m) => m.cols,
            Self::DashuFloat512(m) => m.cols,
            Self::RugRat(m) => m.cols,
            Self::DashuRat(m) => m.cols,
        }
    }

    pub fn cast<N: CoefficientScalar>(&self) -> RowMajorMatrixView<'_, N> {
        N::view(self).unwrap_or_else(|| {
            panic!(
                "CoefficientMatrix::cast<{}> called on {self:?}",
                std::any::type_name::<N>()
            )
        })
    }

    pub fn coerce<N: CoefficientScalar>(
        &self,
    ) -> Result<RowMajorMatrix<N>, calculo::num::ConversionError> {
        N::coerce_from_matrix(self)
    }

    pub fn stringify(&self) -> Result<RowMajorMatrix<String>, calculo::num::ConversionError> {
        let m = self.coerce::<calculo::num::RugRat>()?;
        Ok(RowMajorMatrix {
            rows: m.rows,
            cols: m.cols,
            data: m.data.iter().map(ToString::to_string).collect(),
        })
    }

    pub fn from_num<N: CoefficientScalar>(rows: usize, cols: usize, data: Vec<N>) -> Self {
        N::wrap(RowMajorMatrix { rows, cols, data })
    }
}

impl serde::Serialize for CoefficientMatrix {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;

        fn serialize_rows<S>(
            rows: usize,
            cols: usize,
            data: &[f64],
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut seq = serializer.serialize_seq(Some(rows))?;
            for row in 0..rows {
                let start = row * cols;
                seq.serialize_element(&data[start..start + cols])?;
            }
            seq.end()
        }

        match self {
            Self::F64(m) => serialize_rows(m.rows, m.cols, &m.data, serializer),
            _ => {
                let m = self.coerce::<f64>().map_err(serde::ser::Error::custom)?;
                serialize_rows(m.rows, m.cols, &m.data, serializer)
            }
        }
    }
}

impl<'de> serde::Deserialize<'de> for CoefficientMatrix {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let rows: Vec<Vec<f64>> = Vec::deserialize(deserializer)?;
        let row_count = rows.len();
        let col_count = rows.first().map_or(0, Vec::len);

        if rows.iter().any(|row| row.len() != col_count) {
            return Err(serde::de::Error::custom(
                "coefficient matrix is ragged (inconsistent row widths)",
            ));
        }

        let mut data = Vec::with_capacity(row_count.saturating_mul(col_count));
        for row in rows {
            data.extend(row);
        }
        Ok(Self::F64(RowMajorMatrix {
            rows: row_count,
            cols: col_count,
            data,
        }))
    }
}

#[derive(Debug, Clone)]
pub struct AnyPolytopeCoefficients {
    pub generators: CoefficientMatrix,
    pub inequalities: CoefficientMatrix,
}

#[derive(Debug, Clone, Copy)]
pub struct PolytopeCoefficientsView<'a, N> {
    pub generators: RowMajorMatrixView<'a, N>,
    pub inequalities: RowMajorMatrixView<'a, N>,
}

#[derive(Debug, Clone)]
pub struct PolytopeCoefficients<N> {
    pub generators: RowMajorMatrix<N>,
    pub inequalities: RowMajorMatrix<N>,
}

impl AnyPolytopeCoefficients {
    pub fn cast<N: CoefficientScalar>(&self) -> PolytopeCoefficientsView<'_, N> {
        PolytopeCoefficientsView {
            generators: self.generators.cast::<N>(),
            inequalities: self.inequalities.cast::<N>(),
        }
    }

    pub fn coerce<N: CoefficientScalar>(
        &self,
    ) -> Result<PolytopeCoefficients<N>, calculo::num::ConversionError> {
        Ok(PolytopeCoefficients {
            generators: self.generators.coerce::<N>()?,
            inequalities: self.inequalities.coerce::<N>()?,
        })
    }

    pub fn stringify(&self) -> Result<PolytopeCoefficients<String>, calculo::num::ConversionError> {
        Ok(PolytopeCoefficients {
            generators: self.generators.stringify()?,
            inequalities: self.inequalities.stringify()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stats {
    pub dimension: usize,
    pub vertices: usize,
    pub facets: usize,
    pub ridges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendTiming {
    pub total: Duration,
    pub fast: Option<Duration>,
    pub resolve: Option<Duration>,
    pub exact: Option<Duration>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CddlibTimingDetail {
    pub build: Duration,
    pub incidence: Duration,
    pub vertex_adjacency: Duration,
    pub facet_adjacency: Duration,
    pub vertex_positions: Duration,
    pub post_inc: Duration,
    pub post_v_adj: Duration,
    pub post_f_adj: Duration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HowzatDdTimingDetail {
    pub fast_matrix: Duration,
    pub fast_dd: Duration,
    pub cert: Duration,
    pub repair_partial: Duration,
    pub repair_graph: Duration,
    pub exact_matrix: Duration,
    pub exact_dd: Duration,
    pub incidence: Duration,
    pub vertex_adjacency: Duration,
    pub facet_adjacency: Duration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HowzatLrsTimingDetail {
    pub matrix: Duration,
    pub lrs: Duration,
    pub cert: Duration,
    pub incidence: Duration,
    pub vertex_adjacency: Duration,
    pub facet_adjacency: Duration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LrslibTimingDetail {
    pub build: Duration,
    pub incidence: Duration,
    pub vertex_adjacency: Duration,
    pub facet_adjacency: Duration,
    pub post_inc: Duration,
    pub post_v_adj: Duration,
    pub post_f_adj: Duration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PplTimingDetail {
    pub build: Duration,
    pub incidence: Duration,
    pub vertex_adjacency: Duration,
    pub facet_adjacency: Duration,
    pub post_inc: Duration,
    pub post_v_adj: Duration,
    pub post_f_adj: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimingDetail {
    Cddlib(CddlibTimingDetail),
    HowzatDd(HowzatDdTimingDetail),
    HowzatLrs(HowzatLrsTimingDetail),
    Lrslib(LrslibTimingDetail),
    Ppl(PplTimingDetail),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendRun<Inc = SetFamily, Adj = SetFamily> {
    pub spec: Backend,
    pub stats: Stats,
    pub timing: BackendTiming,
    pub facets: Option<Vec<Vec<f64>>>,
    #[serde(skip, default)]
    pub coefficients: Option<AnyPolytopeCoefficients>,
    pub geometry: BackendGeometry<Inc, Adj>,
    pub fails: usize,
    pub fallbacks: usize,
    pub error: Option<String>,
    pub detail: Option<TimingDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendGeometry<Inc = SetFamily, Adj = SetFamily> {
    Baseline(BaselineGeometry<Inc, Adj>),
    Input(InputGeometry<Inc, Adj>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineGeometry<Inc = SetFamily, Adj = SetFamily> {
    pub vertex_positions: CoefficientMatrix,
    pub vertex_adjacency: Adj,
    pub facets_to_vertices: Inc,
    pub facet_adjacency: Adj,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputGeometry<Inc = SetFamily, Adj = SetFamily> {
    pub vertex_adjacency: Adj,
    pub facets_to_vertices: Inc,
    pub facet_adjacency: Adj,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendRunAny {
    Dense(BackendRun<SetFamily, SetFamily>),
    Sparse(BackendRun<ListFamily, AdjacencyList>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_accepts_adj_option() {
        let backend: Backend = "howzat-dd[adj[dense]]:f64".parse().unwrap();
        assert_eq!(backend.to_string(), "howzat-dd[adj[dense]]:f64");
    }

    #[test]
    fn parse_backend_rejects_cddlib_dense_adj() {
        assert!("cddlib[adj[dense]]:gmprational".parse::<Backend>().is_err());
    }

    #[test]
    fn parse_backend_accepts_sparse_adj_for_cddlib() {
        let backend: Backend = "cddlib[adj[sparse]]:gmprational".parse().unwrap();
        assert_eq!(backend.to_string(), "cddlib[adj[sparse]]:gmprational");
    }

    #[test]
    fn parse_backend_spec_accepts_f64_eps_syntax() {
        let spec: Backend = "howzat-dd[purify[snap]]:f64[min,eps[1e-12]]"
            .parse()
            .unwrap();
        assert_eq!(
            spec.to_string(),
            "howzat-dd[purify[snap]]:f64[eps[1e-12],min]"
        );

        let spec: Backend = "howzat-dd:f64[eps[0.0]]".parse().unwrap();
        assert!(matches!(spec.0, BackendSpec::HowzatDd { .. }));
    }

    #[test]
    fn parse_backend_spec_canonicalizes_backend_options() {
        let spec: Backend = "howzat-dd[adj[sparse],purify[snap]]:f64".parse().unwrap();
        assert_eq!(spec.to_string(), "howzat-dd[purify[snap],adj[sparse]]:f64");
    }

    #[test]
    fn parse_backend_accepts_howzat_dd_umpire_selectors() {
        let spec: Backend = "howzat-dd@int:gmprat".parse().unwrap();
        assert_eq!(spec.to_string(), "howzat-dd@int:gmprat");

        let spec: Backend = "howzat-dd@sp:gmprat".parse().unwrap();
        assert_eq!(spec.to_string(), "howzat-dd@sp:gmprat");
    }

    #[test]
    fn parse_backend_arg_accepts_marker_prefixes() {
        let arg: BackendArg = "cddlib:f64".parse().unwrap();
        assert_eq!(arg.spec.to_string(), "cddlib:f64");
        assert!(!arg.authoritative);
        assert!(!arg.perf_baseline);

        let arg: BackendArg = "^cddlib:f64".parse().unwrap();
        assert!(arg.authoritative);
        assert!(!arg.perf_baseline);

        let arg: BackendArg = "%cddlib:f64".parse().unwrap();
        assert!(!arg.authoritative);
        assert!(arg.perf_baseline);

        let arg: BackendArg = "^%howzat-dd[purify[snap]]:f64".parse().unwrap();
        assert_eq!(arg.spec.to_string(), "howzat-dd[purify[snap]]:f64");
        assert!(arg.authoritative);
        assert!(arg.perf_baseline);

        let arg: BackendArg = "%^cddlib:gmprational".parse().unwrap();
        assert!(arg.authoritative);
        assert!(arg.perf_baseline);
    }

    #[test]
    fn parse_backend_arg_rejects_duplicate_marker_prefixes() {
        let err = "^^cddlib:f64".parse::<BackendArg>().unwrap_err();
        assert!(
            err.contains("multiple '^'"),
            "expected '^'-prefix error, got: {err}"
        );

        let err = "%%cddlib:f64".parse::<BackendArg>().unwrap_err();
        assert!(
            err.contains("multiple '%'"),
            "expected '%'-prefix error, got: {err}"
        );
    }
}
