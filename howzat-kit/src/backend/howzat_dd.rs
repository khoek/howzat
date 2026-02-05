use std::time::{Duration, Instant};

use anyhow::{anyhow, ensure};
use calculo::num::{
    CoerceFrom, DashuFloat, DashuRat, DynamicEpsilon, Epsilon, F64Em7Epsilon, F64Em9Epsilon,
    F64Em12Epsilon, MaxNormalizer, MinNormalizer, NoNormalizer, Normalizer, Num, Rat,
    RugFloat, RugRat,
};
use howzat::dd::{
    ConeOptions, DefaultNormalizer, IntUmpire, SinglePrecisionUmpire as SpUmpire,
    SnapPurifier as Snap, UpcastingSnapPurifier,
};
use hullabaloo::adjacency::{AdjacencyBuilder, AdjacencyStore};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{AdjacencyOutput, IncidenceOutput};
use tracing::warn;

use crate::inequalities::HowzatInequalities;
use crate::vertices::HowzatVertices;

use super::howzat_common::{extract_howzat_coefficients, summarize_howzat_geometry};
use super::{
    Backend, BackendGeometry, BackendRun, BackendTiming, CoefficientMode, HowzatDdTimingDetail,
    InputGeometry, Stats, TimingDetail,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(super) enum HowzatDdNum {
    F64,
    RugFloat128,
    RugFloat256,
    RugFloat512,
    DashuFloat128,
    DashuFloat256,
    DashuFloat512,
    RugRat,
    DashuRat,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(super) enum HowzatDdUmpire {
    Default,
    Int,
    Sp,
}

impl HowzatDdUmpire {
    pub(super) fn canonical_token(self) -> Option<&'static str> {
        match self {
            Self::Default => None,
            Self::Int => Some("int"),
            Self::Sp => Some("sp"),
        }
    }
}

impl HowzatDdNum {
    fn canonical_token(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::RugFloat128 => "rugfloat[128]",
            Self::RugFloat256 => "rugfloat[256]",
            Self::RugFloat512 => "rugfloat[512]",
            Self::DashuFloat128 => "dashufloat[128]",
            Self::DashuFloat256 => "dashufloat[256]",
            Self::DashuFloat512 => "dashufloat[512]",
            Self::RugRat => "gmprat",
            Self::DashuRat => "dashurat",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum HowzatDdNormalizer {
    No,
    Min,
    Max,
}

impl HowzatDdNormalizer {
    fn canonical_token(self) -> &'static str {
        match self {
            Self::No => "no",
            Self::Min => "min",
            Self::Max => "max",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(super) enum HowzatDdPurifierSpec {
    Snap,
    UpSnap(HowzatDdNum),
}

impl HowzatDdPurifierSpec {
    pub(super) fn canonical_token(self) -> String {
        match self {
            Self::Snap => "snap".to_string(),
            Self::UpSnap(target) => format!("upsnap[{}]", target.canonical_token()),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum HowzatDdF64Eps {
    BuiltinEm7,
    BuiltinEm9,
    BuiltinEm12,
    Dynamic(u64),
}

impl HowzatDdF64Eps {
    fn canonical_value_token(self) -> String {
        match self {
            Self::BuiltinEm7 => "1e-7".to_string(),
            Self::BuiltinEm9 => "1e-9".to_string(),
            Self::BuiltinEm12 => "1e-12".to_string(),
            Self::Dynamic(bits) => format!("{:.17e}", f64::from_bits(bits)),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct HowzatDdCompute {
    num: HowzatDdNum,
    f64_eps: Option<HowzatDdF64Eps>,
    normalizer: Option<HowzatDdNormalizer>,
}

impl HowzatDdCompute {
    fn canonical_token(self) -> String {
        let num = self.num.canonical_token();
        if self.num == HowzatDdNum::F64 && (self.f64_eps.is_some() || self.normalizer.is_some()) {
            let mut parts = Vec::new();
            if let Some(eps) = self.f64_eps {
                parts.push(format!("eps[{}]", eps.canonical_value_token()));
            }
            if let Some(normalizer) = self.normalizer {
                parts.push(normalizer.canonical_token().to_string());
            }
            return format!("{num}[{}]", parts.join(","));
        }

        let Some(normalizer) = self.normalizer else {
            return num.to_string();
        };
        format!("{num}[{}]", normalizer.canonical_token())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum HowzatDdCheckKind {
    Resolve,
    Repair,
}

impl HowzatDdCheckKind {
    fn canonical_token(self) -> &'static str {
        match self {
            Self::Resolve => "resolve",
            Self::Repair => "repair",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct HowzatDdCheck {
    kind: HowzatDdCheckKind,
    target: HowzatDdNum,
}

impl HowzatDdCheck {
    fn canonical_token(self) -> String {
        format!(
            "{}[{}]",
            self.kind.canonical_token(),
            self.target.canonical_token()
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum HowzatDdStep {
    Compute(HowzatDdCompute),
    Check(HowzatDdCheck),
}

impl HowzatDdStep {
    fn canonical_token(self) -> String {
        match self {
            Self::Compute(compute) => compute.canonical_token(),
            Self::Check(check) => check.canonical_token(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(super) struct HowzatDdPipelineSpec {
    steps: Vec<HowzatDdStep>,
}

impl HowzatDdPipelineSpec {
    pub(super) fn canonical(&self) -> String {
        self.steps
            .iter()
            .copied()
            .map(HowzatDdStep::canonical_token)
            .collect::<Vec<_>>()
            .join("-")
    }

    pub(super) fn has_checks(&self) -> bool {
        self.steps
            .iter()
            .any(|step| matches!(step, HowzatDdStep::Check(_)))
    }

    pub(super) fn is_exact(&self) -> bool {
        self.steps.iter().all(|step| match *step {
            HowzatDdStep::Compute(compute) => {
                matches!(compute.num, HowzatDdNum::RugRat | HowzatDdNum::DashuRat)
            }
            HowzatDdStep::Check(check) => {
                matches!(check.target, HowzatDdNum::RugRat | HowzatDdNum::DashuRat)
            }
        })
    }
}

pub(super) const DEFAULT_HOWZAT_DD_PIPELINE: &str = "f64-repair[gmprat]";

fn split_howzat_dd_brackets(raw: &str) -> Option<(&str, Vec<&str>)> {
    let token = raw.trim();
    if token.is_empty() {
        return None;
    }

    if !token.contains('[') && token.contains(']') {
        return None;
    }

    let base_end = token.find('[').unwrap_or(token.len());
    let base = token.get(..base_end)?.trim();
    if base.is_empty() {
        return None;
    }

    let mut brackets: Vec<&str> = Vec::new();
    let mut pos = base_end;
    while pos < token.len() {
        let tail = token.get(pos..)?;
        let after_open = tail.strip_prefix('[')?;
        let mut depth = 1usize;
        let mut idx = 0usize;
        for (offset, ch) in after_open.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        idx = offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth != 0 {
            return None;
        }

        let inner = after_open.get(..idx)?.trim();
        brackets.push(inner);
        pos = token
            .len()
            .saturating_sub(after_open.len())
            .saturating_add(idx + 1);
    }

    Some((base, brackets))
}

pub(super) fn parse_howzat_dd_purifier(raw: &str) -> Result<HowzatDdPurifierSpec, String> {
    let token = raw.trim();
    if token.is_empty() {
        return Err("howzat-dd purifier cannot be empty".to_string());
    }
    let Some((base, brackets)) = split_howzat_dd_brackets(token) else {
        return Err(format!(
            "invalid howzat-dd purifier '{token}' (expected snap or upsnap[gmprat|dashurat])"
        ));
    };

    match base {
        "snap" => {
            if !brackets.is_empty() {
                return Err("snap does not take parameters (expected snap)".to_string());
            }
            Ok(HowzatDdPurifierSpec::Snap)
        }
        "upsnap" => {
            let [target_raw] = brackets.as_slice() else {
                return Err(
                    "upsnap expects exactly one parameter: upsnap[gmprat|dashurat]".to_string(),
                );
            };
            let Some(target) = parse_howzat_dd_num(target_raw) else {
                return Err(format!(
                    "unknown upsnap target '{target_raw}' (expected gmprat or dashurat)"
                ));
            };
            if !matches!(target, HowzatDdNum::RugRat | HowzatDdNum::DashuRat) {
                return Err(format!(
                    "upsnap only supports targets '{}' and '{}' (got '{}')",
                    HowzatDdNum::RugRat.canonical_token(),
                    HowzatDdNum::DashuRat.canonical_token(),
                    target.canonical_token()
                ));
            }
            Ok(HowzatDdPurifierSpec::UpSnap(target))
        }
        _ => Err(format!(
            "unknown howzat-dd purifier '{token}' (expected snap or upsnap[gmprat|dashurat])"
        )),
    }
}

fn parse_howzat_dd_num(raw: &str) -> Option<HowzatDdNum> {
    let (base, brackets) = split_howzat_dd_brackets(raw)?;

    match (base, brackets.as_slice()) {
        ("f64", []) => return Some(HowzatDdNum::F64),
        ("gmprat", []) => return Some(HowzatDdNum::RugRat),
        ("dashurat", []) => return Some(HowzatDdNum::DashuRat),
        _ => {}
    }

    let bits = match (base, brackets.as_slice()) {
        ("rugfloat" | "dashufloat", [bits]) => bits.parse::<u32>().ok(),
        _ => None,
    }?;

    match (base, bits) {
        ("rugfloat", 128) => Some(HowzatDdNum::RugFloat128),
        ("rugfloat", 256) => Some(HowzatDdNum::RugFloat256),
        ("rugfloat", 512) => Some(HowzatDdNum::RugFloat512),
        ("dashufloat", 128) => Some(HowzatDdNum::DashuFloat128),
        ("dashufloat", 256) => Some(HowzatDdNum::DashuFloat256),
        ("dashufloat", 512) => Some(HowzatDdNum::DashuFloat512),
        _ => None,
    }
}

fn parse_howzat_dd_normalizer(raw: &str) -> Option<HowzatDdNormalizer> {
    match raw.trim() {
        "no" => Some(HowzatDdNormalizer::No),
        "min" => Some(HowzatDdNormalizer::Min),
        "max" => Some(HowzatDdNormalizer::Max),
        _ => None,
    }
}

fn parse_howzat_dd_compute(raw: &str) -> Result<Option<HowzatDdCompute>, String> {
    let token = raw.trim();
    if token.is_empty() {
        return Ok(None);
    }

    let Some((base, brackets)) = split_howzat_dd_brackets(token) else {
        return Ok(None);
    };

    fn split_bracket_options(raw: &str) -> Result<Vec<&str>, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("option list cannot be empty".to_string());
        }

        let mut parts = Vec::new();
        let mut depth = 0usize;
        let mut start = 0usize;
        for (idx, ch) in raw.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or_else(|| "option list contains unmatched ']'".to_string())?;
                }
                ',' if depth == 0 => {
                    parts.push(raw[start..idx].trim());
                    start = idx + ch.len_utf8();
                }
                _ => {}
            }
        }
        if depth != 0 {
            return Err("option list contains unmatched '['".to_string());
        }
        parts.push(raw[start..].trim());

        if parts.iter().any(|p| p.is_empty()) {
            return Err("option list contains empty elements".to_string());
        }
        Ok(parts)
    }

    let (num, f64_eps, normalizer_raw): (HowzatDdNum, Option<HowzatDdF64Eps>, Option<&str>) =
        match base {
            "f64" => {
                let mut eps = None;
                let mut norm = None;

                if brackets.len() > 1 {
                    return Err(
                        "f64 accepts a single option list: f64[eps[...],no|min|max]".to_string()
                    );
                }

                for &bracket_group in &brackets {
                    for option in split_bracket_options(bracket_group)? {
                        if let Some(inner) = option
                            .strip_prefix("eps[")
                            .and_then(|s| s.strip_suffix(']'))
                        {
                            if eps.is_some() {
                                return Err("f64 accepts at most one eps spec: f64[eps[...],...]"
                                    .to_string());
                            }
                            let raw = inner.trim();
                            let value = raw.parse::<f64>().map_err(|_| {
                                format!(
                                    "invalid f64 eps '{raw}' (expected a finite floating literal)"
                                )
                            })?;
                            if !value.is_finite() || value < 0.0 {
                                return Err(format!(
                                    "invalid f64 eps '{raw}' (expected a finite non-negative float)"
                                ));
                            }

                            let bits = value.to_bits();
                            eps = Some(if bits == (1.0e-7f64).to_bits() {
                                HowzatDdF64Eps::BuiltinEm7
                            } else if bits == (1.0e-9f64).to_bits() {
                                HowzatDdF64Eps::BuiltinEm9
                            } else if bits == (1.0e-12f64).to_bits() {
                                HowzatDdF64Eps::BuiltinEm12
                            } else {
                                warn!(
                                    "howzat-dd:f64 uses non-builtin eps={raw} (parsed={parsed:.17e}); \
cannot inline at compile time; performance may be degraded",
                                    raw = raw,
                                    parsed = value,
                                );
                                HowzatDdF64Eps::Dynamic(bits)
                            });
                            continue;
                        }

                        if norm.is_some() {
                            return Err(
                                "f64 accepts at most one normalizer option: f64[...,no|min|max]"
                                    .to_string(),
                            );
                        }
                        norm = Some(option);
                    }
                }
                (HowzatDdNum::F64, eps, norm)
            }
            "rugfloat" | "dashufloat" => {
                let bits = brackets
                    .first()
                    .copied()
                    .ok_or_else(|| format!("{base} requires a precision: {base}[128|256|512]"))?;
                let bits = bits.parse::<u32>().map_err(|_| {
                    format!("unsupported {base} precision '{bits}' (supported: 128, 256, 512)")
                })?;
                let num = match (base, bits) {
                    ("rugfloat", 128) => HowzatDdNum::RugFloat128,
                    ("rugfloat", 256) => HowzatDdNum::RugFloat256,
                    ("rugfloat", 512) => HowzatDdNum::RugFloat512,
                    ("dashufloat", 128) => HowzatDdNum::DashuFloat128,
                    ("dashufloat", 256) => HowzatDdNum::DashuFloat256,
                    ("dashufloat", 512) => HowzatDdNum::DashuFloat512,
                    _ => {
                        return Err(format!(
                            "unsupported {base} precision '{bits}' (supported: 128, 256, 512)"
                        ));
                    }
                };
                let normalizer_raw = match brackets.as_slice() {
                    [_bits] => None,
                    [_bits, norm] => Some(*norm),
                    _ => {
                        return Err(format!(
                            "{base} accepts at most one normalizer suffix: {base}[bits][no|min|max]"
                        ));
                    }
                };
                (num, None, normalizer_raw)
            }
            "gmprat" | "dashurat" => {
                let num = match base {
                    "gmprat" => HowzatDdNum::RugRat,
                    "dashurat" => HowzatDdNum::DashuRat,
                    _ => unreachable!("match arms cover all cases"),
                };
                let normalizer_raw = match brackets.as_slice() {
                    [] => None,
                    [norm] => Some(*norm),
                    _ => {
                        return Err(format!(
                            "{base} accepts at most one normalizer suffix: {base}[no|min|max]"
                        ));
                    }
                };
                (num, None, normalizer_raw)
            }
            _ => return Ok(None),
        };

    let normalizer: Option<HowzatDdNormalizer> = match normalizer_raw {
        None => None,
        Some(raw) => Some(parse_howzat_dd_normalizer(raw).ok_or_else(|| {
            format!("unknown howzat-dd normalizer '{raw}' (expected no|min|max)")
        })?),
    };
    Ok(Some(HowzatDdCompute {
        num,
        f64_eps,
        normalizer,
    }))
}

fn parse_howzat_dd_check(raw: &str) -> Result<Option<HowzatDdCheck>, String> {
    let token = raw.trim();
    if token.is_empty() {
        return Ok(None);
    }

    let (kind, inner) = if let Some(inner) = token
        .strip_prefix("resolve[")
        .and_then(|s| s.strip_suffix(']'))
    {
        (HowzatDdCheckKind::Resolve, inner)
    } else if let Some(inner) = token
        .strip_prefix("repair[")
        .and_then(|s| s.strip_suffix(']'))
    {
        (HowzatDdCheckKind::Repair, inner)
    } else {
        return Ok(None);
    };

    let Some(target) = parse_howzat_dd_num(inner) else {
        return Err(format!(
            "unknown {} target '{inner}'",
            kind.canonical_token()
        ));
    };

    if !matches!(target, HowzatDdNum::RugRat | HowzatDdNum::DashuRat) {
        return Err(format!(
            "{} currently only supports targets '{}' and '{}' (got '{}')",
            kind.canonical_token(),
            HowzatDdNum::RugRat.canonical_token(),
            HowzatDdNum::DashuRat.canonical_token(),
            target.canonical_token()
        ));
    }

    Ok(Some(HowzatDdCheck { kind, target }))
}

pub(super) fn parse_howzat_dd_pipeline(raw: &str) -> Result<HowzatDdPipelineSpec, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err("howzat-dd pipeline cannot be empty".to_string());
    }

    let parts: Vec<&str> = {
        let mut parts = Vec::new();
        let mut depth = 0usize;
        let mut start = 0usize;
        for (idx, ch) in raw.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or_else(|| "howzat-dd pipeline contains unmatched ']'".to_string())?;
                }
                '-' if depth == 0 => {
                    parts.push(raw[start..idx].trim());
                    start = idx + ch.len_utf8();
                }
                _ => {}
            }
        }
        if depth != 0 {
            return Err("howzat-dd pipeline contains unmatched '['".to_string());
        }
        parts.push(raw[start..].trim());
        parts
    };

    let mut steps: Vec<HowzatDdStep> = Vec::new();
    for token in parts {
        if token.is_empty() {
            return Err("howzat-dd pipeline cannot contain empty tokens".to_string());
        }
        if let Some(check) = parse_howzat_dd_check(token)? {
            steps.push(HowzatDdStep::Check(check));
            continue;
        }
        if let Some(compute) = parse_howzat_dd_compute(token)? {
            steps.push(HowzatDdStep::Compute(compute));
            continue;
        }
        return Err(format!("unknown howzat-dd pipeline token '{token}'"));
    }

    let Some(first) = steps.first().copied() else {
        return Err("howzat-dd pipeline cannot be empty".to_string());
    };
    if !matches!(first, HowzatDdStep::Compute(_)) {
        return Err("howzat-dd pipeline must start with a numeric stage (e.g. f64)".to_string());
    }
    for pair in steps.windows(2) {
        let [a, b] = pair else { continue };
        if matches!(a, HowzatDdStep::Compute(_)) && matches!(b, HowzatDdStep::Compute(_)) {
            return Err(format!(
                "howzat-dd pipeline cannot have consecutive numeric stages ('{}-{}'); \
insert resolve[...] or repair[...] between them",
                a.canonical_token(),
                b.canonical_token()
            ));
        }
    }

    let mut last_compute: Option<HowzatDdNum> = None;
    for step in steps.iter().copied() {
        match step {
            HowzatDdStep::Compute(compute) => last_compute = Some(compute.num),
            HowzatDdStep::Check(check) => {
                let Some(prev) = last_compute else {
                    return Err("howzat-dd pipeline check must follow a numeric stage".to_string());
                };
                if !howzat_dd_can_coerce(prev, check.target) {
                    return Err(format!(
                        "{} after {} is not supported (no {} -> {} coercion available)",
                        check.canonical_token(),
                        prev.canonical_token(),
                        prev.canonical_token(),
                        check.target.canonical_token()
                    ));
                }
            }
        }
    }

    Ok(HowzatDdPipelineSpec { steps })
}

fn howzat_dd_can_coerce(from: HowzatDdNum, to: HowzatDdNum) -> bool {
    use HowzatDdNum::*;
    matches!(
        (from, to),
        (F64, _)
            | (RugFloat128, F64 | RugFloat128 | RugRat)
            | (RugFloat256, F64 | RugFloat256 | RugRat)
            | (RugFloat512, F64 | RugFloat512 | RugRat)
            | (DashuFloat128, F64 | DashuFloat128 | DashuRat)
            | (DashuFloat256, F64 | DashuFloat256 | DashuRat)
            | (DashuFloat512, F64 | DashuFloat512 | DashuRat)
            | (RugRat, F64 | RugRat)
            | (DashuRat, F64 | DashuRat)
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_howzat_dd_backend<
    V: HowzatVertices + ?Sized,
    Inc: From<SetFamily>,
    Adj: AdjacencyStore,
>(
    spec: Backend,
    dd_umpire: HowzatDdUmpire,
    purifier: Option<HowzatDdPurifierSpec>,
    pipeline: &HowzatDdPipelineSpec,
    output_incidence: bool,
    output_adjacency: bool,
    howzat_output_adjacency: AdjacencyOutput,
    coeff_mode: CoefficientMode,
    vertices: &V,
    howzat_options: &ConeOptions,
    timing: bool,
) -> Result<BackendRun<Inc, Adj>, anyhow::Error> {
    type HowzatRepr = hullabaloo::types::Generator;
    type Poly<N> = howzat::polyhedron::PolyhedronOutput<N, HowzatRepr>;
    type HowzatMatrix<N> = howzat::matrix::LpMatrix<N, HowzatRepr>;

    fn default_norm<N: DefaultNormalizer>() -> <N as DefaultNormalizer>::Norm {
        <N as DefaultNormalizer>::Norm::default()
    }

    fn dd_poly<N: Num, U: howzat::dd::Umpire<N>>(
        matrix: HowzatMatrix<N>,
        cone_options: &ConeOptions,
        poly_options: howzat::polyhedron::PolyhedronOptions,
        umpire: U,
    ) -> Result<Poly<N>, howzat::HowzatError> {
        Poly::<N>::from_matrix_dd(
            matrix,
            howzat::polyhedron::DdConfig {
                cone: cone_options.clone(),
                poly: poly_options,
            },
            umpire,
        )
    }

    #[derive(Clone, Debug)]
    enum PolyAny {
        F64(Poly<f64>),
        RugFloat128(Poly<RugFloat<128>>),
        RugFloat256(Poly<RugFloat<256>>),
        RugFloat512(Poly<RugFloat<512>>),
        DashuFloat128(Poly<DashuFloat<128>>),
        DashuFloat256(Poly<DashuFloat<256>>),
        DashuFloat512(Poly<DashuFloat<512>>),
        RugRat(Poly<RugRat>),
        DashuRat(Poly<DashuRat>),
    }

    struct ComputeStageArgs<'a, Vtx: HowzatVertices + ?Sized> {
        vertices: &'a Vtx,
        howzat_options: &'a ConeOptions,
        poly_options: howzat::polyhedron::PolyhedronOptions,
        spec_label: &'a str,
    }

    fn compute_dd<N: Num, E: Epsilon<N>, NM: Normalizer<N>, Vtx: HowzatVertices + ?Sized>(
        args: ComputeStageArgs<'_, Vtx>,
        stage: HowzatDdCompute,
        wrap: fn(Poly<N>) -> PolyAny,
        eps: E,
        normalizer: NM,
        purifier: Option<HowzatDdPurifierSpec>,
    ) -> Result<(PolyAny, Duration, Duration), anyhow::Error> {
        let ComputeStageArgs {
            vertices,
            howzat_options,
            poly_options,
            spec_label,
        } = args;

        let start_matrix = Instant::now();
        let matrix = vertices.build_howzat_generator_matrix::<N>()?;
        let time_matrix = start_matrix.elapsed();

        let start_dd = Instant::now();
        let poly = match purifier {
            Some(HowzatDdPurifierSpec::Snap) => {
                let umpire = SpUmpire::with_purifier(eps, normalizer, Snap::new());
                dd_poly(matrix, howzat_options, poly_options, umpire)
            }
            None => {
                let umpire = SpUmpire::with_normalizer(eps, normalizer);
                dd_poly(matrix, howzat_options, poly_options, umpire)
            }
            Some(HowzatDdPurifierSpec::UpSnap(_)) => {
                return Err(anyhow!(
                    "{spec_label} dd({}) does not support purify[upsnap[...]]",
                    stage.canonical_token()
                ));
            }
        }
        .map_err(|e| anyhow!("{spec_label} dd({}) failed: {e:?}", stage.canonical_token()))?;
        let time_dd = start_dd.elapsed();

        Ok((wrap(poly), time_matrix, time_dd))
    }

    fn compute_dd_int<N: Rat, E: Epsilon<N>, Vtx: HowzatVertices + ?Sized>(
        args: ComputeStageArgs<'_, Vtx>,
        stage: HowzatDdCompute,
        wrap: fn(Poly<N>) -> PolyAny,
        eps: E,
    ) -> Result<(PolyAny, Duration, Duration), anyhow::Error> {
        let ComputeStageArgs {
            vertices,
            howzat_options,
            poly_options,
            spec_label,
        } = args;

        let start_matrix = Instant::now();
        let matrix = vertices.build_howzat_generator_matrix::<N>()?;
        let time_matrix = start_matrix.elapsed();

        let start_dd = Instant::now();
        let umpire = IntUmpire::new(eps);
        let poly = Poly::<N>::from_matrix_dd_int_with_umpire(
            matrix,
            howzat::polyhedron::DdConfig {
                cone: howzat_options.clone(),
                poly: poly_options,
            },
            umpire,
        )
            .map_err(|e| anyhow!("{spec_label} dd({}) failed: {e:?}", stage.canonical_token()))?;
        let time_dd = start_dd.elapsed();

        Ok((wrap(poly), time_matrix, time_dd))
    }

    fn compute_poly_any<Vtx: HowzatVertices + ?Sized>(
        dd_umpire: HowzatDdUmpire,
        stage: HowzatDdCompute,
        purifier: Option<HowzatDdPurifierSpec>,
        vertices: &Vtx,
        howzat_options: &ConeOptions,
        poly_options: howzat::polyhedron::PolyhedronOptions,
        spec_label: &str,
    ) -> Result<(PolyAny, Duration, Duration), anyhow::Error> {
        use HowzatDdNormalizer::{Max, Min, No};
        use HowzatDdNum::{
            DashuFloat128, DashuFloat256, DashuFloat512, F64, RugFloat128, RugFloat256, RugFloat512,
        };

        let args = ComputeStageArgs {
            vertices,
            howzat_options,
            poly_options,
            spec_label,
        };

        fn compute_as_num<N: DefaultNormalizer, Vtx: HowzatVertices + ?Sized>(
            args: ComputeStageArgs<'_, Vtx>,
            stage: HowzatDdCompute,
            wrap: fn(Poly<N>) -> PolyAny,
            purifier: Option<HowzatDdPurifierSpec>,
        ) -> Result<(PolyAny, Duration, Duration), anyhow::Error> {
            match stage.normalizer {
                None => compute_dd(
                    args,
                    stage,
                    wrap,
                    N::default_eps(),
                    default_norm::<N>(),
                    purifier,
                ),
                Some(No) => compute_dd(args, stage, wrap, N::default_eps(), NoNormalizer, purifier),
                Some(Min) => compute_dd(args, stage, wrap, N::default_eps(), MinNormalizer, purifier),
                Some(Max) => compute_dd(args, stage, wrap, N::default_eps(), MaxNormalizer, purifier),
            }
        }

        match stage.num {
            F64 => {
                fn compute_f64_upsnap<M, E: Epsilon<f64>, Vtx: HowzatVertices + ?Sized>(
                    args: ComputeStageArgs<'_, Vtx>,
                    stage: HowzatDdCompute,
                    eps: E,
                    normalizer: impl Normalizer<f64>,
                ) -> Result<(PolyAny, Duration, Duration), anyhow::Error>
                where
                    M: CoerceFrom<f64> + Num,
                    f64: CoerceFrom<M>,
                {
                    let ComputeStageArgs {
                        vertices,
                        howzat_options,
                        poly_options,
                        spec_label,
                    } = args;

                    let start_matrix = Instant::now();
                    let matrix = vertices.build_howzat_generator_matrix::<f64>()?;
                    let time_matrix = start_matrix.elapsed();

                    let purifier = UpcastingSnapPurifier::<M, _>::new(M::default_eps());
                    let umpire = SpUmpire::with_purifier(eps, normalizer, purifier);

                    let start_dd = Instant::now();
                    let poly = dd_poly(matrix, howzat_options, poly_options, umpire).map_err(|e| {
                        anyhow!("{spec_label} dd({}) failed: {e:?}", stage.canonical_token())
                    })?;
                    let time_dd = start_dd.elapsed();

                    Ok((PolyAny::F64(poly), time_matrix, time_dd))
                }

                fn compute_f64_upsnap_any<E: Epsilon<f64>, Vtx: HowzatVertices + ?Sized>(
                    args: ComputeStageArgs<'_, Vtx>,
                    stage: HowzatDdCompute,
                    target: HowzatDdNum,
                    eps: E,
                    normalizer: impl Normalizer<f64>,
                ) -> Result<(PolyAny, Duration, Duration), anyhow::Error> {
                    match target {
                        HowzatDdNum::RugRat => {
                            compute_f64_upsnap::<RugRat, _, _>(args, stage, eps, normalizer)
                        }
                        HowzatDdNum::DashuRat => {
                            compute_f64_upsnap::<DashuRat, _, _>(args, stage, eps, normalizer)
                        }
                        _ => Err(anyhow!(
                            "internal: upsnap target must be gmprat or dashurat (got {})",
                            target.canonical_token()
                        )),
                    }
                }

                fn compute_f64_eps<E: Epsilon<f64>, Vtx: HowzatVertices + ?Sized>(
                    args: ComputeStageArgs<'_, Vtx>,
                    stage: HowzatDdCompute,
                    purifier: Option<HowzatDdPurifierSpec>,
                    eps: E,
                ) -> Result<(PolyAny, Duration, Duration), anyhow::Error> {
                    let Some(HowzatDdPurifierSpec::UpSnap(target)) = purifier else {
                        return match stage.normalizer {
                            None => compute_dd(
                                args,
                                stage,
                                PolyAny::F64,
                                eps,
                                default_norm::<f64>(),
                                purifier,
                            ),
                            Some(No) => compute_dd(args, stage, PolyAny::F64, eps, NoNormalizer, purifier),
                            Some(Min) => {
                                compute_dd(args, stage, PolyAny::F64, eps, MinNormalizer, purifier)
                            }
                            Some(Max) => {
                                compute_dd(args, stage, PolyAny::F64, eps, MaxNormalizer, purifier)
                            }
                        };
                    };

                    match stage.normalizer {
                        None => compute_f64_upsnap_any(args, stage, target, eps, default_norm::<f64>()),
                        Some(No) => compute_f64_upsnap_any(args, stage, target, eps, NoNormalizer),
                        Some(Min) => compute_f64_upsnap_any(args, stage, target, eps, MinNormalizer),
                        Some(Max) => compute_f64_upsnap_any(args, stage, target, eps, MaxNormalizer),
                    }
                }

                match stage.f64_eps {
                    None => compute_f64_eps(args, stage, purifier, f64::default_eps()),
                    Some(HowzatDdF64Eps::BuiltinEm7) => compute_f64_eps(args, stage, purifier, F64Em7Epsilon),
                    Some(HowzatDdF64Eps::BuiltinEm9) => compute_f64_eps(args, stage, purifier, F64Em9Epsilon),
                    Some(HowzatDdF64Eps::BuiltinEm12) => {
                        compute_f64_eps(args, stage, purifier, F64Em12Epsilon)
                    }
                    Some(HowzatDdF64Eps::Dynamic(bits)) => compute_f64_eps(
                        args,
                        stage,
                        purifier,
                        DynamicEpsilon::new(f64::from_bits(bits)),
                    ),
                }
            }
            RugFloat128 => compute_as_num::<RugFloat<128>, _>(args, stage, PolyAny::RugFloat128, purifier),
            RugFloat256 => compute_as_num::<RugFloat<256>, _>(args, stage, PolyAny::RugFloat256, purifier),
            RugFloat512 => compute_as_num::<RugFloat<512>, _>(args, stage, PolyAny::RugFloat512, purifier),
            DashuFloat128 => {
                compute_as_num::<DashuFloat<128>, _>(args, stage, PolyAny::DashuFloat128, purifier)
            }
            DashuFloat256 => {
                compute_as_num::<DashuFloat<256>, _>(args, stage, PolyAny::DashuFloat256, purifier)
            }
            DashuFloat512 => {
                compute_as_num::<DashuFloat<512>, _>(args, stage, PolyAny::DashuFloat512, purifier)
            }
            HowzatDdNum::RugRat => match dd_umpire {
                HowzatDdUmpire::Sp => compute_as_num::<RugRat, _>(args, stage, PolyAny::RugRat, purifier),
                HowzatDdUmpire::Default | HowzatDdUmpire::Int => {
                    ensure!(
                        purifier.is_none(),
                        "{spec_label} dd({}) does not support purification under IntUmpire",
                        stage.canonical_token()
                    );
                    ensure!(
                        stage.normalizer.is_none(),
                        "{spec_label} dd({}) does not support normalizer options under IntUmpire (use howzat-dd@sp:...)",
                        stage.canonical_token()
                    );
                    compute_dd_int(args, stage, PolyAny::RugRat, RugRat::default_eps())
                }
            },
            HowzatDdNum::DashuRat => match dd_umpire {
                HowzatDdUmpire::Sp => {
                    compute_as_num::<DashuRat, _>(args, stage, PolyAny::DashuRat, purifier)
                }
                HowzatDdUmpire::Default | HowzatDdUmpire::Int => {
                    ensure!(
                        purifier.is_none(),
                        "{spec_label} dd({}) does not support purification under IntUmpire",
                        stage.canonical_token()
                    );
                    ensure!(
                        stage.normalizer.is_none(),
                        "{spec_label} dd({}) does not support normalizer options under IntUmpire (use howzat-dd@sp:...)",
                        stage.canonical_token()
                    );
                    compute_dd_int(args, stage, PolyAny::DashuRat, DashuRat::default_eps())
                }
            },
        }
    }

    fn try_resolve_to_rat<Src, Dst>(
        poly: &Poly<Src>,
        poly_options: &howzat::polyhedron::PolyhedronOptions,
        wrap: fn(Poly<Dst>) -> PolyAny,
    ) -> Option<PolyAny>
    where
        Src: Num,
        Dst: Rat + CoerceFrom<Src>,
    {
        let eps = Dst::default_eps();
        resolve_howzat_certificate_as::<Src, Dst>(poly, poly_options, &eps).map(wrap)
    }

    fn resolve_any(
        inexact: &PolyAny,
        target: HowzatDdNum,
        poly_options: &howzat::polyhedron::PolyhedronOptions,
    ) -> Option<PolyAny> {
        use PolyAny::*;

        match target {
            HowzatDdNum::RugRat => {
                let wrap = PolyAny::RugRat;
                match inexact {
                    F64(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    RugFloat128(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    RugFloat256(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    RugFloat512(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    RugRat(poly) => Some(RugRat(poly.clone())),
                    _ => None,
                }
            }
            HowzatDdNum::DashuRat => {
                let wrap = PolyAny::DashuRat;
                match inexact {
                    F64(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    DashuFloat128(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    DashuFloat256(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    DashuFloat512(poly) => try_resolve_to_rat(poly, poly_options, wrap),
                    DashuRat(poly) => Some(DashuRat(poly.clone())),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn try_repair_to_rat<Src, Dst>(
        poly: &Poly<Src>,
        poly_options: &howzat::polyhedron::PolyhedronOptions,
        wrap: fn(Poly<Dst>) -> PolyAny,
    ) -> Option<(PolyAny, bool, Duration, Duration, Duration)>
    where
        Src: Num,
        Dst: Rat + CoerceFrom<Src>,
    {
        let eps = Dst::default_eps();
        repair_howzat_facet_graph_as::<Dst, Src>(poly, poly_options, &eps)
            .ok()
            .map(|(poly, report, cert, partial, repair)| {
                let frontier_ok = report
                    .frontier
                    .as_ref()
                    .map(|r| r.remaining_frontier_ridges() == 0 && !r.step_limit_reached())
                    .unwrap_or(true);

                (
                    wrap(poly),
                    report.unresolved_nodes == 0 && frontier_ok,
                    cert,
                    partial,
                    repair,
                )
            })
    }

    fn repair_any(
        inexact: &PolyAny,
        target: HowzatDdNum,
        poly_options: &howzat::polyhedron::PolyhedronOptions,
    ) -> Option<(PolyAny, bool, Duration, Duration, Duration)> {
        use PolyAny::*;

        match target {
            HowzatDdNum::RugRat => match inexact {
                F64(poly) => try_repair_to_rat(poly, poly_options, PolyAny::RugRat),
                RugFloat128(poly) => try_repair_to_rat(poly, poly_options, PolyAny::RugRat),
                RugFloat256(poly) => try_repair_to_rat(poly, poly_options, PolyAny::RugRat),
                RugFloat512(poly) => try_repair_to_rat(poly, poly_options, PolyAny::RugRat),
                RugRat(poly) => Some((
                    RugRat(poly.clone()),
                    true,
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                )),
                _ => None,
            },
            HowzatDdNum::DashuRat => match inexact {
                F64(poly) => try_repair_to_rat(poly, poly_options, PolyAny::DashuRat),
                DashuFloat128(poly) => try_repair_to_rat(poly, poly_options, PolyAny::DashuRat),
                DashuFloat256(poly) => try_repair_to_rat(poly, poly_options, PolyAny::DashuRat),
                DashuFloat512(poly) => try_repair_to_rat(poly, poly_options, PolyAny::DashuRat),
                DashuRat(poly) => Some((
                    DashuRat(poly.clone()),
                    true,
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                )),
                _ => None,
            },
            _ => None,
        }
    }

    let needs_certificate = pipeline
        .steps
        .iter()
        .any(|step| matches!(step, HowzatDdStep::Check(_)));
    let howzat_output_incidence = if output_incidence || needs_certificate {
        IncidenceOutput::Set
    } else {
        IncidenceOutput::Off
    };

    let poly_options_final = howzat::polyhedron::PolyhedronOptions {
        output_incidence: howzat_output_incidence,
        output_adjacency: howzat_output_adjacency,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let poly_options_inexact_repair = howzat::polyhedron::PolyhedronOptions {
        output_incidence: howzat_output_incidence,
        output_adjacency: howzat_output_adjacency,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: true,
        profile_adjacency: false,
    };

    let poly_options_cert = howzat::polyhedron::PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::Off,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let spec_label = spec.to_string();

    let start_total = Instant::now();

    let mut time_fast_matrix = Duration::ZERO;
    let mut time_fast_dd = Duration::ZERO;
    let mut time_cert = Duration::ZERO;
    let mut time_repair_partial = Duration::ZERO;
    let mut time_repair_graph = Duration::ZERO;
    let mut time_exact_matrix = Duration::ZERO;
    let mut time_exact_dd = Duration::ZERO;

    let mut executed_computes = 0usize;
    let mut fails = 0usize;

    let mut current: Option<PolyAny> = None;
    for (idx, step) in pipeline.steps.iter().copied().enumerate() {
        match step {
            HowzatDdStep::Compute(num) => {
                let mut needs_repair_hints = false;
                let mut has_next_compute = false;
                for next in pipeline.steps.iter().copied().skip(idx + 1) {
                    match next {
                        HowzatDdStep::Compute(_) => {
                            has_next_compute = true;
                            break;
                        }
                        HowzatDdStep::Check(check) => {
                            if check.kind == HowzatDdCheckKind::Repair {
                                needs_repair_hints = true;
                            }
                        }
                    }
                }

                let options = if has_next_compute {
                    if needs_repair_hints {
                        poly_options_inexact_repair.clone()
                    } else {
                        poly_options_cert.clone()
                    }
                } else if needs_repair_hints {
                    poly_options_inexact_repair.clone()
                } else {
                    poly_options_final.clone()
                };

                let (poly_any, time_matrix, time_dd) = compute_poly_any(
                    dd_umpire,
                    num,
                    purifier,
                    vertices,
                    howzat_options,
                    options,
                    &spec_label,
                )?;

                executed_computes += 1;
                if executed_computes == 1 {
                    time_fast_matrix += time_matrix;
                    time_fast_dd += time_dd;
                } else {
                    time_exact_matrix += time_matrix;
                    time_exact_dd += time_dd;
                }
                current = Some(poly_any);
            }
            HowzatDdStep::Check(check) => {
                let Some(inexact) = current.as_ref() else {
                    return Err(anyhow!(
                        "internal: howzat-dd check without preceding compute"
                    ));
                };

                let target = check.target;
                let opts = &poly_options_final;

                match check.kind {
                    HowzatDdCheckKind::Resolve => {
                        let start_cert = Instant::now();
                        if let Some(poly_any) = resolve_any(inexact, target, opts) {
                            current = Some(poly_any);
                            time_cert += start_cert.elapsed();
                            break;
                        }
                        time_cert += start_cert.elapsed();
                    }
                    HowzatDdCheckKind::Repair => {
                        let repaired = repair_any(inexact, target, opts);

                        if let Some((poly_any, ok, cert, partial, repair)) = repaired {
                            time_cert += cert;
                            time_repair_partial += partial;
                            time_repair_graph += repair;
                            if ok {
                                current = Some(poly_any);
                                break;
                            } else if idx + 1 == pipeline.steps.len() {
                                current = Some(poly_any);
                                fails = 1;
                            }
                        }
                    }
                }

                if idx + 1 == pipeline.steps.len() {
                    fails = 1;
                }
            }
        }
    }

    let Some(final_poly) = current else {
        return Err(anyhow!("internal: howzat-dd pipeline produced no poly"));
    };

    let fallbacks = executed_computes.saturating_sub(1);
    if !output_incidence {
        fn extract_facets<N: Num>(
            poly: &howzat::polyhedron::PolyhedronOutput<N, hullabaloo::types::Generator>,
        ) -> (usize, usize, usize, Vec<Vec<f64>>) {
            let vertex_count = poly.input().row_count();
            let output = poly.output();
            let rows = output.row_count();
            let cols = output.col_count();
            let linearity = output.linearity();
            let dim = poly.dimension();

            let facet_count = rows.saturating_sub(linearity.cardinality());
            let mut facet_rows: Vec<Vec<f64>> = Vec::with_capacity(facet_count);
            for r in 0..rows {
                if linearity.contains(r) {
                    continue;
                }
                let mut row: Vec<f64> = Vec::with_capacity(cols);
                for x in output.row(r).unwrap().iter() {
                    row.push(x.to_f64());
                }
                facet_rows.push(row);
            }
            (dim, vertex_count, facet_count, facet_rows)
        }

        let (dim, vertex_count, facet_count, facet_rows, coefficients) = match &final_poly {
            PolyAny::F64(poly) => {
                let (dim, vertex_count, facet_count, facet_rows) = extract_facets(poly);
                (dim, vertex_count, facet_count, Some(facet_rows), None)
            }
            PolyAny::RugFloat128(poly) => {
                let (dim, vertex_count, facet_count, facet_rows) = extract_facets(poly);
                (dim, vertex_count, facet_count, Some(facet_rows), None)
            }
            PolyAny::RugFloat256(poly) => {
                let (dim, vertex_count, facet_count, facet_rows) = extract_facets(poly);
                (dim, vertex_count, facet_count, Some(facet_rows), None)
            }
            PolyAny::RugFloat512(poly) => {
                let (dim, vertex_count, facet_count, facet_rows) = extract_facets(poly);
                (dim, vertex_count, facet_count, Some(facet_rows), None)
            }
            PolyAny::DashuFloat128(poly) => {
                let (dim, vertex_count, facet_count, facet_rows) = extract_facets(poly);
                (dim, vertex_count, facet_count, Some(facet_rows), None)
            }
            PolyAny::DashuFloat256(poly) => {
                let (dim, vertex_count, facet_count, facet_rows) = extract_facets(poly);
                (dim, vertex_count, facet_count, Some(facet_rows), None)
            }
            PolyAny::DashuFloat512(poly) => {
                let (dim, vertex_count, facet_count, facet_rows) = extract_facets(poly);
                (dim, vertex_count, facet_count, Some(facet_rows), None)
            }
            PolyAny::RugRat(poly) => {
                let vertex_count = poly.input().row_count();
                let output = poly.output();
                let rows = output.row_count();
                let linearity = output.linearity();
                let dim = poly.dimension();

                let mut facet_row_indices = Vec::new();
                facet_row_indices.reserve(rows.saturating_sub(linearity.cardinality()));
                for r in 0..rows {
                    if !linearity.contains(r) {
                        facet_row_indices.push(r);
                    }
                }

                let coefficients =
                    extract_howzat_coefficients(poly, &facet_row_indices, CoefficientMode::Exact)?;
                (
                    dim,
                    vertex_count,
                    facet_row_indices.len(),
                    None,
                    coefficients,
                )
            }
            PolyAny::DashuRat(poly) => {
                let vertex_count = poly.input().row_count();
                let output = poly.output();
                let rows = output.row_count();
                let linearity = output.linearity();
                let dim = poly.dimension();

                let mut facet_row_indices = Vec::new();
                facet_row_indices.reserve(rows.saturating_sub(linearity.cardinality()));
                for r in 0..rows {
                    if !linearity.contains(r) {
                        facet_row_indices.push(r);
                    }
                }

                let coefficients =
                    extract_howzat_coefficients(poly, &facet_row_indices, CoefficientMode::Exact)?;
                (
                    dim,
                    vertex_count,
                    facet_row_indices.len(),
                    None,
                    coefficients,
                )
            }
        };

        let total = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::HowzatDd(HowzatDdTimingDetail {
                fast_matrix: time_fast_matrix,
                fast_dd: time_fast_dd,
                cert: time_cert,
                repair_partial: time_repair_partial,
                repair_graph: time_repair_graph,
                exact_matrix: time_exact_matrix,
                exact_dd: time_exact_dd,
                incidence: Duration::ZERO,
                vertex_adjacency: Duration::ZERO,
                facet_adjacency: Duration::ZERO,
            }))
        } else {
            None
        };

        let time_checks = time_cert + time_repair_partial + time_repair_graph;
        return Ok(BackendRun {
            spec,
            stats: Stats {
                dimension: dim,
                vertices: vertex_count,
                facets: facet_count,
                ridges: 0,
            },
            timing: BackendTiming {
                total,
                fast: Some(time_fast_matrix + time_fast_dd),
                resolve: Some(time_checks),
                exact: if fallbacks > 0 {
                    Some(time_exact_matrix + time_exact_dd)
                } else {
                    None
                },
            },
            facets: facet_rows,
            coefficients,
            geometry: BackendGeometry::Input(InputGeometry {
                vertex_adjacency: Adj::Builder::new(0).finish(),
                facets_to_vertices: SetFamily::new(0, vertex_count).into(),
                facet_adjacency: Adj::Builder::new(0).finish(),
            }),
            fails,
            fallbacks,
            error: None,
            detail,
        });
    }

    let (geometry, extract_detail) = match &final_poly {
        PolyAny::F64(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::RugFloat128(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::RugFloat256(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::RugFloat512(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::DashuFloat128(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::DashuFloat256(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::DashuFloat512(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::RugRat(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
        PolyAny::DashuRat(poly) => {
            summarize_howzat_geometry::<Inc, Adj, _, _>(
                poly,
                output_adjacency,
                timing,
                coeff_mode != CoefficientMode::Off,
            )?
        }
    };

    let coefficients = if coeff_mode == CoefficientMode::Off {
        None
    } else {
        let Some(facet_row_indices) = geometry.facet_row_indices.as_deref() else {
            return Err(anyhow!("internal: facet_row_indices missing"));
        };
        match &final_poly {
            PolyAny::F64(poly) => extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?,
            PolyAny::RugFloat128(poly) => {
                extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?
            }
            PolyAny::RugFloat256(poly) => {
                extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?
            }
            PolyAny::RugFloat512(poly) => {
                extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?
            }
            PolyAny::DashuFloat128(poly) => {
                extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?
            }
            PolyAny::DashuFloat256(poly) => {
                extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?
            }
            PolyAny::DashuFloat512(poly) => {
                extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?
            }
            PolyAny::RugRat(poly) => extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?,
            PolyAny::DashuRat(poly) => extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?,
        }
    };

    let total = start_total.elapsed();
    let detail = if timing {
        Some(TimingDetail::HowzatDd(HowzatDdTimingDetail {
            fast_matrix: time_fast_matrix,
            fast_dd: time_fast_dd,
            cert: time_cert,
            repair_partial: time_repair_partial,
            repair_graph: time_repair_graph,
            exact_matrix: time_exact_matrix,
            exact_dd: time_exact_dd,
            incidence: extract_detail.incidence,
            vertex_adjacency: extract_detail.vertex_adjacency,
            facet_adjacency: extract_detail.facet_adjacency,
        }))
    } else {
        None
    };

    let time_checks = time_cert + time_repair_partial + time_repair_graph;
    Ok(BackendRun {
        spec,
        stats: geometry.stats,
        timing: BackendTiming {
            total,
            fast: Some(time_fast_matrix + time_fast_dd),
            resolve: Some(time_checks),
            exact: if fallbacks > 0 {
                Some(time_exact_matrix + time_exact_dd)
            } else {
                None
            },
        },
        facets: None,
        coefficients,
        geometry: BackendGeometry::Input(InputGeometry {
            vertex_adjacency: geometry.vertex_adjacency,
            facets_to_vertices: geometry.facets_to_vertices,
            facet_adjacency: geometry.facet_adjacency,
        }),
        fails,
        fallbacks,
        error: None,
        detail,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_howzat_dd_backend_from_inequalities<
    I: HowzatInequalities + ?Sized,
    Inc: From<SetFamily>,
    Adj: AdjacencyStore,
>(
    spec: Backend,
    dd_umpire: HowzatDdUmpire,
    purifier: Option<HowzatDdPurifierSpec>,
    pipeline: &HowzatDdPipelineSpec,
    output_incidence: bool,
    output_adjacency: bool,
    howzat_output_adjacency: AdjacencyOutput,
    coeff_mode: CoefficientMode,
    inequalities: &I,
    howzat_options: &ConeOptions,
    timing: bool,
) -> Result<BackendRun<Inc, Adj>, anyhow::Error> {
    ensure!(
        !pipeline.has_checks(),
        "howzat-dd inequality input currently does not support resolve/repair pipeline steps"
    );

    let Some(HowzatDdStep::Compute(stage)) = pipeline.steps.last().copied() else {
        return Err(anyhow!("howzat-dd pipeline must end with a numeric stage"));
    };

    type HowzatRepr = hullabaloo::types::Inequality;
    type Poly<N> = howzat::polyhedron::PolyhedronOutput<N, HowzatRepr>;
    type HowzatMatrix<N> = howzat::matrix::LpMatrix<N, HowzatRepr>;

    fn default_norm<N: DefaultNormalizer>() -> <N as DefaultNormalizer>::Norm {
        <N as DefaultNormalizer>::Norm::default()
    }

    fn dd_poly<N: Num, U: howzat::dd::Umpire<N>>(
        matrix: HowzatMatrix<N>,
        cone_options: &ConeOptions,
        poly_options: howzat::polyhedron::PolyhedronOptions,
        umpire: U,
    ) -> Result<Poly<N>, howzat::HowzatError> {
        Poly::<N>::from_matrix_dd(
            matrix,
            howzat::polyhedron::DdConfig {
                cone: cone_options.clone(),
                poly: poly_options,
            },
            umpire,
        )
    }

    struct ComputeStageArgs<'a, Input: HowzatInequalities + ?Sized> {
        inequalities: &'a Input,
        howzat_options: &'a ConeOptions,
        poly_options: howzat::polyhedron::PolyhedronOptions,
        spec_label: &'a str,
    }

    fn compute_dd<
        N: Num,
        E: Epsilon<N>,
        NM: Normalizer<N>,
        Input: HowzatInequalities + ?Sized,
    >(
        args: ComputeStageArgs<'_, Input>,
        stage: HowzatDdCompute,
        eps: E,
        normalizer: NM,
        purifier: Option<HowzatDdPurifierSpec>,
    ) -> Result<(Poly<N>, Duration, Duration), anyhow::Error> {
        let ComputeStageArgs {
            inequalities,
            howzat_options,
            poly_options,
            spec_label,
        } = args;

        ensure!(
            !matches!(purifier, Some(HowzatDdPurifierSpec::UpSnap(_))),
            "purify[upsnap[...]] is not supported for inequality input"
        );

        let start_matrix = Instant::now();
        let matrix = inequalities.build_howzat_inequality_matrix::<N>()?;
        let time_matrix = start_matrix.elapsed();

        let start_dd = Instant::now();
        let poly = match purifier {
            Some(HowzatDdPurifierSpec::Snap) => {
                let umpire = SpUmpire::with_purifier(eps, normalizer, Snap::new());
                dd_poly(matrix, howzat_options, poly_options, umpire)
            }
            None => {
                let umpire = SpUmpire::with_normalizer(eps, normalizer);
                dd_poly(matrix, howzat_options, poly_options, umpire)
            }
            Some(HowzatDdPurifierSpec::UpSnap(_)) => unreachable!("upsnap rejected above"),
        }
        .map_err(|e| anyhow!("{spec_label} dd({}) failed: {e:?}", stage.canonical_token()))?;
        let time_dd = start_dd.elapsed();

        Ok((poly, time_matrix, time_dd))
    }

    fn compute_dd_int<N: Rat, E: Epsilon<N>, Input: HowzatInequalities + ?Sized>(
        args: ComputeStageArgs<'_, Input>,
        stage: HowzatDdCompute,
        eps: E,
    ) -> Result<(Poly<N>, Duration, Duration), anyhow::Error> {
        let ComputeStageArgs {
            inequalities,
            howzat_options,
            poly_options,
            spec_label,
        } = args;

        let start_matrix = Instant::now();
        let matrix = inequalities.build_howzat_inequality_matrix::<N>()?;
        let time_matrix = start_matrix.elapsed();

        let start_dd = Instant::now();
        let umpire = IntUmpire::new(eps);
        let poly = Poly::<N>::from_matrix_dd_int_with_umpire(
            matrix,
            howzat::polyhedron::DdConfig {
                cone: howzat_options.clone(),
                poly: poly_options,
            },
            umpire,
        )
            .map_err(|e| anyhow!("{spec_label} dd({}) failed: {e:?}", stage.canonical_token()))?;
        let time_dd = start_dd.elapsed();

        Ok((poly, time_matrix, time_dd))
    }

    // Match stage numerics explicitly to avoid re-implementing the full multi-stage pipeline.
    let howzat_output_incidence = if output_incidence {
        IncidenceOutput::Set
    } else {
        IncidenceOutput::Off
    };

    let poly_options = howzat::polyhedron::PolyhedronOptions {
        output_incidence: howzat_output_incidence,
        output_adjacency: howzat_output_adjacency,
        input_incidence: IncidenceOutput::Off,
        input_adjacency: AdjacencyOutput::Off,
        save_basis_and_tableau: false,
        save_repair_hints: false,
        profile_adjacency: false,
    };

    let spec_label = spec.to_string();
    let start_total = Instant::now();

    use HowzatDdNormalizer::{Max, Min, No};

    let args = ComputeStageArgs {
        inequalities,
        howzat_options,
        poly_options,
        spec_label: &spec_label,
    };

    fn compute_as_num<N: DefaultNormalizer, Input: HowzatInequalities + ?Sized>(
        args: ComputeStageArgs<'_, Input>,
        stage: HowzatDdCompute,
        purifier: Option<HowzatDdPurifierSpec>,
    ) -> Result<(Poly<N>, Duration, Duration), anyhow::Error> {
        match stage.normalizer {
            None => compute_dd(
                args,
                stage,
                N::default_eps(),
                default_norm::<N>(),
                purifier,
            ),
            Some(No) => compute_dd(args, stage, N::default_eps(), NoNormalizer, purifier),
            Some(Min) => compute_dd(args, stage, N::default_eps(), MinNormalizer, purifier),
            Some(Max) => compute_dd(args, stage, N::default_eps(), MaxNormalizer, purifier),
        }
    }

    fn finish_run<
        N: Num + std::fmt::Display + super::CoefficientScalar,
        Inc: From<SetFamily>,
        Adj: AdjacencyStore,
    >(
        spec: Backend,
        poly: &Poly<N>,
        output_incidence: bool,
        output_adjacency: bool,
        coeff_mode: CoefficientMode,
        timing: bool,
        start_total: Instant,
        time_matrix: Duration,
        time_dd: Duration,
    ) -> Result<BackendRun<Inc, Adj>, anyhow::Error> {
        if !output_incidence {
            let total = start_total.elapsed();
            let detail = if timing {
                Some(TimingDetail::HowzatDd(HowzatDdTimingDetail {
                    fast_matrix: time_matrix,
                    fast_dd: time_dd,
                    cert: Duration::ZERO,
                    repair_partial: Duration::ZERO,
                    repair_graph: Duration::ZERO,
                    exact_matrix: Duration::ZERO,
                    exact_dd: Duration::ZERO,
                    incidence: Duration::ZERO,
                    vertex_adjacency: Duration::ZERO,
                    facet_adjacency: Duration::ZERO,
                }))
            } else {
                None
            };

            let vertices = poly.output().row_count();
            let facets = poly.input().row_count();
            let coefficients = if coeff_mode == CoefficientMode::Off {
                None
            } else {
                let facet_row_indices: Vec<usize> = (0..facets).collect();
                extract_howzat_coefficients(poly, &facet_row_indices, coeff_mode)?
            };
            return Ok(BackendRun {
                spec,
                stats: Stats {
                    dimension: poly.dimension(),
                    vertices,
                    facets,
                    ridges: 0,
                },
                timing: BackendTiming {
                    total,
                    fast: Some(time_matrix + time_dd),
                    resolve: None,
                    exact: None,
                },
                facets: None,
                coefficients,
                geometry: BackendGeometry::Input(InputGeometry {
                    vertex_adjacency: Adj::Builder::new(0).finish(),
                    facets_to_vertices: SetFamily::new(0, vertices).into(),
                    facet_adjacency: Adj::Builder::new(0).finish(),
                }),
                fails: 0,
                fallbacks: 0,
                error: None,
                detail,
            });
        }

        let store_facet_row_indices = coeff_mode != CoefficientMode::Off;
        let (geometry, extract_detail) = summarize_howzat_geometry::<Inc, Adj, _, _>(
            poly,
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
            extract_howzat_coefficients(poly, facet_row_indices, coeff_mode)?
        };

        let total = start_total.elapsed();
        let detail = if timing {
            Some(TimingDetail::HowzatDd(HowzatDdTimingDetail {
                fast_matrix: time_matrix,
                fast_dd: time_dd,
                cert: Duration::ZERO,
                repair_partial: Duration::ZERO,
                repair_graph: Duration::ZERO,
                exact_matrix: Duration::ZERO,
                exact_dd: Duration::ZERO,
                incidence: extract_detail.incidence,
                vertex_adjacency: extract_detail.vertex_adjacency,
                facet_adjacency: extract_detail.facet_adjacency,
            }))
        } else {
            None
        };

        Ok(BackendRun {
            spec,
            stats: geometry.stats,
            timing: BackendTiming {
                total,
                fast: Some(time_matrix + time_dd),
                resolve: None,
                exact: None,
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

    match stage.num {
        HowzatDdNum::F64 => {
            fn compute_f64_eps<E: Epsilon<f64>, Input: HowzatInequalities + ?Sized>(
                args: ComputeStageArgs<'_, Input>,
                stage: HowzatDdCompute,
                eps: E,
                purifier: Option<HowzatDdPurifierSpec>,
            ) -> Result<(Poly<f64>, Duration, Duration), anyhow::Error> {
                match stage.normalizer {
                    None => compute_dd(args, stage, eps, default_norm::<f64>(), purifier),
                    Some(HowzatDdNormalizer::No) => compute_dd(args, stage, eps, NoNormalizer, purifier),
                    Some(HowzatDdNormalizer::Min) => compute_dd(args, stage, eps, MinNormalizer, purifier),
                    Some(HowzatDdNormalizer::Max) => compute_dd(args, stage, eps, MaxNormalizer, purifier),
                }
            }

            let (poly, time_matrix, time_dd) = match stage.f64_eps {
                None => compute_f64_eps(args, stage, f64::default_eps(), purifier)?,
                Some(HowzatDdF64Eps::BuiltinEm7) => compute_f64_eps(args, stage, F64Em7Epsilon, purifier)?,
                Some(HowzatDdF64Eps::BuiltinEm9) => compute_f64_eps(args, stage, F64Em9Epsilon, purifier)?,
                Some(HowzatDdF64Eps::BuiltinEm12) => compute_f64_eps(args, stage, F64Em12Epsilon, purifier)?,
                Some(HowzatDdF64Eps::Dynamic(bits)) => compute_f64_eps(
                    args,
                    stage,
                    DynamicEpsilon::new(f64::from_bits(bits)),
                    purifier,
                )?,
            };

            finish_run::<_, Inc, Adj>(
                spec,
                &poly,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
                start_total,
                time_matrix,
                time_dd,
            )
        }
        HowzatDdNum::RugFloat128 => {
            let (poly, time_matrix, time_dd) = compute_as_num::<RugFloat<128>, _>(args, stage, purifier)?;
            finish_run::<_, Inc, Adj>(
                spec,
                &poly,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
                start_total,
                time_matrix,
                time_dd,
            )
        }
        HowzatDdNum::RugFloat256 => {
            let (poly, time_matrix, time_dd) = compute_as_num::<RugFloat<256>, _>(args, stage, purifier)?;
            finish_run::<_, Inc, Adj>(
                spec,
                &poly,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
                start_total,
                time_matrix,
                time_dd,
            )
        }
        HowzatDdNum::RugFloat512 => {
            let (poly, time_matrix, time_dd) = compute_as_num::<RugFloat<512>, _>(args, stage, purifier)?;
            finish_run::<_, Inc, Adj>(
                spec,
                &poly,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
                start_total,
                time_matrix,
                time_dd,
            )
        }
        HowzatDdNum::DashuFloat128 => {
            let (poly, time_matrix, time_dd) = compute_as_num::<DashuFloat<128>, _>(args, stage, purifier)?;
            finish_run::<_, Inc, Adj>(
                spec,
                &poly,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
                start_total,
                time_matrix,
                time_dd,
            )
        }
        HowzatDdNum::DashuFloat256 => {
            let (poly, time_matrix, time_dd) = compute_as_num::<DashuFloat<256>, _>(args, stage, purifier)?;
            finish_run::<_, Inc, Adj>(
                spec,
                &poly,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
                start_total,
                time_matrix,
                time_dd,
            )
        }
        HowzatDdNum::DashuFloat512 => {
            let (poly, time_matrix, time_dd) = compute_as_num::<DashuFloat<512>, _>(args, stage, purifier)?;
            finish_run::<_, Inc, Adj>(
                spec,
                &poly,
                output_incidence,
                output_adjacency,
                coeff_mode,
                timing,
                start_total,
                time_matrix,
                time_dd,
            )
        }
        HowzatDdNum::RugRat => match dd_umpire {
            HowzatDdUmpire::Sp => {
                let (poly, time_matrix, time_dd) = compute_as_num::<RugRat, _>(args, stage, purifier)?;
                finish_run::<_, Inc, Adj>(
                    spec,
                    &poly,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                    start_total,
                    time_matrix,
                    time_dd,
                )
            }
            HowzatDdUmpire::Default | HowzatDdUmpire::Int => {
                ensure!(
                    purifier.is_none(),
                    "{spec_label} dd({}) does not support purification under IntUmpire",
                    stage.canonical_token()
                );
                ensure!(
                    stage.normalizer.is_none(),
                    "{spec_label} dd({}) does not support normalizer options under IntUmpire (use howzat-dd@sp:...)",
                    stage.canonical_token()
                );

                let (poly, time_matrix, time_dd) = compute_dd_int(args, stage, RugRat::default_eps())?;
                finish_run::<_, Inc, Adj>(
                    spec,
                    &poly,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                    start_total,
                    time_matrix,
                    time_dd,
                )
            }
        },
        HowzatDdNum::DashuRat => match dd_umpire {
            HowzatDdUmpire::Sp => {
                let (poly, time_matrix, time_dd) = compute_as_num::<DashuRat, _>(args, stage, purifier)?;
                finish_run::<_, Inc, Adj>(
                    spec,
                    &poly,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                    start_total,
                    time_matrix,
                    time_dd,
                )
            }
            HowzatDdUmpire::Default | HowzatDdUmpire::Int => {
                ensure!(
                    purifier.is_none(),
                    "{spec_label} dd({}) does not support purification under IntUmpire",
                    stage.canonical_token()
                );
                ensure!(
                    stage.normalizer.is_none(),
                    "{spec_label} dd({}) does not support normalizer options under IntUmpire (use howzat-dd@sp:...)",
                    stage.canonical_token()
                );

                let (poly, time_matrix, time_dd) = compute_dd_int(args, stage, DashuRat::default_eps())?;
                finish_run::<_, Inc, Adj>(
                    spec,
                    &poly,
                    output_incidence,
                    output_adjacency,
                    coeff_mode,
                    timing,
                    start_total,
                    time_matrix,
                    time_dd,
                )
            }
        },
    }
}

fn resolve_howzat_certificate_as<N: Num, M>(
    poly: &howzat::polyhedron::PolyhedronOutput<N, hullabaloo::types::Generator>,
    poly_options: &howzat::polyhedron::PolyhedronOptions,
    eps: &impl Epsilon<M>,
) -> Option<howzat::polyhedron::PolyhedronOutput<M, hullabaloo::types::Generator>>
where
    M: Rat + CoerceFrom<N>,
{
    let cert = poly.certificate().ok()?;
    cert.resolve_as::<M>(
        poly_options.clone(),
        howzat::polyhedron::ResolveOptions::default(),
        eps,
    )
    .ok()
}

type HowzatFacetGraphRepair<M> = (
    howzat::polyhedron::PolyhedronOutput<M, hullabaloo::types::Generator>,
    howzat::verify::FacetGraphRepairReport,
    Duration,
    Duration,
    Duration,
);

fn repair_howzat_facet_graph_as<M, N>(
    poly: &howzat::polyhedron::PolyhedronOutput<N, hullabaloo::types::Generator>,
    poly_options: &howzat::polyhedron::PolyhedronOptions,
    eps: &impl Epsilon<M>,
) -> Result<HowzatFacetGraphRepair<M>, anyhow::Error>
where
    M: Rat + CoerceFrom<N>,
    N: Num,
{
    let start_cert = Instant::now();
    let cert = poly
        .certificate()
        .map_err(|e| anyhow!("howzat-dd repair: missing certificate: {e:?}"))?;
    let time_cert = start_cert.elapsed();

    let resolve_options = howzat::polyhedron::ResolveOptions {
        partial_use_certificate_only: true,
        ..howzat::polyhedron::ResolveOptions::default()
    };
    let mut partial_options = poly_options.clone();
    partial_options.output_adjacency = AdjacencyOutput::Off;
    let start_partial = Instant::now();
    let prepared = cert
        .resolve_partial_prepared_minimal_as::<M>(partial_options, resolve_options, eps)
        .map_err(|e| anyhow!("howzat-dd repair: partial resolve failed: {e:?}"))?;
    let time_partial = start_partial.elapsed();

    let repair_options = howzat::verify::FacetGraphRepairOptions {
        rebuild_polyhedron_output: true,
        frontier: howzat::verify::FrontierRepairMode::General,
        ..howzat::verify::FacetGraphRepairOptions::default()
    };
    let start_repair = Instant::now();
    let repaired = prepared
        .repair_facet_graph(poly, repair_options, eps)
        .map_err(|e| anyhow!("howzat-dd repair: facet-graph repair failed: {e:?}"))?;

    let report = repaired.report().clone();
    let Some(rebuilt) = repaired.rebuilt_polyhedron().cloned() else {
        return Err(anyhow!(
            "howzat-dd repair: facet-graph repair did not rebuild polyhedron"
        ));
    };
    let time_repair = start_repair.elapsed();
    Ok((rebuilt, report, time_cert, time_partial, time_repair))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_howzat_dd_brackets_supports_nested_eps() {
        let (base, brackets) = split_howzat_dd_brackets("f64[eps[1e-12],max]").unwrap();
        assert_eq!(base, "f64");
        assert_eq!(brackets, vec!["eps[1e-12],max"]);
    }

    #[test]
    fn parse_howzat_dd_compute_parses_f64_builtin_eps() {
        let compute = parse_howzat_dd_compute("f64[eps[1e-12]]")
            .unwrap()
            .expect("expected f64 compute stage");
        assert_eq!(compute.num, HowzatDdNum::F64);
        assert_eq!(compute.f64_eps, Some(HowzatDdF64Eps::BuiltinEm12));
        assert_eq!(compute.normalizer, None);
    }

    #[test]
    fn parse_howzat_dd_compute_rejects_conflicting_f64_options() {
        let err = parse_howzat_dd_compute("f64[min,max]")
            .unwrap_err()
            .to_ascii_lowercase();
        assert!(
            err.contains("at most one normalizer"),
            "expected normalizer conflict error, got: {err}"
        );
    }
}
