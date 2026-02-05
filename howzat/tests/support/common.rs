#![allow(dead_code)]

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use calculo::num::Num;
use howzat::lp::LpObjective;
use howzat::matrix::{CanonicalizationResult, LpMatrix, LpMatrixBuilder};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{
    AdjacencyOutput, Generator, IncidenceOutput, Inequality, Representation, RepresentationKind,
    RowId, RowSet,
};

pub type TestNumber = f64;

static LOGGING: OnceLock<()> = OnceLock::new();

pub fn init_logging() {
    LOGGING.get_or_init(|| {
        tracing_subscriber::fmt()
            .with_test_writer()
            .without_time()
            .init();
    });
}

pub fn normalize_matrix<R: Representation>(
    matrix: &LpMatrix<TestNumber, R>,
) -> LpMatrix<TestNumber, R> {
    let eps = TestNumber::default_eps();
    matrix.normalized_sorted_unique(&eps).0
}

pub fn format_matrix<R: Representation>(matrix: &LpMatrix<TestNumber, R>) -> String {
    let mut out = String::new();
    let rep = match matrix.representation() {
        RepresentationKind::Inequality => "H-representation",
        RepresentationKind::Generator => "V-representation",
    };
    out.push_str(rep);
    out.push('\n');
    out.push_str("begin\n");
    let m = matrix.row_count();
    let d = matrix.col_count();
    out.push_str(&format!("{m:4} {d:4} real\n"));
    for row in matrix.rows() {
        for val in row {
            out.push_str(&format!(" {:.12e}", *val));
        }
        out.push('\n');
    }
    out.push_str("end\n");
    if !matrix.linearity().is_empty() {
        out.push_str("linearity:");
        for idx in matrix.linearity().iter() {
            out.push_str(&format!(" {}", idx));
        }
        out.push('\n');
    }
    out
}

pub fn matrices_approx_equal(
    a: &LpMatrix<TestNumber, Inequality>,
    b: &LpMatrix<TestNumber, Inequality>,
    tol: TestNumber,
) -> bool {
    if a.row_count() != b.row_count() || a.col_count() != b.col_count() {
        return false;
    }
    for (ra, rb) in a.rows().zip(b.rows()) {
        for (va, vb) in ra.iter().zip(rb.iter()) {
            if (va - vb).abs() > tol {
                return false;
            }
        }
    }
    a.linearity() == b.linearity() && a.representation() == b.representation()
}

pub fn rows_present(
    sup: &LpMatrix<TestNumber, Inequality>,
    sub: &LpMatrix<TestNumber, Inequality>,
    tol: TestNumber,
) -> bool {
    for sr in sub.rows() {
        let mut found = false;
        for rr in sup.rows() {
            if sr.len() != rr.len() {
                continue;
            }
            if sr.iter().zip(rr.iter()).all(|(a, b)| (a - b).abs() <= tol) {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

#[derive(Clone, Debug, Default)]
pub struct ProjectSpec {
    pub dimension: usize,
    pub keep_columns: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ParsedInput {
    pub matrix: ParsedMatrix,
    pub project: Option<ProjectSpec>,
    pub representation: RepresentationKind,
    pub adjacency_output: Option<AdjacencyOutput>,
    pub incidence_output: Option<IncidenceOutput>,
    pub input_adjacency_output: Option<AdjacencyOutput>,
    pub input_incidence_output: Option<IncidenceOutput>,
    pub row_count: usize,
    pub col_count: usize,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub enum ParsedMatrix {
    Inequality(LpMatrix<TestNumber, Inequality>),
    Generator(LpMatrix<TestNumber, Generator>),
}

#[derive(Clone, Debug)]
pub enum ParsedCanonicalizationResult {
    Inequality(CanonicalizationResult<TestNumber, Inequality>),
    Generator(CanonicalizationResult<TestNumber, Generator>),
}

impl ParsedCanonicalizationResult {
    pub fn matrix(&self) -> ParsedMatrix {
        match self {
            ParsedCanonicalizationResult::Inequality(r) => {
                ParsedMatrix::Inequality(r.matrix().clone())
            }
            ParsedCanonicalizationResult::Generator(r) => {
                ParsedMatrix::Generator(r.matrix().clone())
            }
        }
    }
}

impl ParsedMatrix {
    pub fn representation(&self) -> RepresentationKind {
        match self {
            ParsedMatrix::Inequality(_) => RepresentationKind::Inequality,
            ParsedMatrix::Generator(_) => RepresentationKind::Generator,
        }
    }

    pub fn row_count(&self) -> usize {
        match self {
            ParsedMatrix::Inequality(m) => m.row_count(),
            ParsedMatrix::Generator(m) => m.row_count(),
        }
    }

    pub fn col_count(&self) -> usize {
        match self {
            ParsedMatrix::Inequality(m) => m.col_count(),
            ParsedMatrix::Generator(m) => m.col_count(),
        }
    }

    pub fn rows(&self) -> Vec<Vec<TestNumber>> {
        match self {
            ParsedMatrix::Inequality(m) => m.rows().map(|r| r.to_vec()).collect(),
            ParsedMatrix::Generator(m) => m.rows().map(|r| r.to_vec()).collect(),
        }
    }

    pub fn linearity(&self) -> RowSet {
        match self {
            ParsedMatrix::Inequality(m) => m.linearity().clone(),
            ParsedMatrix::Generator(m) => m.linearity().clone(),
        }
    }

    pub fn row_vec(&self) -> Vec<TestNumber> {
        match self {
            ParsedMatrix::Inequality(m) => m.row_vec().to_vec(),
            ParsedMatrix::Generator(m) => m.row_vec().to_vec(),
        }
    }

    pub fn objective(&self) -> LpObjective {
        match self {
            ParsedMatrix::Inequality(m) => m.objective(),
            ParsedMatrix::Generator(m) => m.objective(),
        }
    }

    pub fn redundant_rows(&self) -> Result<RowSet, howzat::HowzatError> {
        let eps = TestNumber::default_eps();
        match self {
            ParsedMatrix::Inequality(m) => m.redundant_rows(&eps),
            ParsedMatrix::Generator(m) => m.redundant_rows(&eps),
        }
    }

    pub fn implicit_linearity_rows(&self) -> Result<RowSet, howzat::HowzatError> {
        let eps = TestNumber::default_eps();
        match self {
            ParsedMatrix::Inequality(m) => m.implicit_linearity_rows(&eps),
            ParsedMatrix::Generator(m) => m.implicit_linearity_rows(&eps),
        }
    }

    pub fn strongly_redundant_rows(&self) -> Result<RowSet, howzat::HowzatError> {
        let eps = TestNumber::default_eps();
        match self {
            ParsedMatrix::Inequality(m) => m.strongly_redundant_rows(&eps),
            ParsedMatrix::Generator(m) => m.strongly_redundant_rows(&eps),
        }
    }

    pub fn redundant_rows_via_shooting(&self) -> Result<RowSet, howzat::HowzatError> {
        let eps = TestNumber::default_eps();
        match self {
            ParsedMatrix::Inequality(m) => m.redundant_rows_via_shooting(&eps),
            // For Generator matrices, fall back to redundant_rows
            ParsedMatrix::Generator(m) => m.redundant_rows(&eps),
        }
    }

    pub fn fourier_project(
        &self,
        dimension: usize,
        keep_columns: &hullabaloo::types::ColSet,
    ) -> Result<ParsedMatrix, howzat::HowzatError> {
        let eps = TestNumber::default_eps();
        let keep: Vec<usize> = keep_columns.iter().map(|c| c.as_index()).collect();
        match self {
            ParsedMatrix::Inequality(m) => m
                .fourier_project(dimension, &keep, &eps)
                .map(ParsedMatrix::Inequality),
            // Fourier projection is only available for Inequality matrices
            ParsedMatrix::Generator(_) => {
                panic!("Fourier projection is not available for generator matrices")
            }
        }
    }

    pub fn canonicalize(&self) -> Result<ParsedCanonicalizationResult, howzat::HowzatError> {
        let eps = TestNumber::default_eps();
        match self {
            ParsedMatrix::Inequality(m) => m
                .canonicalize(&eps)
                .map(ParsedCanonicalizationResult::Inequality),
            ParsedMatrix::Generator(m) => m
                .canonicalize(&eps)
                .map(ParsedCanonicalizationResult::Generator),
        }
    }

    pub fn adjacency(&self) -> Result<SetFamily, howzat::HowzatError> {
        let eps = TestNumber::default_eps();
        match self {
            ParsedMatrix::Inequality(m) => m.adjacency(&eps),
            ParsedMatrix::Generator(m) => m.adjacency(&eps),
        }
    }

    pub fn as_inequality(&self) -> Option<&LpMatrix<TestNumber, Inequality>> {
        match self {
            ParsedMatrix::Inequality(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_generator(&self) -> Option<&LpMatrix<TestNumber, Generator>> {
        match self {
            ParsedMatrix::Generator(m) => Some(m),
            _ => None,
        }
    }
}

pub fn cdd_input_files() -> Vec<PathBuf> {
    collect_files_with_extensions(&PathBuf::from("tests/data"), &["ine", "ext"])
}

pub fn parse_cdd_file(path: &Path) -> Result<ParsedInput, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    parse_cdd_content(&content, path)
}

pub fn parse_cdd_content(content: &str, path: &Path) -> Result<ParsedInput, String> {
    let mut representation = None;
    let mut row_count = 0usize;
    let mut col_count = 0usize;
    let mut rows: Vec<Vec<TestNumber>> = Vec::new();
    let mut linearity_indices: Vec<usize> = Vec::new();
    let mut objective = LpObjective::None;
    let mut row_vec: Vec<TestNumber> = Vec::new();
    let mut project = None;
    let mut awaiting_size = false;
    let mut rows_remaining: Option<usize> = None;

    let mut lines = content.lines().peekable();
    let mut pending_tokens: Vec<String> = Vec::new();
    while let Some(raw_line) = lines.next() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') {
            continue;
        }
        let lowered = trimmed.to_ascii_lowercase();
        if awaiting_size {
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.len() < 2 {
                return Err(format!(
                    "size line missing entries in {}: {}",
                    path.display(),
                    trimmed
                ));
            }
            row_count = tokens[0]
                .parse::<usize>()
                .map_err(|e| format!("bad row count in {}: {e}", path.display()))?;
            col_count = tokens[1]
                .parse::<usize>()
                .map_err(|e| format!("bad col count in {}: {e}", path.display()))?;
            rows_remaining = Some(row_count);
            awaiting_size = false;
            continue;
        }

        if let Some(remaining) = rows_remaining {
            if remaining > 0 {
                pending_tokens.extend(
                    trimmed
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>(),
                );
                if pending_tokens.len() >= col_count {
                    let row_tokens: Vec<String> = pending_tokens.drain(0..col_count).collect();
                    let parsed_row = parse_row_tokens(&row_tokens, col_count, path)?;
                    rows.push(parsed_row);
                    rows_remaining = Some(remaining - 1);
                }
                continue;
            } else {
                pending_tokens.clear();
                rows_remaining = None;
            }
        }

        if lowered.starts_with("h-representation") {
            representation = Some(RepresentationKind::Inequality);
            continue;
        }
        if lowered.starts_with("v-representation") {
            representation = Some(RepresentationKind::Generator);
            continue;
        }
        if lowered.starts_with("linearity") {
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            for token in tokens.into_iter().skip(2) {
                if let Ok(idx) = token.parse::<usize>() {
                    linearity_indices.push(idx);
                }
            }
            continue;
        }
        if lowered == "begin" {
            awaiting_size = true;
            continue;
        }
        if lowered.starts_with("end") {
            rows_remaining = None;
            continue;
        }
        if lowered.starts_with("maximize") || lowered.starts_with("minimize") {
            objective = if lowered.starts_with("maximize") {
                LpObjective::Maximize
            } else {
                LpObjective::Minimize
            };
            let maybe_inline = trimmed.split_whitespace().skip(1).collect::<Vec<&str>>();
            let value_line = if maybe_inline.is_empty() {
                lines
                    .find(|l| !l.trim().is_empty() && !l.trim().starts_with('*'))
                    .unwrap_or("")
                    .to_string()
            } else {
                maybe_inline.join(" ")
            };
            if !value_line.is_empty() {
                row_vec = parse_row(value_line.trim(), col_count, path)?;
            }
            continue;
        }
        if lowered.starts_with("project") {
            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.len() >= 2 {
                let dimension = tokens[1].parse::<usize>().unwrap_or(0);
                let keep_columns = tokens
                    .into_iter()
                    .skip(2)
                    .filter_map(|t| t.parse::<usize>().ok())
                    .collect();
                project = Some(ProjectSpec {
                    dimension,
                    keep_columns,
                });
            }
            continue;
        }
        match lowered.as_str() {
            "incidence"
            | "incidence_cardinality"
            | "incidencecardinality"
            | "adjacency"
            | "adjacency_degree"
            | "adjacency_degrees"
            | "inputincidence"
            | "inputincidence_cardinality"
            | "inputincidencecardinality"
            | "inputadjacency"
            | "inputadjacency_degree"
            | "inputadjacency_degrees" => continue,
            _ => {}
        }
    }

    let rep = representation
        .ok_or_else(|| format!("missing representation header in {}", path.display()))?;
    if rows.len() != row_count {
        return Err(format!(
            "row count mismatch in {}: expected {row_count}, got {}",
            path.display(),
            rows.len()
        ));
    }
    let row_vec = if row_vec.is_empty() {
        vec![0.0; col_count]
    } else {
        row_vec
    };

    let mut linearity = RowSet::new(row_count);
    for idx in linearity_indices {
        if idx > 0 {
            linearity.insert(idx - 1);
        }
    }

    let matrix = match rep {
        RepresentationKind::Inequality => {
            let builder = LpMatrixBuilder::from_rows(rows)
                .with_objective(objective)
                .with_row_vec(row_vec)
                .with_linearity(linearity);
            ParsedMatrix::Inequality(builder.build())
        }
        RepresentationKind::Generator => {
            let builder = LpMatrixBuilder::<f64, Generator>::from_rows(rows)
                .with_objective(objective)
                .with_row_vec(row_vec)
                .with_linearity(linearity);
            ParsedMatrix::Generator(builder.build())
        }
    };

    Ok(ParsedInput {
        matrix,
        project,
        representation: rep,
        row_count,
        col_count,
        path: path.to_path_buf(),
        adjacency_output: None,
        incidence_output: None,
        input_adjacency_output: None,
        input_incidence_output: None,
    })
}

fn parse_row(line: &str, col_count: usize, path: &Path) -> Result<Vec<TestNumber>, String> {
    let mut values = Vec::with_capacity(col_count);
    for token in line.split_whitespace() {
        if values.len() == col_count {
            break;
        }
        values.push(parse_number(token, path)?);
    }
    if values.len() != col_count {
        return Err(format!(
            "expected {col_count} entries in {}, got {}",
            path.display(),
            values.len()
        ));
    }
    Ok(values)
}

fn parse_row_tokens(
    tokens: &[String],
    col_count: usize,
    path: &Path,
) -> Result<Vec<TestNumber>, String> {
    let mut values = Vec::with_capacity(col_count);
    for token in tokens.iter().take(col_count) {
        values.push(parse_number(token, path)?);
    }
    if values.len() != col_count {
        return Err(format!(
            "expected {col_count} entries in {}, got {}",
            path.display(),
            values.len()
        ));
    }
    Ok(values)
}

fn parse_number(token: &str, path: &Path) -> Result<TestNumber, String> {
    if let Some((num, denom)) = token.split_once('/') {
        let n = num
            .parse::<f64>()
            .map_err(|e| format!("bad numerator {token} in {}: {e}", path.display()))?;
        let d = denom
            .parse::<f64>()
            .map_err(|e| format!("bad denominator {token} in {}: {e}", path.display()))?;
        if d == 0.0 {
            return Err(format!(
                "zero denominator in {} for {}",
                path.display(),
                token
            ));
        }
        return Ok(n / d);
    }
    token
        .parse::<f64>()
        .map_err(|e| format!("failed to parse number {token} in {}: {e}", path.display()))
}

pub fn parse_set_family_file(path: &Path) -> Result<SetFamily, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    parse_set_family_content(&content, path)
}

pub fn parse_set_family_content(content: &str, path: &Path) -> Result<SetFamily, String> {
    let mut family_size = None;
    let mut set_capacity = None;
    let mut builder = None;
    let mut lines = content.lines();

    while let Some(raw_line) = lines.next() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('*') {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("begin") {
            if let Some(size_line) = lines.next() {
                let tokens: Vec<&str> = size_line.split_whitespace().collect();
                if tokens.len() < 2 {
                    return Err(format!(
                        "set family size line missing entries in {}: {}",
                        path.display(),
                        size_line
                    ));
                }
                family_size = Some(
                    tokens[0]
                        .parse::<usize>()
                        .map_err(|e| format!("bad family size in {}: {e}", path.display()))?,
                );
                set_capacity = Some(
                    tokens[1]
                        .parse::<usize>()
                        .map_err(|e| format!("bad set capacity in {}: {e}", path.display()))?,
                );
                builder = Some(SetFamily::builder(
                    family_size.expect("family size parsed"),
                    set_capacity.expect("set capacity parsed"),
                ));
            }
            continue;
        }
        if trimmed.eq_ignore_ascii_case("end") {
            break;
        }
        if builder.is_none() {
            continue;
        }
        let (index_part, members_part) = trimmed
            .split_once(':')
            .ok_or_else(|| format!("missing ':' in {} line: {}", path.display(), raw_line))?;
        let idx_tokens: Vec<&str> = index_part.split_whitespace().collect();
        if idx_tokens.len() < 2 {
            return Err(format!(
                "missing set index/cardinality in {} line: {}",
                path.display(),
                raw_line
            ));
        }
        let set_index_raw = idx_tokens[0]
            .parse::<usize>()
            .map_err(|e| format!("bad set index in {}: {e}", path.display()))?;
        let cardinality = idx_tokens[1]
            .parse::<isize>()
            .map_err(|e| format!("bad cardinality in {}: {e}", path.display()))?;
        let capacity = set_capacity.ok_or_else(|| {
            format!(
                "set capacity not initialized before reading members in {}",
                path.display()
            )
        })?;
        let family = family_size.ok_or_else(|| {
            format!(
                "family size not initialized before reading members in {}",
                path.display()
            )
        })?;
        if set_index_raw == 0 || set_index_raw > family {
            return Err(format!(
                "set index {set_index_raw} out of bounds (1..={family}) in {}",
                path.display()
            ));
        }
        let set_index = set_index_raw - 1;
        let mut members: Vec<usize> = members_part
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|e| format!("bad member {s} in {}: {e}", path.display()))
            })
            .collect::<Result<_, _>>()?;
        members.sort_unstable();
        members.dedup();
        let builder_ref = builder
            .as_mut()
            .expect("builder should be initialized after begin line");
        builder_ref.clear_set(set_index);
        if cardinality < 0 {
            let mut complement: Vec<usize> = (0..capacity).map(|i| i + 1).collect();
            complement.retain(|v| !members.contains(v));
            for member in complement {
                builder_ref.insert_into_set(set_index, RowId::new(member - 1));
            }
        } else {
            for member in members {
                if member == 0 || member > capacity {
                    return Err(format!(
                        "member {member} out of bounds for capacity {capacity} in {}",
                        path.display()
                    ));
                }
                builder_ref.insert_into_set(set_index, RowId::new(member - 1));
            }
        }
    }

    builder
        .map(|b| b.build())
        .ok_or_else(|| format!("no set family found in {}", path.display()))
}

fn collect_files_with_extensions(root: &Path, exts: &[&str]) -> Vec<PathBuf> {
    let mut stack = VecDeque::from([root.to_path_buf()]);
    let mut files = Vec::new();
    while let Some(dir) = stack.pop_front() {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push_back(path);
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                    && exts.iter().any(|needle| needle.eq_ignore_ascii_case(ext))
                {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}
