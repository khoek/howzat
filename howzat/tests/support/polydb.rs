use anyhow::{Context, Result, anyhow};
use howzat::matrix::{LpMatrix, LpMatrixBuilder};
use hullabaloo::types::{Generator, Inequality, RowSet};
use rand::Rng;
use rand::prelude::IndexedRandom;
use rand::rng;
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::{Value, json};
use std::env;

pub const DEFAULT_BASE: &str = "https://polydb.org/rest/current";
pub const MAX_LIMIT_DEFAULT: usize = 1000;

pub use crate::common::ParsedMatrix;

#[derive(Debug, Serialize)]
pub struct PolydbSample {
    pub base_url: String,
    pub chosen_collection: String,
    pub sampled: usize,
    pub skip: usize,
    pub total_in_collection: usize,
    pub items: Vec<ItemSummary>,
}

#[derive(Debug, Serialize)]
pub struct ItemSummary {
    pub id: Option<String>,
    pub dim: Option<usize>,
    pub definition: Option<Definition>,
    pub stats: Stats,
    pub polydb_hints: PolyDbHints,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
pub enum Definition {
    HRep {
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        a_eq: Vec<Vec<f64>>,
        b_eq: Vec<f64>,
    },
    VRep {
        vertices: Vec<Vec<f64>>,
        rays: Vec<Vec<f64>>,
        lines: Vec<Vec<f64>>,
    },
}

#[derive(Debug, Default, Serialize)]
pub struct Stats {
    pub n_vertices: Option<usize>,
    pub n_inequalities: Option<usize>,
    pub n_equalities: Option<usize>,
    pub n_rays: Option<usize>,
    pub n_lines: Option<usize>,
    pub bounded: Option<bool>,
    pub affine_rank_vertices: Option<usize>,
    pub f_vector: Option<Vec<i64>>,
}

#[derive(Debug, Default, Serialize)]
pub struct PolyDbHints {
    pub n_vertices: Option<usize>,
    pub n_facets: Option<usize>,
    pub dim: Option<usize>,
    pub simple: Option<bool>,
    pub simplicial: Option<bool>,
}

pub fn fetch_random_polytopes(max_limit: usize) -> Result<PolydbSample> {
    let limit = if max_limit == 0 {
        MAX_LIMIT_DEFAULT
    } else {
        max_limit
    };
    let base = env::var("POLYDB_BASE").unwrap_or_else(|_| DEFAULT_BASE.to_string());
    fetch_from_base(&base, limit)
}

pub fn fetch_from_base(base: &str, max_limit: usize) -> Result<PolydbSample> {
    let client = Client::builder().build()?;
    let mut collections = Vec::new();
    for sec in &[
        "Polytopes.Combinatorial",
        "Polytopes.Geometric",
        "Polytopes.Lattice",
    ] {
        collections.extend(list_collections(&client, base, sec)?);
    }
    if collections.is_empty() {
        return Err(anyhow!("No collections found under Polytopes.*"));
    }
    let query = definition_query();
    let mut rng = rng();
    let chosen = collections
        .choose(&mut rng)
        .ok_or_else(|| anyhow!("random choice failed"))?
        .to_string();

    let total = count_docs(&client, base, &chosen, &query)?;
    let limit = std::cmp::min(total, max_limit);
    let skip = if total > limit {
        rng.random_range(0..=(total - limit))
    } else {
        0
    };

    let docs = find_docs(&client, base, &chosen, &query, limit, skip)?;
    let items = docs
        .into_iter()
        .map(|v| extract_one(&v))
        .collect::<Result<Vec<_>>>()?;

    Ok(PolydbSample {
        base_url: base.to_string(),
        chosen_collection: chosen,
        sampled: limit,
        skip,
        total_in_collection: total,
        items,
    })
}

pub fn item_to_matrix(item: &ItemSummary) -> Result<ParsedMatrix> {
    let def = item
        .definition
        .as_ref()
        .ok_or_else(|| anyhow!("missing polytope definition for {:?}", item.id))?;
    definition_to_matrix(def, item.dim)
}

pub fn definition_to_matrix(def: &Definition, dim_hint: Option<usize>) -> Result<ParsedMatrix> {
    match def {
        Definition::HRep { a, b, a_eq, b_eq } => {
            hrep_to_matrix(a, b, a_eq, b_eq, dim_hint).map(ParsedMatrix::Inequality)
        }
        Definition::VRep {
            vertices,
            rays,
            lines,
        } => vrep_to_matrix(vertices, rays, lines, dim_hint).map(ParsedMatrix::Generator),
    }
}

fn hrep_to_matrix(
    a: &[Vec<f64>],
    b: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    dim_hint: Option<usize>,
) -> Result<LpMatrix<f64, Inequality>> {
    let dim = a
        .first()
        .map(|r| r.len())
        .or_else(|| a_eq.first().map(|r| r.len()))
        .or(dim_hint)
        .ok_or_else(|| anyhow!("cannot infer dimension for H-rep polytope"))?;
    if a.len() != b.len() {
        return Err(anyhow!(
            "inequality coefficient count {} does not match offsets {}",
            a.len(),
            b.len()
        ));
    }
    if a_eq.len() != b_eq.len() {
        return Err(anyhow!(
            "equality coefficient count {} does not match offsets {}",
            a_eq.len(),
            b_eq.len()
        ));
    }

    let row_count = a.len() + a_eq.len();
    let mut rows_data = vec![vec![0.0; dim + 1]; row_count];
    let mut linearity = RowSet::new(row_count);

    for (i, coeffs) in a.iter().enumerate() {
        if coeffs.len() != dim {
            return Err(anyhow!(
                "inequality {} has dimension {}, expected {}",
                i,
                coeffs.len(),
                dim
            ));
        }
        rows_data[i][0] = b[i];
        for (j, val) in coeffs.iter().enumerate() {
            rows_data[i][j + 1] = -val;
        }
    }

    for (eq_idx, coeffs) in a_eq.iter().enumerate() {
        if coeffs.len() != dim {
            return Err(anyhow!(
                "equality {} has dimension {}, expected {}",
                eq_idx,
                coeffs.len(),
                dim
            ));
        }
        let row = a.len() + eq_idx;
        rows_data[row][0] = b_eq[eq_idx];
        for (j, val) in coeffs.iter().enumerate() {
            rows_data[row][j + 1] = -val;
        }
        linearity.insert(row + 1);
    }

    Ok(LpMatrixBuilder::from_rows(rows_data)
        .with_linearity(linearity)
        .build())
}

fn vrep_to_matrix(
    vertices: &[Vec<f64>],
    rays: &[Vec<f64>],
    lines: &[Vec<f64>],
    dim_hint: Option<usize>,
) -> Result<LpMatrix<f64, Generator>> {
    let dim = vertices
        .first()
        .map(|v| v.len())
        .or_else(|| rays.first().map(|v| v.len()))
        .or_else(|| lines.first().map(|v| v.len()))
        .or(dim_hint)
        .ok_or_else(|| anyhow!("cannot infer dimension for V-rep polytope"))?;
    let rows = vertices.len() + rays.len() + lines.len();
    let mut data = vec![vec![0.0; dim + 1]; rows];
    let mut linearity = RowSet::new(rows);

    let mut row_idx = 0;
    for v in vertices {
        if v.len() != dim {
            return Err(anyhow!(
                "vertex {} has dimension {}, expected {}",
                row_idx,
                v.len(),
                dim
            ));
        }
        data[row_idx][0] = 1.0;
        for (j, val) in v.iter().enumerate() {
            data[row_idx][j + 1] = *val;
        }
        row_idx += 1;
    }
    for r in rays {
        if r.len() != dim {
            return Err(anyhow!(
                "ray {} has dimension {}, expected {}",
                row_idx,
                r.len(),
                dim
            ));
        }
        for (j, val) in r.iter().enumerate() {
            data[row_idx][j + 1] = *val;
        }
        row_idx += 1;
    }
    for l in lines {
        if l.len() != dim {
            return Err(anyhow!(
                "line {} has dimension {}, expected {}",
                row_idx,
                l.len(),
                dim
            ));
        }
        for (j, val) in l.iter().enumerate() {
            data[row_idx][j + 1] = *val;
        }
        linearity.insert(row_idx + 1);
        row_idx += 1;
    }

    Ok(LpMatrixBuilder::<f64, Generator>::from_rows(data)
        .with_linearity(linearity)
        .build())
}

fn list_collections(client: &Client, base: &str, section: &str) -> Result<Vec<String>> {
    let url = format!("{}/collections/{}", base, section);
    let v: Value = client.get(&url).send()?.error_for_status()?.json()?;
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow!("collections response not an array"))?;
    Ok(arr
        .iter()
        .filter_map(|x| x.as_str().map(|s| s.to_string()))
        .collect())
}

fn count_docs(client: &Client, base: &str, collection: &str, query: &Value) -> Result<usize> {
    let url = format!("{}/count", base);
    let resp = client
        .get(&url)
        .query(&[
            ("collection", collection),
            ("query", &serde_json::to_string(query)?),
        ])
        .send()?
        .error_for_status()?
        .text()?;
    let trimmed = resp.trim().trim_matches('`').trim();
    trimmed
        .parse::<usize>()
        .with_context(|| format!("failed to parse count from '{}'", resp))
}

fn find_docs(
    client: &Client,
    base: &str,
    collection: &str,
    query: &Value,
    limit: usize,
    skip: usize,
) -> Result<Vec<Value>> {
    let url = format!("{}/find", base);
    let v: Value = client
        .get(&url)
        .query(&[
            ("collection", collection),
            ("query", &serde_json::to_string(query)?),
            ("limit", &limit.to_string()),
            ("skip", &skip.to_string()),
        ])
        .send()?
        .error_for_status()?
        .json()?;
    Ok(v.as_array().cloned().unwrap_or_default())
}

fn definition_query() -> Value {
    json!({
        "$or": [
            { "VERTICES": { "$exists": true } },
            { "RAYS": { "$exists": true } },
            { "FACETS": { "$exists": true } },
            { "INEQUALITIES": { "$exists": true } }
        ]
    })
}

fn extract_one(obj: &Value) -> Result<ItemSummary> {
    let id = obj
        .get("_id")
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());

    let polydb_hints = PolyDbHints {
        n_vertices: obj.get("N_VERTICES").and_then(as_usize),
        n_facets: obj.get("N_FACETS").and_then(as_usize),
        dim: obj.get("DIM").and_then(as_usize),
        simple: obj.get("SIMPLE").and_then(|v| v.as_bool()),
        simplicial: obj.get("SIMPLICIAL").and_then(|v| v.as_bool()),
    };

    let mut def: Option<Definition> = None;

    let vertices = obj.get("VERTICES").and_then(as_matrix);
    let rays = obj.get("RAYS").and_then(as_matrix).unwrap_or_default();
    let lines = obj.get("LINES").and_then(as_matrix).unwrap_or_default();

    let facets = obj
        .get("FACETS")
        .or_else(|| obj.get("INEQUALITIES"))
        .and_then(as_matrix);
    let eqs = obj
        .get("AFFINE_HULL")
        .or_else(|| obj.get("EQUATIONS"))
        .and_then(as_matrix)
        .unwrap_or_default();

    if let Some(vs_hom) = vertices.clone() {
        let mut vs = Vec::with_capacity(vs_hom.len());
        for row in vs_hom {
            if row.is_empty() {
                continue;
            }
            let w = row[0];
            if approx_eq(w, 0.0) {
                continue;
            }
            let x = row[1..].iter().map(|t| *t / w).collect::<Vec<_>>();
            vs.push(x);
        }

        def = Some(Definition::VRep {
            vertices: vs,
            rays: drop_leading_zero(&rays),
            lines: drop_leading_zero(&lines),
        });
    }

    if def.is_none()
        && let Some(facet_rows) = facets.clone()
    {
        let mut a = Vec::with_capacity(facet_rows.len());
        let mut b = Vec::with_capacity(facet_rows.len());
        for row in facet_rows {
            if row.is_empty() {
                continue;
            }
            b.push(row[0]);
            a.push(row[1..].iter().map(|t| -(*t)).collect());
        }

        let (a_eq, b_eq) = if !eqs.is_empty() {
            let mut ae = Vec::with_capacity(eqs.len());
            let mut be = Vec::with_capacity(eqs.len());
            for row in &eqs {
                if row.is_empty() {
                    continue;
                }
                ae.push(row[1..].to_vec());
                be.push(-row[0]);
            }
            (ae, be)
        } else {
            (Vec::new(), Vec::new())
        };

        def = Some(Definition::HRep { a, b, a_eq, b_eq });
    }

    let stats = Stats {
        f_vector: obj.get("F_VECTOR").and_then(as_int_vector),
        ..Default::default()
    };

    let dim = polydb_hints.dim.or_else(|| match &def {
        Some(Definition::VRep { vertices, .. }) => {
            vertices.first().map(|x| x.len()).or(Some(0usize))
        }
        Some(Definition::HRep { a, .. }) => a.first().map(|row| row.len()),
        None => None,
    });

    Ok(ItemSummary {
        id,
        dim,
        definition: def,
        stats,
        polydb_hints,
    })
}

fn as_matrix(v: &Value) -> Option<Vec<Vec<f64>>> {
    if let Some(arr) = v.as_array() {
        let mut out = Vec::with_capacity(arr.len());
        for row in arr {
            let row_arr = row.as_array()?;
            let mut r = Vec::with_capacity(row_arr.len());
            for x in row_arr {
                r.push(as_f64(x)?);
            }
            out.push(r);
        }
        return Some(out);
    }
    if let Some(obj) = v.as_object()
        && let (Some(rows), Some(cols), Some(data)) =
            (obj.get("rows"), obj.get("cols"), obj.get("data"))
    {
        let r = rows.as_u64()? as usize;
        let c = cols.as_u64()? as usize;
        let flat = data.as_array()?;
        if flat.len() != r * c {
            return None;
        }
        let mut out = vec![vec![0.0; c]; r];
        for i in 0..r {
            for j in 0..c {
                out[i][j] = as_f64(&flat[i * c + j])?;
            }
        }
        return Some(out);
    }
    None
}

fn as_int_vector(v: &Value) -> Option<Vec<i64>> {
    let arr = v.as_array()?;
    let mut res = Vec::with_capacity(arr.len());
    for x in arr {
        if let Some(n) = x.as_i64() {
            res.push(n);
        } else if let Some(s) = x.as_str() {
            let parsed = if let Some((p, q)) = s.split_once('/') {
                let p: f64 = p.parse().ok()?;
                let q: f64 = q.parse().ok()?;
                (p / q).round() as i64
            } else {
                s.parse::<i64>().ok()?
            };
            res.push(parsed);
        } else if let Some(n) = x.as_f64() {
            res.push(n.round() as i64);
        } else {
            return None;
        }
    }
    Some(res)
}

fn as_usize(v: &Value) -> Option<usize> {
    v.as_u64().map(|u| u as usize).or_else(|| {
        v.as_str()
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| v.as_f64().map(|f| f as usize))
    })
}

fn as_f64(v: &Value) -> Option<f64> {
    if let Some(x) = v.as_f64() {
        return Some(x);
    }
    if let Some(s) = v.as_str() {
        if let Some((p, q)) = s.split_once('/') {
            let p: f64 = p.parse().ok()?;
            let q: f64 = q.parse().ok()?;
            return Some(p / q);
        }
        return s.parse::<f64>().ok();
    }
    if let Some(n) = v.as_i64() {
        return Some(n as f64);
    }
    if let Some(u) = v.as_u64() {
        return Some(u as f64);
    }
    None
}

pub fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs() + b.abs())
}

fn drop_leading_zero(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter()
        .filter(|r| !r.is_empty())
        .map(|r| r[1..].to_vec())
        .collect()
}

pub fn affine_rank(points: &[Vec<f64>]) -> usize {
    if points.is_empty() {
        return 0;
    }
    let d = points[0].len();
    if d == 0 {
        return 0;
    }
    let p0 = &points[0];
    let mut m: Vec<Vec<f64>> = points
        .iter()
        .skip(1)
        .map(|p| p.iter().zip(p0).map(|(x, y)| x - y).collect())
        .collect();
    row_rank(&mut m)
}

fn row_rank(m: &mut [Vec<f64>]) -> usize {
    let mut rank = 0usize;
    let nrows = m.len();
    if nrows == 0 {
        return 0;
    }
    let ncols = m[0].len();
    let mut col = 0usize;
    for r in 0..nrows {
        if col >= ncols {
            break;
        }
        let mut pivot = r;
        for i in r..nrows {
            if m[i][col].abs() > m[pivot][col].abs() {
                pivot = i;
            }
        }
        if approx_eq(m[pivot][col], 0.0) {
            col += 1;
            if col >= ncols {
                break;
            }
            continue;
        }
        m.swap(r, pivot);
        let pivot_val = m[r][col];
        for x in &mut m[r][col..] {
            *x /= pivot_val;
        }
        for i in 0..nrows {
            if i == r {
                continue;
            }
            let factor = m[i][col];
            if approx_eq(factor, 0.0) {
                continue;
            }
            let (pivot_row, target_row) = if i < r {
                let (left, right) = m.split_at_mut(r);
                (&right[0], &mut left[i])
            } else {
                let (left, right) = m.split_at_mut(i);
                (&left[r], &mut right[0])
            };
            for (x, p) in target_row[col..].iter_mut().zip(&pivot_row[col..]) {
                *x -= factor * *p;
            }
        }
        rank += 1;
        col += 1;
        if col >= ncols {
            break;
        }
    }
    rank
}
