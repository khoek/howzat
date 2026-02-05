use super::ops::is_lex_min_basis;
use super::tableau::Tableau;
use super::{Options, Result};
use crate::matrix::LpMatrix;
use calculo::num::{Int, Rat};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{Representation, RepresentationKind, RowSet};

#[derive(Clone, Debug)]
pub(super) struct BasisSolutions<Z: Int> {
    pub vertex: Option<Vec<Z>>,
    pub rays: Vec<(usize, Vec<Z>)>,
}

pub(super) fn basis_solutions<Z: Int>(
    p: &Tableau<Z>,
    hull: bool,
    decision_vars: usize,
    emit_all_bases: bool,
) -> BasisSolutions<Z> {
    let d = p.decision_vars;

    let vertex = if hull {
        None
    } else if p.is_lex_min || emit_all_bases {
        Some(get_vertex(p))
    } else {
        None
    };

    let mut rays = Vec::new();
    for col_loc in 1..=d {
        if let Some(ray) = maybe_get_ray(p, hull, decision_vars, emit_all_bases, col_loc) {
            rays.push((col_loc, ray));
        }
    }

    BasisSolutions { vertex, rays }
}

fn tight_set_for_vertex<Z: Int>(p: &Tableau<Z>) -> RowSet {
    let m = p.constraint_count;
    let mut out = RowSet::new(m);

    for &var_u32 in p.cobasis_vars() {
        if let Some(idx) = p.constraint_of_var(var_u32 as usize) {
            out.insert(idx);
        }
    }

    // Degeneracy: additional constraints can be tight with a basic slack value of 0.
    for bas_pos in 1..=m {
        let var = p.basic_vars[bas_pos] as usize;
        let Some(idx) = p.constraint_of_var(var) else {
            continue;
        };
        let row_loc = p.basic_rows[bas_pos] as usize;
        if row_loc == 0 || row_loc > m {
            continue;
        }
        if p.coeff(row_loc, 0).is_zero() {
            out.insert(idx);
        }
    }

    out
}

fn tight_set_for_ray<Z: Int>(p: &Tableau<Z>, col_loc: usize) -> RowSet {
    let m = p.constraint_count;
    let d = p.decision_vars;
    let mut out = RowSet::new(m);

    // Determine which cobasic variable is entering for this ray column.
    let mut entering_var: Option<usize> = None;
    for cob_pos in 0..d {
        if p.cobasic_cols[cob_pos] as usize == col_loc {
            entering_var = Some(p.cobasic_vars[cob_pos] as usize);
            break;
        }
    }

    // Start from the origin vertex cobasis and drop the starred constraint (the entering one).
    for &var_u32 in p.cobasis_vars() {
        let var = var_u32 as usize;
        if Some(var) == entering_var {
            continue;
        }
        if let Some(idx) = p.constraint_of_var(var) {
            out.insert(idx);
        }
    }

    // Degeneracy: basic constraint variables can remain tight on the ray if both the origin value
    // and the direction coefficient are 0.
    for bas_pos in 1..=m {
        let var = p.basic_vars[bas_pos] as usize;
        let Some(idx) = p.constraint_of_var(var) else {
            continue;
        };
        let row_loc = p.basic_rows[bas_pos] as usize;
        if row_loc == 0 || row_loc > m {
            continue;
        }
        if !p.coeff(row_loc, 0).is_zero() {
            continue;
        }
        if p.coeff(row_loc, col_loc).is_zero() {
            out.insert(idx);
        }
    }

    out
}

fn get_vertex<Z: Int>(p: &Tableau<Z>) -> Vec<Z> {
    let n_out = p
        .original_decision_vars
        .checked_add(1)
        .expect("vertex output length overflow");
    let mut out = vec![Z::zero(); n_out];
    Z::assign_from(&mut out[0], &p.det);

    let mut i = 1usize;
    let mut ired = 0usize;
    for (ind, slot) in out.iter_mut().enumerate().skip(1) {
        if ired < p.redundant_columns.len() && p.redundant_columns[ired] == ind {
            *slot = Z::zero();
            ired += 1;
            continue;
        }
        let row_loc = p.basic_rows[i] as usize;
        Z::assign_from(slot, p.coeff(row_loc, 0));
        i += 1;
    }
    out
}

fn maybe_get_ray<Z: Int>(
    p: &Tableau<Z>,
    hull: bool,
    decision_vars: usize,
    emit_all_bases: bool,
    col_loc: usize,
) -> Option<Vec<Z>> {
    let m = p.constraint_count;
    let d = p.decision_vars;
    debug_assert!(col_loc > 0 && col_loc <= d);

    if !p.coeff(0, col_loc).is_negative() {
        return None;
    }

    let mut j = decision_vars + 1;
    while j <= m && !p.coeff(p.basic_rows[j] as usize, col_loc).is_negative() {
        j += 1;
    }
    if j <= m {
        return None;
    }

    if !emit_all_bases && !is_lex_min_basis(p, decision_vars, col_loc) {
        return None;
    }

    Some(get_ray(p, hull, col_loc))
}

fn get_ray<Z: Int>(p: &Tableau<Z>, hull: bool, col_loc: usize) -> Vec<Z> {
    let n_out = if hull {
        p.original_decision_vars
    } else {
        p.original_decision_vars
            .checked_add(1)
            .expect("ray output length overflow")
    };

    let mut out = vec![Z::zero(); n_out];
    let mut i = 1usize;
    let mut ired = 0usize;
    for (ind, slot) in out.iter_mut().enumerate() {
        if ind == 0 && !hull {
            *slot = Z::zero();
            continue;
        }
        if ired < p.redundant_columns.len() && p.redundant_columns[ired] == ind {
            *slot = Z::zero();
            ired += 1;
            continue;
        }
        let row_loc = p.basic_rows[i] as usize;
        Z::assign_from(slot, p.coeff(row_loc, col_loc));
        i += 1;
    }
    out
}

impl<Q, NRep> LpMatrix<Q, NRep>
where
    Q: Rat,
    NRep: Representation,
{
    pub fn lrs_enumerate_rows(self, options: Options) -> Result<(usize, usize, Vec<Q>)> {
        let hull = NRep::KIND == RepresentationKind::Generator;
        let col_count = self.col_count();
        let mut traversal = self.lrs_traversal(options.clone())?;
        let decision_vars = traversal.tableau().decision_vars;

        let mut data: Vec<Q> = Vec::new();
        let mut row_count = 0usize;
        loop {
            let sols = basis_solutions(
                traversal.tableau(),
                hull,
                decision_vars,
                options.emit_all_bases,
            );
            if let Some(mut vertex) = sols.vertex {
                if vertex.len() != col_count {
                    return Err(super::Error::InvariantViolation);
                }
                reduce_int_row_in_place(&mut vertex)?;
                push_vertex_to_rat::<Q>(&mut data, vertex)?;
                row_count += 1;
            }
            for (_col_loc, mut ray) in sols.rays {
                if ray.len() != col_count {
                    return Err(super::Error::InvariantViolation);
                }
                reduce_int_row_in_place(&mut ray)?;
                push_int_row_to_rat::<Q>(&mut data, ray);
                row_count += 1;
            }
            if !traversal.advance()? {
                break;
            }
        }
        Ok((row_count, col_count, data))
    }

    pub fn lrs_enumerate_rows_with_incidence(
        self,
        options: Options,
    ) -> Result<(usize, usize, Vec<Q>, SetFamily)> {
        let hull = NRep::KIND == RepresentationKind::Generator;
        let col_count = self.col_count();
        let mut traversal = self.lrs_traversal(options.clone())?;
        let decision_vars = traversal.tableau().decision_vars;
        let constraint_count = traversal.tableau().constraint_count;

        let mut data: Vec<Q> = Vec::new();
        let mut incidence_sets: Vec<RowSet> = Vec::new();
        let mut row_count = 0usize;
        loop {
            let sols = basis_solutions(
                traversal.tableau(),
                hull,
                decision_vars,
                options.emit_all_bases,
            );

            if let Some(mut vertex) = sols.vertex {
                if vertex.len() != col_count {
                    return Err(super::Error::InvariantViolation);
                }
                reduce_int_row_in_place(&mut vertex)?;
                push_vertex_to_rat::<Q>(&mut data, vertex)?;
                row_count += 1;
                incidence_sets.push(tight_set_for_vertex(traversal.tableau()));
            }
            for (col_loc, mut ray) in sols.rays {
                if ray.len() != col_count {
                    return Err(super::Error::InvariantViolation);
                }
                reduce_int_row_in_place(&mut ray)?;
                push_int_row_to_rat::<Q>(&mut data, ray);
                row_count += 1;
                incidence_sets.push(tight_set_for_ray(traversal.tableau(), col_loc));
            }
            if !traversal.advance()? {
                break;
            }
        }

        Ok((
            row_count,
            col_count,
            data,
            SetFamily::from_sets(constraint_count, incidence_sets),
        ))
    }
}

fn push_int_row_to_rat<Q: Rat>(out: &mut Vec<Q>, row: Vec<Q::Int>) {
    out.reserve(row.len());
    for numer in row {
        out.push(Q::from_frac(numer, Q::Int::one()));
    }
}

fn push_vertex_to_rat<Q: Rat>(out: &mut Vec<Q>, vertex: Vec<Q::Int>) -> Result<()> {
    if vertex.is_empty() {
        return Ok(());
    }

    let mut it = vertex.into_iter();
    let det = it.next().expect("vertex vector must be non-empty");
    if det.is_zero() {
        let mut row = Vec::with_capacity(it.len() + 1);
        row.push(det);
        row.extend(it);
        push_int_row_to_rat::<Q>(out, row);
        return Ok(());
    }

    out.reserve(it.len() + 1);
    out.push(Q::one());

    let rem = it.len();
    let mut det_opt = Some(det);
    for (idx, numer) in it.enumerate() {
        let denom = if idx + 1 == rem {
            det_opt
                .take()
                .expect("det must be available for last coordinate")
        } else {
            det_opt
                .as_ref()
                .expect("det must be available for non-last coordinates")
                .clone()
        };
        out.push(Q::from_frac(numer, denom));
    }
    Ok(())
}

fn reduce_int_row_in_place<Z: Int>(row: &mut [Z]) -> Result<()> {
    let mut i = 0usize;
    while i < row.len() && row[i].is_zero() {
        i += 1;
    }
    if i == row.len() {
        return Ok(());
    }

    let one = Z::one();
    let mut divisor = row[i].clone();
    divisor.abs_mut()?;
    i += 1;

    let mut abs = Z::zero();
    for v in &row[i..] {
        if v.is_zero() {
            continue;
        }
        Z::assign_from(&mut abs, v);
        abs.abs_mut()?;
        divisor.gcd_assign(&abs)?;
        if divisor == one {
            return Ok(());
        }
    }

    if divisor > one {
        for v in row.iter_mut() {
            if v.is_zero() {
                continue;
            }
            v.div_assign_exact(&divisor)?;
        }
    }
    Ok(())
}
