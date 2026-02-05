use crate::matrix::LpMatrix;
use calculo::num::{Int, Rat};
use hullabaloo::types::{Representation, RepresentationKind};

use super::enumerator::Traversal;
use super::tableau::Tableau;
use super::{Error, Options, Result, Start};

impl<Q, NRep> LpMatrix<Q, NRep>
where
    Q: Rat,
    NRep: Representation,
{
    /// Build an LRS reverse-search enumerator from an in-memory (exact) matrix.
    pub fn lrs_traversal(self, options: Options) -> Result<Traversal<Q::Int>> {
        let start = options.start.clone();
        let mut tableau = build_root_tableau_from_matrix(self)?;
        match start {
            Start::Root => Ok(Traversal::new(tableau, options)),
            Start::Cobasis { cobasis } => {
                tableau.pivot_to_cobasis(&cobasis)?;
                Ok(Traversal::new(tableau, options))
            }
            Start::Checkpoint(checkpoint) => {
                tableau.pivot_to_cobasis(&checkpoint.cobasis)?;
                tableau.depth = checkpoint.depth;
                let mut traversal = Traversal::new(tableau, options);
                traversal.apply_checkpoint_cursor(&checkpoint)?;
                Ok(traversal)
            }
        }
    }
}

/// Construct the initial (root) tableau for the traversal.
fn build_root_tableau_from_matrix<Q, NRep>(matrix: LpMatrix<Q, NRep>) -> Result<Tableau<Q::Int>>
where
    Q: Rat,
    NRep: Representation,
{
    let hull = NRep::KIND == RepresentationKind::Generator;
    let m = matrix.row_count();
    let n = matrix.col_count();
    if n == 0 {
        return Err(Error::DimensionTooLarge);
    }

    let d = if hull {
        n
    } else {
        n.checked_sub(1).ok_or(Error::DimensionTooLarge)?
    };

    let mut p = Tableau::<Q::Int>::new(m, d)?;

    for j in 0..=d {
        *p.coeff_mut(0, j) = Q::Int::zero();
    }

    // Load + scale input rows into integer dictionary rows 1..=m.
    let (storage, matrix_linearity) = matrix.into_matrix_and_linearity();
    let rows = storage.row_count();
    let cols = storage.col_count();
    let mut it = storage.into_data().into_iter();
    for row_idx in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            let v = it.next().ok_or(Error::InvariantViolation)?;
            row.push(v);
        }

        let scaled = if hull {
            scaled_row_hull::<Q>(row)?
        } else {
            scaled_row_hrep::<Q>(row)?
        };
        debug_assert_eq!(scaled.len(), d + 1);
        let dict_row = row_idx + 1;
        for (j, v) in scaled.into_iter().enumerate() {
            *p.coeff_mut(dict_row, j) = v;
        }
    }
    debug_assert!(it.next().is_none(), "matrix storage length mismatch");

    // Process explicit linearities first, then remaining constraints in reverse order.
    let mut linearity: Vec<usize> = matrix_linearity
        .iter()
        .map(|r| r.as_index() + 1) // dictionary rows are 1-based (row 0 is the cost row)
        .collect();
    linearity.sort_unstable();

    let mut is_lin = vec![false; m + 1];
    for &r in &linearity {
        if r <= m {
            is_lin[r] = true;
        }
    }

    let mut order = Vec::with_capacity(m);
    order.extend_from_slice(&linearity);
    for i in (1..=m).rev() {
        if !is_lin[i] {
            order.push(i);
        }
    }

    pivot_to_initial_basis(&mut p, &order, &mut linearity)?;

    linearity.retain(|&r| r != 0);
    remove_explicit_linearities_from_cobasis(&mut p, &linearity)?;

    p.redundant_columns = find_redundant_columns(&p, hull)?;
    let nredund = p.redundant_columns.len();
    for _ in 0..nredund {
        p.remove_cobasis_pos(0)?;
    }

    let decision_vars = p.decision_vars;
    make_primal_feasible(&mut p, decision_vars)?;

    for j in 1..=p.decision_vars {
        let mut v = p.det.clone();
        v.neg_mut()?;
        *p.coeff_mut(0, j) = v;
    }
    *p.coeff_mut(0, 0) = Q::Int::zero();

    normalize_var_order(&mut p)?;

    Ok(p)
}

fn scaled_row_hrep<Q: Rat>(row: Vec<Q>) -> Result<Vec<Q::Int>> {
    let mut nums = Vec::with_capacity(row.len());
    let mut dens = Vec::with_capacity(row.len());
    for v in row {
        let (n, d) = v.into_parts();
        nums.push(n);
        dens.push(d);
    }
    scale_nums_dens::<Q::Int>(nums, dens, /*skip_prefix=*/ 0)
}

fn scaled_row_hull<Q: Rat>(row: Vec<Q>) -> Result<Vec<Q::Int>> {
    let mut nums = Vec::with_capacity(row.len() + 1);
    let mut dens = Vec::with_capacity(row.len() + 1);
    nums.push(Q::Int::zero());
    dens.push(Q::Int::one());
    for v in row {
        let (n, d) = v.into_parts();
        nums.push(n);
        dens.push(d);
    }
    scale_nums_dens::<Q::Int>(nums, dens, /*skip_prefix=*/ 1)
}

fn scale_nums_dens<I: Int>(mut nums: Vec<I>, dens: Vec<I>, skip_prefix: usize) -> Result<Vec<I>> {
    if nums.len() != dens.len() {
        return Err(Error::DimensionTooLarge);
    }

    let one = I::one();
    let mut lcm = I::one();
    let mut gcd = I::zero();

    let mut abs_n = I::zero();
    for (n, d) in nums.iter().zip(dens.iter()).skip(skip_prefix) {
        lcm.lcm_assign(d)?;

        I::assign_from(&mut abs_n, n);
        abs_n.abs_mut()?;
        gcd.gcd_assign(&abs_n)?;
    }

    let needs_scale = gcd > one || lcm > one;
    if !needs_scale {
        return Ok(nums);
    }

    for (n, d) in nums.iter_mut().zip(dens.iter()) {
        if gcd > one {
            n.div_assign_exact(&gcd)?;
        }
        if lcm > one {
            n.mul_assign(&lcm)?;
        }
        if d != &one {
            n.div_assign_exact(d)?;
        }
    }
    Ok(nums)
}

fn pivot_to_initial_basis<Z: Int>(
    p: &mut Tableau<Z>,
    order: &[usize],
    linearity: &mut [usize],
) -> Result<()> {
    let m = p.constraint_count;
    let d = p.decision_vars;
    let nlinearity = linearity.len();

    let mut pivot_scratch = Z::PivotScratch::default();
    for (pos, &ord) in order.iter().take(m).enumerate() {
        let mut bas = 0usize;
        let target = d.checked_add(ord).ok_or(Error::DimensionTooLarge)?;
        while bas <= m && (p.basic_vars[bas] as usize) != target {
            bas += 1;
        }
        if bas > m {
            if pos < nlinearity {
                return Err(Error::Infeasible);
            }
            continue;
        }

        let mut cob = 0usize;
        while cob < d {
            if (p.cobasic_vars[cob] as usize) > d {
                break;
            }
            let row_loc = p.basic_rows[bas] as usize;
            let col_loc = p.cobasic_cols[cob] as usize;
            if !p.coeff(row_loc, col_loc).is_zero() {
                break;
            }
            cob += 1;
        }

        if cob < d && (p.cobasic_vars[cob] as usize) <= d {
            p.pivot_with_scratch(bas, cob, &mut pivot_scratch)?;
        } else if pos < nlinearity {
            let row_loc = p.basic_rows[bas] as usize;
            if p.coeff(row_loc, 0).is_zero() {
                linearity[pos] = 0;
            } else {
                return Err(Error::Infeasible);
            }
        }
    }

    Ok(())
}

fn remove_explicit_linearities_from_cobasis<Z: Int>(
    p: &mut Tableau<Z>,
    linearity: &[usize],
) -> Result<()> {
    for &lin_row in linearity {
        let target_var: u32 = p
            .decision_vars
            .checked_add(lin_row)
            .and_then(|v| v.try_into().ok())
            .ok_or(Error::DimensionTooLarge)?;

        let mut cob = 0usize;
        while cob < p.decision_vars && p.cobasic_vars[cob] != target_var {
            cob += 1;
        }
        if cob >= p.decision_vars {
            return Err(Error::InvariantViolation);
        }
        p.remove_cobasis_pos(cob)?;
    }
    Ok(())
}

fn find_redundant_columns<Z: Int>(p: &Tableau<Z>, hull: bool) -> Result<Vec<usize>> {
    let d = p.decision_vars;
    let hull_offset = if hull { 1usize } else { 0usize };

    let mut redund: Vec<usize> = Vec::new();
    let mut cob = 0usize;
    while cob < d && (p.cobasic_vars[cob] as usize) <= d {
        let idx = p.cobasic_vars[cob] as usize;
        if idx < hull_offset {
            return Err(Error::InvariantViolation);
        }
        redund.push(idx - hull_offset);
        cob += 1;
    }
    Ok(redund)
}

fn make_primal_feasible<Z: Int>(p: &mut Tableau<Z>, decision_vars: usize) -> Result<()> {
    let m = p.constraint_count;
    let d = p.decision_vars;

    let mut pivot_scratch = Z::PivotScratch::default();
    loop {
        let mut bas = decision_vars + 1;
        while bas <= m {
            let row_loc = p.basic_rows[bas] as usize;
            if p.coeff(row_loc, 0).is_negative() {
                break;
            }
            bas += 1;
        }

        if bas > m {
            return Ok(());
        }

        let mut cob = 0usize;
        while cob < d {
            let row_loc = p.basic_rows[bas] as usize;
            let col_loc = p.cobasic_cols[cob] as usize;
            if p.coeff(row_loc, col_loc).is_positive() {
                break;
            }
            cob += 1;
        }
        if cob >= d {
            return Err(Error::Infeasible);
        }

        p.pivot_with_scratch(bas, cob, &mut pivot_scratch)?;
    }
}

fn normalize_var_order<Z: Int>(p: &mut Tableau<Z>) -> Result<()> {
    let m = p.constraint_count;
    let d = p.decision_vars;

    while (p.cobasic_vars[0] as usize) <= m {
        let i = p.cobasic_vars[0] as usize;
        if i > m {
            break;
        }
        let swap_with = p.basic_vars[i] as usize;
        if i < p.var_to_constraint.len() && swap_with < p.var_to_constraint.len() {
            p.var_to_constraint.swap(i, swap_with);
        }
        p.cobasic_vars[0] = p.basic_vars[i];
        p.basic_vars[i] = i.try_into().map_err(|_| Error::DimensionTooLarge)?;
        super::util::repair_sorted_pair_u32(&mut p.cobasic_vars, &mut p.cobasic_cols, 0, d);
    }
    Ok(())
}
