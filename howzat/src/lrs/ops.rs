use std::cmp::Ordering;

use super::tableau::Tableau;
use calculo::num::Int;

#[inline(always)]
fn cmp_product_sign<Z: Int>(a: &Z, b: &Z, c: &Z, d: &Z, scratch: &mut Z::CmpScratch) -> i32 {
    match Z::cmp_product(a, b, c, d, scratch) {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

pub(super) fn lex_ratio_test<Z: Int>(
    p: &Tableau<Z>,
    decision_vars: usize,
    col_loc: usize,
    min_ratio: &mut [usize],
    cmp_scratch: &mut Z::CmpScratch,
) -> usize {
    let m = p.constraint_count;
    let d = p.decision_vars;
    debug_assert!(min_ratio.len() > m, "min_ratio scratch must be length m+1");
    debug_assert!(col_loc <= d);

    let mut degencount = 0usize;
    let mut nondegenerate = true;

    for j in (decision_vars + 1)..=m {
        let row_loc = p.basic_rows[j] as usize;
        if p.coeff(row_loc, col_loc).is_negative() {
            min_ratio[degencount] = j;
            degencount += 1;
            if p.coeff(row_loc, 0).is_zero() {
                nondegenerate = false;
            }
        }
    }

    // We store a non-degenerate flag in the last slot to avoid an extra boolean argument.
    min_ratio[m] = if nondegenerate { 1 } else { 0 };

    if degencount == 0 {
        return 0;
    }

    let mut ratiocol = 0usize; // physical column being checked; initially RHS
    let mut start = 0usize; // starting location in min_ratio array
    let mut nstart = 0usize;
    let mut ndegencount = 0usize;

    let mut bindex = d + 1; // index of next basic variable to consider
    let mut cindex = 0usize; // index of next cobasic variable to consider
    let mut basicindex = d; // index of basis inverse column to check (except rhs test)

    let mut nmin = Z::zero();
    let mut dmin = Z::zero();

    while degencount > 1 {
        if bindex < p.basic_vars.len() && (p.basic_vars[bindex] as usize) == basicindex {
            if min_ratio[start] == bindex {
                start += 1;
                degencount -= 1;
            }
            bindex += 1;
        } else {
            let mut firstime = true;

            if basicindex != d {
                ratiocol = p.cobasic_cols[cindex] as usize;
                cindex += 1;
            }

            for jpos in start..(start + degencount) {
                let basis_pos = min_ratio[jpos];
                let row_loc = p.basic_rows[basis_pos] as usize;

                // comp: 1 => lhs > rhs, 0 => equal, -1 => lhs < rhs
                let mut comp = 1i32;
                if firstime {
                    firstime = false;
                } else {
                    let a_ir = p.coeff(row_loc, ratiocol);
                    if nmin.is_positive() || a_ir.is_negative() {
                        if nmin.is_negative() || a_ir.is_positive() {
                            comp = cmp_product_sign(
                                &nmin,
                                p.coeff(row_loc, col_loc),
                                a_ir,
                                &dmin,
                                cmp_scratch,
                            );
                        } else {
                            comp = -1;
                        }
                    } else if nmin.is_zero() && a_ir.is_zero() {
                        comp = 0;
                    }
                    if ratiocol == 0 {
                        comp = -comp;
                    }
                }

                if comp == 1 {
                    nstart = jpos;
                    Z::assign_from(&mut nmin, p.coeff(row_loc, ratiocol));
                    Z::assign_from(&mut dmin, p.coeff(row_loc, col_loc));
                    ndegencount = 1;
                } else if comp == 0 {
                    min_ratio[nstart + ndegencount] = basis_pos;
                    ndegencount += 1;
                }
            }

            degencount = ndegencount;
            start = nstart;
        }

        basicindex += 1;
    }

    min_ratio[start]
}

pub(super) fn choose_parent_pivot<Z: Int>(
    p: &Tableau<Z>,
    decision_vars: usize,
    min_ratio: &mut [usize],
    cmp_scratch: &mut Z::CmpScratch,
) -> Option<(usize, usize)> {
    let d = p.decision_vars;
    let mut cob = 0usize;
    while cob < d && !p.coeff(0, p.cobasic_cols[cob] as usize).is_positive() {
        cob += 1;
    }
    if cob >= d {
        return None;
    }

    let col_loc = p.cobasic_cols[cob] as usize;
    let bas = lex_ratio_test(p, decision_vars, col_loc, min_ratio, cmp_scratch);
    if bas == 0 {
        return None;
    }
    Some((bas, cob))
}

pub(super) fn is_reverse_pivot_child<Z: Int>(
    p: &Tableau<Z>,
    decision_vars: usize,
    cob: usize,
    min_ratio: &mut [usize],
    cmp_scratch: &mut Z::CmpScratch,
) -> Option<usize> {
    let m = p.constraint_count;
    let d = p.decision_vars;
    if cob >= p.cobasic_vars.len() {
        return None;
    }

    let col_loc = p.cobasic_cols[cob] as usize;
    if !p.coeff(0, col_loc).is_negative() {
        if min_ratio.len() > m {
            min_ratio[m] = 0;
        }
        return None;
    }

    let bas = lex_ratio_test(p, decision_vars, col_loc, min_ratio, cmp_scratch);
    if bas == 0 {
        if min_ratio.len() > m {
            min_ratio[m] = 0;
        }
        return None;
    }

    let row_loc = p.basic_rows[bas] as usize;

    for i in 0..d {
        if p.cobasic_vars[i] >= p.basic_vars[bas] {
            break;
        }
        if i == cob {
            continue;
        }

        let jcol = p.cobasic_cols[i] as usize;
        if (p.coeff(0, jcol).is_positive() || p.coeff(row_loc, jcol).is_negative())
            && ((!p.coeff(0, jcol).is_negative() && !p.coeff(row_loc, jcol).is_positive())
                || cmp_product_sign(
                    p.coeff(0, jcol),
                    p.coeff(row_loc, col_loc),
                    p.coeff(0, col_loc),
                    p.coeff(row_loc, jcol),
                    cmp_scratch,
                ) == -1)
        {
            if min_ratio.len() > m {
                min_ratio[m] = 0;
            }
            return None;
        }
    }

    Some(bas)
}

fn is_min_ratio<Z: Int>(
    p: &Tableau<Z>,
    r_row_loc: usize,
    s_col_loc: usize,
    cmp_scratch: &mut Z::CmpScratch,
) -> bool {
    for i in 1..=p.storage_row_count {
        if i == r_row_loc {
            continue;
        }
        if p.coeff(i, s_col_loc).is_negative()
            && cmp_product_sign(
                p.coeff(i, 0),
                p.coeff(r_row_loc, s_col_loc),
                p.coeff(i, s_col_loc),
                p.coeff(r_row_loc, 0),
                cmp_scratch,
            ) != 0
        {
            return false;
        }
    }
    true
}

pub(super) fn is_lex_min_basis<Z: Int>(
    p: &Tableau<Z>,
    decision_vars: usize,
    col_loc: usize,
) -> bool {
    let mut cmp_scratch = Z::CmpScratch::default();
    is_lex_min_basis_with_scratch(p, decision_vars, col_loc, &mut cmp_scratch)
}

pub(super) fn is_lex_min_basis_with_scratch<Z: Int>(
    p: &Tableau<Z>,
    decision_vars: usize,
    col_loc: usize,
    cmp_scratch: &mut Z::CmpScratch,
) -> bool {
    let m = p.constraint_count;
    let d = p.decision_vars;
    for i in (decision_vars + 1)..=m {
        let r = p.basic_rows[i] as usize;
        if p.coeff(r, col_loc).is_zero() {
            for j in 0..d {
                let s = p.cobasic_cols[j] as usize;
                if p.basic_vars[i] > p.cobasic_vars[j] {
                    if p.coeff(r, 0).is_zero() {
                        if !p.coeff(r, s).is_zero() {
                            return false;
                        }
                    } else if p.coeff(r, s).is_negative() && is_min_ratio(p, r, s, cmp_scratch) {
                        return false;
                    }
                }
            }
        }
    }
    true
}
