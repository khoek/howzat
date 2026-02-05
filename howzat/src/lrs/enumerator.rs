use std::collections::VecDeque;

use super::ops::{choose_parent_pivot, is_lex_min_basis_with_scratch, is_reverse_pivot_child};
use super::tableau::Tableau;
use super::{Checkpoint, Cursor, Error, Options, Result};
use calculo::num::Int;

#[derive(Clone, Debug)]
struct CacheEntry<Z: Int> {
    tableau: Tableau<Z>,
    resume_scan_pos: usize,
}

#[derive(Debug)]
pub struct Traversal<Z: Int> {
    tableau: Tableau<Z>,
    options: Options,
    min_ratio_scratch: Vec<usize>,
    cmp_scratch: Z::CmpScratch,
    pivot_scratch: Z::PivotScratch,
    cache: VecDeque<CacheEntry<Z>>,
    cursor: Cursor,
}

impl<Z: Int> Clone for Traversal<Z> {
    fn clone(&self) -> Self {
        Self {
            tableau: self.tableau.clone(),
            options: self.options.clone(),
            min_ratio_scratch: self.min_ratio_scratch.clone(),
            cmp_scratch: Z::CmpScratch::default(),
            pivot_scratch: Z::PivotScratch::default(),
            cache: self.cache.clone(),
            cursor: self.cursor,
        }
    }
}

impl<Z: Int> Traversal<Z> {
    pub(super) fn new(mut tableau: Tableau<Z>, options: Options) -> Self {
        let min_ratio_scratch = vec![0usize; tableau.constraint_count + 1];
        let d = tableau.decision_vars;
        let mut cmp_scratch = Z::CmpScratch::default();
        tableau.is_lex_min = is_lex_min_basis_with_scratch(&tableau, d, 0, &mut cmp_scratch);
        Self {
            tableau,
            options,
            min_ratio_scratch,
            cmp_scratch,
            pivot_scratch: Z::PivotScratch::default(),
            cache: VecDeque::new(),
            cursor: Cursor::Scan {
                next_cobasis_pos: 0,
            },
        }
    }

    pub(super) fn tableau(&self) -> &Tableau<Z> {
        &self.tableau
    }

    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            cobasis: self.tableau.cobasis_vars().to_vec(),
            depth: self.tableau.depth,
            cursor: self.cursor,
        }
    }

    pub(crate) fn apply_checkpoint_cursor(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        if self.tableau.cobasis_vars() != checkpoint.cobasis.as_slice() {
            return Err(Error::InvalidWarmStart);
        }

        let d = self.tableau.decision_vars;
        match checkpoint.cursor {
            Cursor::Scan { next_cobasis_pos } if next_cobasis_pos <= d => {}
            Cursor::Backtrack => {}
            _ => return Err(Error::InvalidWarmStart),
        }

        self.tableau.depth = checkpoint.depth;
        self.cursor = checkpoint.cursor;
        self.cache.clear();
        Ok(())
    }

    pub fn advance(&mut self) -> Result<bool> {
        let d = self.tableau.decision_vars;

        loop {
            if let Some(max_depth) = self.options.max_depth
                && self.tableau.depth >= max_depth
            {
                self.cursor = Cursor::Backtrack;
            }

            match self.cursor {
                Cursor::Backtrack if self.tableau.depth == 0 => return Ok(false),
                Cursor::Backtrack => {
                    if let Some(entry) = self.cache.pop_back() {
                        self.tableau = entry.tableau;
                        self.cursor = Cursor::Scan {
                            next_cobasis_pos: entry.resume_scan_pos,
                        };
                        continue;
                    }

                    self.tableau.depth -= 1;
                    let Some((bas, cob)) = choose_parent_pivot(
                        &self.tableau,
                        d,
                        &mut self.min_ratio_scratch,
                        &mut self.cmp_scratch,
                    ) else {
                        return Err(Error::InvariantViolation);
                    };

                    let (_bas, cob) =
                        self.tableau
                            .pivot_with_scratch(bas, cob, &mut self.pivot_scratch)?;
                    self.cursor = Cursor::Scan {
                        next_cobasis_pos: cob.saturating_add(1),
                    };
                    continue;
                }
                Cursor::Scan { next_cobasis_pos } => {
                    let mut scan_pos = next_cobasis_pos;
                    while scan_pos < d {
                        let s = scan_pos;
                        if let Some(bas) = is_reverse_pivot_child(
                            &self.tableau,
                            d,
                            s,
                            &mut self.min_ratio_scratch,
                            &mut self.cmp_scratch,
                        ) {
                            if self.options.cache_limit > 0 {
                                let resume_scan_pos = s.saturating_add(1);
                                let mut entry = if self.cache.len() < self.options.cache_limit {
                                    CacheEntry {
                                        tableau: self.tableau.allocate_like()?,
                                        resume_scan_pos,
                                    }
                                } else {
                                    let mut entry =
                                        self.cache.pop_front().expect("cache_limit > 0");
                                    entry.resume_scan_pos = resume_scan_pos;
                                    entry
                                };
                                entry.tableau.copy_from(&self.tableau);
                                self.cache.push_back(entry);
                            }

                            self.tableau.depth += 1;
                            self.tableau
                                .pivot_with_scratch(bas, s, &mut self.pivot_scratch)?;
                            self.tableau.is_lex_min = is_lex_min_basis_with_scratch(
                                &self.tableau,
                                d,
                                0,
                                &mut self.cmp_scratch,
                            );
                            self.cursor = Cursor::Scan {
                                next_cobasis_pos: 0,
                            };
                            return Ok(true);
                        }
                        scan_pos += 1;
                    }

                    self.cursor = Cursor::Backtrack;
                    continue;
                }
            }
        }
    }
}
