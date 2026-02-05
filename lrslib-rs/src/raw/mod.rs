use std::{
    marker::PhantomData,
    ptr,
    rc::Rc,
    sync::{Mutex, Once},
};

use crate::{LrsError, LrsResult, sys};

pub(crate) mod mp;

static INIT: Once = Once::new();
static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

fn take_ticket() -> std::sync::MutexGuard<'static, ()> {
    let lock = match GLOBAL_LOCK.try_lock() {
        Ok(lock) => lock,
        Err(std::sync::TryLockError::WouldBlock) => {
            panic!("lrslib is already in use (concurrent calls are unsupported)")
        }
        Err(std::sync::TryLockError::Poisoned(_)) => {
            panic!("lrslib global lock poisoned (concurrent calls are unsupported)")
        }
    };

    INIT.call_once(|| {
        let name = c"lrslib-rs";
        let ok = unsafe { sys::lrs_init(name.as_ptr()) };
        assert!(ok != 0, "lrslib initialization failed");
    });

    lock
}

pub(crate) struct Handle {
    _ticket: std::sync::MutexGuard<'static, ()>,
    q: *mut sys::lrs_dat,
    p: *mut sys::lrs_dic,
    output: sys::lrs_mp_vector,
    tmp_num: sys::lrs_mp_vector,
    tmp_den: sys::lrs_mp_vector,
    lin: sys::lrs_mp_matrix,
    n: usize,
    _no_send_sync: PhantomData<Rc<()>>,
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            let n = self.n as i64;
            if !self.tmp_num.is_null() {
                sys::lrs_clear_mp_vector(self.tmp_num, n);
            }
            if !self.tmp_den.is_null() {
                sys::lrs_clear_mp_vector(self.tmp_den, n);
            }
            if !self.output.is_null() {
                sys::lrs_clear_mp_vector(self.output, n);
            }
            if !self.p.is_null() && !self.q.is_null() {
                sys::lrs_free_dic(self.p, self.q);
            }
            if !self.q.is_null() {
                sys::lrs_free_dat(self.q);
            }
        }
    }
}

impl Handle {
    pub(crate) fn new(m: usize, n: usize, hull: bool, polytope: bool) -> LrsResult<Self> {
        let lock = take_ticket();

        let q_name = c"LRS globals";
        let q = unsafe { sys::lrs_alloc_dat(q_name.as_ptr()) };
        if q.is_null() {
            return Err(LrsError::NullPointer);
        }

        unsafe {
            (*q).m = m as i64;
            (*q).n = n as i64;
            (*q).hull = if hull { 1 } else { 0 };
            (*q).polytope = if polytope { 1 } else { 0 };
            (*q).messages = 0;
            (*q).verbose = 0;
            (*q).printcobasis = 0;
            (*q).printslack = 0;
        }

        let output = unsafe { sys::lrs_alloc_mp_vector(n as i64) };
        if output.is_null() {
            unsafe {
                sys::lrs_free_dat(q);
            }
            return Err(LrsError::NullPointer);
        }

        let p = unsafe { sys::lrs_alloc_dic(q) };
        if p.is_null() {
            unsafe {
                sys::lrs_clear_mp_vector(output, n as i64);
                sys::lrs_free_dat(q);
            }
            return Err(LrsError::NullPointer);
        }

        let tmp_num = unsafe { sys::lrs_alloc_mp_vector(n as i64) };
        let tmp_den = unsafe { sys::lrs_alloc_mp_vector(n as i64) };
        if tmp_num.is_null() || tmp_den.is_null() {
            unsafe {
                if !tmp_num.is_null() {
                    sys::lrs_clear_mp_vector(tmp_num, n as i64);
                }
                if !tmp_den.is_null() {
                    sys::lrs_clear_mp_vector(tmp_den, n as i64);
                }
                sys::lrs_free_dic(p, q);
                sys::lrs_clear_mp_vector(output, n as i64);
                sys::lrs_free_dat(q);
            }
            return Err(LrsError::NullPointer);
        }

        Ok(Self {
            _ticket: lock,
            q,
            p,
            output,
            tmp_num,
            tmp_den,
            lin: ptr::null_mut(),
            n,
            _no_send_sync: PhantomData,
        })
    }

    #[inline]
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    #[inline]
    pub(crate) fn output_vector(&self) -> sys::lrs_mp_vector {
        self.output
    }

    pub(crate) fn set_row(&mut self, row_idx_1based: usize, row: &[f64]) -> LrsResult<()> {
        if row.len() != self.n {
            return Err(LrsError::InvalidMatrix {
                rows: 1,
                cols: row.len(),
            });
        }

        for (col, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(LrsError::NonFinite {
                    row: row_idx_1based - 1,
                    col,
                    value,
                });
            }
            let (num, den) = crate::f64_to_num_den(value);
            mp::set_mp_from_integer(mp::mp_ptr_from_vec(self.tmp_num, col), &num)?;
            mp::set_mp_from_integer(mp::mp_ptr_from_vec(self.tmp_den, col), &den)?;
        }

        unsafe {
            sys::lrs_set_row_mp(
                self.p,
                self.q,
                row_idx_1based as i64,
                self.tmp_num,
                self.tmp_den,
                sys::GE as i64,
            );
        }
        Ok(())
    }

    pub(crate) fn set_vertex_row_i64(&mut self, row_idx_1based: usize, coords: &[i64]) -> LrsResult<()> {
        let cols = self.n;
        if coords.len().saturating_add(1) != cols {
            return Err(LrsError::InvalidMatrix {
                rows: 1,
                cols: coords.len().saturating_add(1),
            });
        }

        mp::set_mp_from_i64(mp::mp_ptr_from_vec(self.tmp_num, 0), 1)?;
        mp::set_mp_from_i64(mp::mp_ptr_from_vec(self.tmp_den, 0), 1)?;

        for (col, &value) in coords.iter().enumerate() {
            mp::set_mp_from_i64(mp::mp_ptr_from_vec(self.tmp_num, col + 1), value)?;
            mp::set_mp_from_i64(mp::mp_ptr_from_vec(self.tmp_den, col + 1), 1)?;
        }

        unsafe {
            sys::lrs_set_row_mp(
                self.p,
                self.q,
                row_idx_1based as i64,
                self.tmp_num,
                self.tmp_den,
                sys::GE as i64,
            );
        }
        Ok(())
    }

    pub(crate) fn get_first_basis(&mut self) -> LrsResult<()> {
        let ok = unsafe { sys::lrs_getfirstbasis(&mut self.p, self.q, &mut self.lin, 1) };
        if ok == 0 {
            return Err(LrsError::NoInitialBasis);
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn solution_cols(&self) -> i64 {
        unsafe { (*self.p).d }
    }

    #[inline]
    pub(crate) fn get_solution(&mut self, col: i64) -> bool {
        unsafe { sys::lrs_getsolution(self.p, self.q, self.output, col) != 0 }
    }

    #[inline]
    pub(crate) fn next_basis(&mut self) -> bool {
        unsafe { sys::lrs_getnextbasis(&mut self.p, self.q, 0) != 0 }
    }

    pub(crate) fn solution_incidence(&self, col: i64) -> Vec<usize> {
        let (d, m, lastdv, a, row, b, c, inequality) = unsafe {
            (
                (*self.p).d as usize,
                (*self.p).m as usize,
                (*self.q).lastdv as usize,
                (*self.p).A,
                (*self.p).Row,
                (*self.p).B,
                (*self.p).C,
                (*self.q).inequality,
            )
        };

        let mut out = Vec::new();

        // Base cobasis elements.
        for i in 0..d {
            let c_i = unsafe { *c.add(i) as isize };
            let idx = c_i - lastdv as isize;
            if idx <= 0 {
                continue;
            }
            let ineq_num = unsafe { *inequality.add(idx as usize) as isize };
            if ineq_num > 0 {
                out.push((ineq_num - 1) as usize);
            }
        }

        // Additional incident inequalities beyond the cobasis.
        for i in (lastdv + 1)..=m {
            let row_i = unsafe { *row.add(i) as usize };
            let a_row = unsafe { *a.add(row_i) };
            if !mp::mp_is_zero(mp::mp_ptr_from_matrix_row(a_row, 0)) {
                continue;
            }
            if col != 0 && !mp::mp_is_zero(mp::mp_ptr_from_matrix_row(a_row, col as usize)) {
                continue;
            }

            let b_i = unsafe { *b.add(i) as isize };
            let idx = b_i - lastdv as isize;
            if idx <= 0 {
                continue;
            }
            let ineq_num = unsafe { *inequality.add(idx as usize) as isize };
            if ineq_num > 0 {
                out.push((ineq_num - 1) as usize);
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }
}
