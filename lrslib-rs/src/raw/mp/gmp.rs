use std::ffi::c_void;

use gmp_mpfr_sys::gmp;
use libc::{c_int, malloc, realloc};
use rug::{Integer, integer::Order};

use crate::{LrsResult, sys};

#[inline]
pub(crate) fn mp_ptr_from_vec(vec: sys::lrs_mp_vector, idx: usize) -> *mut sys::__mpz_struct {
    unsafe { vec.add(idx).cast::<sys::__mpz_struct>() }
}

#[inline]
pub(crate) fn mp_ptr_from_matrix_row(row: *mut sys::lrs_mp, idx: usize) -> *mut sys::__mpz_struct {
    unsafe { row.add(idx).cast::<sys::__mpz_struct>() }
}

pub(crate) fn set_mp_from_integer(
    target: *mut sys::__mpz_struct,
    value: &Integer,
) -> LrsResult<()> {
    unsafe {
        if value.is_zero() {
            (*target)._mp_size = 0;
            return Ok(());
        }

        let limbs = value.as_limbs();
        let limb_count: c_int = limbs
            .len()
            .try_into()
            .map_err(|_| crate::LrsError::Unsupported("mp coefficient too large"))?;

        if (*target)._mp_alloc < limb_count {
            let bytes = std::mem::size_of_val(limbs);
            let ptr = if (*target)._mp_alloc == 0 {
                malloc(bytes).cast::<sys::mp_limb_t>()
            } else {
                realloc((*target)._mp_d.cast::<c_void>(), bytes).cast::<sys::mp_limb_t>()
            };
            if ptr.is_null() {
                return Err(crate::LrsError::Unsupported("out of memory"));
            }
            (*target)._mp_d = ptr;
            (*target)._mp_alloc = limb_count;
        }

        std::ptr::copy_nonoverlapping(
            limbs.as_ptr().cast::<sys::mp_limb_t>(),
            (*target)._mp_d,
            limbs.len(),
        );
        (*target)._mp_size = if value.is_negative() {
            -limb_count
        } else {
            limb_count
        };
    }
    Ok(())
}

pub(crate) fn set_mp_from_i64(target: *mut sys::__mpz_struct, value: i64) -> LrsResult<()> {
    unsafe {
        if value == 0 {
            (*target)._mp_size = 0;
            return Ok(());
        }

        let abs: u64 = value.unsigned_abs();
        let limb_bits = (std::mem::size_of::<sys::mp_limb_t>() * 8) as u32;

        let (limb0, limb1, limb_count) = if limb_bits >= 64 {
            (abs as sys::mp_limb_t, 0 as sys::mp_limb_t, 1usize)
        } else {
            let mask = (1u64 << limb_bits) - 1;
            let limb0 = (abs & mask) as sys::mp_limb_t;
            let limb1 = (abs >> limb_bits) as sys::mp_limb_t;
            let limb_count = if limb1 != 0 { 2 } else { 1 };
            (limb0, limb1, limb_count)
        };

        let limb_count_i: c_int = limb_count
            .try_into()
            .map_err(|_| crate::LrsError::Unsupported("mp coefficient too large"))?;

        if (*target)._mp_alloc < limb_count_i {
            let bytes = limb_count.saturating_mul(std::mem::size_of::<sys::mp_limb_t>());
            let ptr = if (*target)._mp_alloc == 0 {
                malloc(bytes).cast::<sys::mp_limb_t>()
            } else {
                realloc((*target)._mp_d.cast::<c_void>(), bytes).cast::<sys::mp_limb_t>()
            };
            if ptr.is_null() {
                return Err(crate::LrsError::Unsupported("out of memory"));
            }
            (*target)._mp_d = ptr;
            (*target)._mp_alloc = limb_count_i;
        }

        *(*target)._mp_d = limb0;
        if limb_count == 2 {
            *(*target)._mp_d.add(1) = limb1;
        }
        (*target)._mp_size = if value.is_negative() {
            -limb_count_i
        } else {
            limb_count_i
        };
    }
    Ok(())
}

#[inline]
pub(crate) fn mp_is_zero(mp: *mut sys::__mpz_struct) -> bool {
    // Mini-GMP uses _mp_size==0 for zero.
    unsafe { (*mp)._mp_size == 0 }
}

pub(crate) fn mp_rat_to_f64(num: *mut sys::__mpz_struct, den: *mut sys::__mpz_struct) -> f64 {
    let mut out = 0.0;
    unsafe {
        sys::rattodouble(num, den, &mut out);
    }
    out
}

pub(crate) fn mp_int_to_f64(mp: *mut sys::__mpz_struct) -> LrsResult<f64> {
    Ok(unsafe { gmp::mpz_get_d(mp.cast_const().cast()) })
}

pub(crate) fn mp_int_to_integer(mp: *mut sys::__mpz_struct) -> LrsResult<Integer> {
    let size = unsafe { (*mp)._mp_size };
    if size == 0 {
        return Ok(Integer::new());
    }

    let limb_count = size.unsigned_abs() as usize;
    let limbs = unsafe { std::slice::from_raw_parts((*mp)._mp_d, limb_count) };
    let mut out = Integer::from_digits(limbs, Order::Lsf);
    if size < 0 {
        out = -out;
    }
    Ok(out)
}
