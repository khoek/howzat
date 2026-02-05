use rug::Integer;

use crate::{LrsError, LrsResult, sys};

#[inline]
pub(crate) fn mp_ptr_from_vec(vec: sys::lrs_mp_vector, idx: usize) -> *mut i128 {
    unsafe { *vec.add(idx) }
}

#[inline]
pub(crate) fn mp_ptr_from_matrix_row(row: *mut *mut i128, idx: usize) -> *mut i128 {
    unsafe { *row.add(idx) }
}

pub(crate) fn set_mp_from_integer(target: *mut i128, value: &Integer) -> LrsResult<()> {
    let value: i128 = value
        .to_i128()
        .ok_or(LrsError::Unsupported("coefficient too large for non-GMP backend"))?;
    unsafe { *target = value };
    Ok(())
}

pub(crate) fn set_mp_from_i64(target: *mut i128, value: i64) -> LrsResult<()> {
    unsafe { *target = value as i128 };
    Ok(())
}

#[inline]
pub(crate) fn mp_is_zero(mp: *mut i128) -> bool {
    unsafe { *mp == 0 }
}

pub(crate) fn mp_rat_to_f64(num: *mut i128, den: *mut i128) -> f64 {
    let mut out = 0.0;
    unsafe {
        sys::rattodouble(num, den, &mut out);
    }
    out
}

pub(crate) fn mp_int_to_f64(mp: *mut i128) -> LrsResult<f64> {
    Ok(unsafe { *mp as f64 })
}

pub(crate) fn mp_int_to_integer(mp: *mut i128) -> LrsResult<Integer> {
    Ok(Integer::from(unsafe { *mp }))
}
