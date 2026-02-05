#[inline(always)]
pub(super) fn repair_sorted_pair_u32(
    keys: &mut [u32],
    values: &mut [u32],
    mut idx: usize,
    range: usize,
) {
    debug_assert!(range <= keys.len());
    debug_assert!(range <= values.len());

    while idx > 0 && keys[idx] < keys[idx - 1] {
        keys.swap(idx, idx - 1);
        values.swap(idx, idx - 1);
        idx -= 1;
    }
    while idx + 1 < range && keys[idx] > keys[idx + 1] {
        keys.swap(idx, idx + 1);
        values.swap(idx, idx + 1);
        idx += 1;
    }
}
