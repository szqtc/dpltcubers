// copy from https://github.com/akurmustafa/bisection/blob/master/src/lib.rs
use std::ops::{Bound::*, RangeBounds};

/// Convert bounds to a `(lo, hi)`  pair for indexing into a slice of `a`.
///
/// # Panics
///
/// Panics if `within` is out of bounds of `a`.
fn bounds_to_indices<T, I>(a: &[T], within: I) -> (usize, usize)
where
    I: RangeBounds<usize>,
{
    // TODO: Panics
    let lo = match within.start_bound() {
        Unbounded => 0,
        Included(i) => *i,
        Excluded(i) => i + 1,
    };

    let hi = match within.end_bound() {
        Unbounded => a.len(),
        Included(i) => i + 1,
        Excluded(i) => *i,
    };

    if hi > a.len() {
        panic!("index out of bounds")
    }

    (lo, hi)
}

/// Return the index where `x` should be inserted in `a[within]`, assuming `a`
/// is sorted.
///
/// The return value `i` is such that all `e` in `a[..i]` have `e < x`, and
/// all `e` in `a[i..]` have `e >= x`.
/// - If `a` contains `x`, `a.insert(i, x)` will insert just *before* the
///   *leftmost* `x`.
///
/// # Panics
///
/// Panics if `within` is out of bounds of `a`.
pub fn bisect_left_slice<T, I>(a: &[T], x: &T, within: I) -> usize
where
    I: RangeBounds<usize>,
    T: PartialOrd,
{
    let (mut lo, mut hi) = bounds_to_indices(a, within);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if a[mid] < *x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Return the index where `x` should be inserted in `a`, assuming `a` is sorted.
///
/// The return value `i` is such that all `e` in `a[..i]` have `e < x`, and
/// all `e` in `a[i..]` have `e >= x`.
/// - If `a` contains `x`, `a.insert(i, x)` will insert just *before* the
///   *leftmost* `x`.
pub fn bisect_left<T>(a: &[T], x: &T) -> usize
where
    T: PartialOrd,
{
    bisect_left_slice(a, x, ..)
}

/// Return the index where `x` should be inserted in `a[within]`, assuming `a`
/// is sorted.
///
/// The return value `i` is such that all `e` in `a[..i]` have `e <= x`, and
/// all `e` in `a[i..]` have `e > x`.
/// - If `a` contains `x`, `a.insert(i, x)` will insert just *after* the
///   *rightmost* `x`.
///
/// # Panics
///
/// Panics if `within` is out of bounds of `a`.
pub fn bisect_right_slice<T, I>(a: &[T], x: &T, within: I) -> usize
where
    I: RangeBounds<usize>,
    T: PartialOrd,
{
    let (mut lo, mut hi) = bounds_to_indices(a, within);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if *x < a[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Return the index where `x` should be inserted in `a`, assuming `a` is sorted.
///
/// The return value `i` is such that all `e` in `a[..i]` have `e <= x`, and
/// all `e` in `a[i..]` have `e > x`.
/// - If `a` contains `x`, `a.insert(i, x)` will insert just *after* the
///   *rightmost* occurence of `x`.
pub fn bisect_right<T>(a: &[T], x: &T) -> usize
where
    T: PartialOrd,
{
    bisect_right_slice(a, x, ..)
}
