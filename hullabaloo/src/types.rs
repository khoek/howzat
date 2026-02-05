use calculo::num::Sign;
use fixedbitset::FixedBitSet;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

pub type Big = usize;
pub type Row = usize;
pub type Col = usize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct RowId(pub usize);

impl RowId {
    #[inline(always)]
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline(always)]
    pub fn as_index(self) -> usize {
        self.0
    }
}

impl From<usize> for RowId {
    #[inline(always)]
    fn from(value: usize) -> Self {
        RowId::new(value)
    }
}

impl From<RowId> for usize {
    #[inline(always)]
    fn from(value: RowId) -> Self {
        value.0
    }
}

impl std::fmt::Display for RowId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<usize> for RowId {
    fn eq(&self, other: &usize) -> bool {
        &self.0 == other
    }
}

impl PartialEq<RowId> for usize {
    fn eq(&self, other: &RowId) -> bool {
        self == &other.0
    }
}

impl std::ops::Add<usize> for RowId {
    type Output = RowId;

    fn add(self, rhs: usize) -> Self::Output {
        RowId(self.0 + rhs)
    }
}

impl std::ops::Sub<usize> for RowId {
    type Output = RowId;

    fn sub(self, rhs: usize) -> Self::Output {
        assert!(self.0 >= rhs, "RowId underflow");
        RowId(self.0 - rhs)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct ColId(pub usize);

impl ColId {
    #[inline(always)]
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline(always)]
    pub fn as_index(self) -> usize {
        self.0
    }
}

impl From<usize> for ColId {
    #[inline(always)]
    fn from(value: usize) -> Self {
        ColId::new(value)
    }
}

impl From<ColId> for usize {
    #[inline(always)]
    fn from(value: ColId) -> Self {
        value.0
    }
}

impl std::fmt::Display for ColId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<usize> for ColId {
    fn eq(&self, other: &usize) -> bool {
        &self.0 == other
    }
}

impl PartialEq<ColId> for usize {
    fn eq(&self, other: &ColId) -> bool {
        self == &other.0
    }
}

impl std::ops::Add<usize> for ColId {
    type Output = ColId;

    fn add(self, rhs: usize) -> Self::Output {
        ColId(self.0 + rhs)
    }
}

impl std::ops::Sub<usize> for ColId {
    type Output = ColId;

    fn sub(self, rhs: usize) -> Self::Output {
        assert!(self.0 >= rhs, "ColId underflow");
        ColId(self.0 - rhs)
    }
}

pub(crate) const WORD_BITS: usize = usize::BITS as usize;
const INLINE_BITS: usize = 128;
const INLINE_WORDS: usize = INLINE_BITS / WORD_BITS;

#[inline(always)]
const fn word_count(dimension: usize) -> usize {
    if dimension == 0 {
        0
    } else {
        1 + ((dimension - 1) / WORD_BITS)
    }
}

#[inline(always)]
const fn inline_capacity() -> usize {
    INLINE_WORDS * WORD_BITS
}

#[inline(always)]
pub fn trailing_mask(dimension: usize) -> usize {
    let remainder = dimension % WORD_BITS;
    if remainder == 0 {
        usize::MAX
    } else {
        (1usize << remainder) - 1
    }
}

fn shrink_bits(bits: &FixedBitSet, dimension: usize) -> FixedBitSet {
    let mut shrunk = FixedBitSet::with_capacity(dimension);
    let destination = shrunk.as_mut_slice();
    let source = bits.as_slice();
    let shared = destination.len().min(source.len());
    destination[..shared].copy_from_slice(&source[..shared]);
    mask_trailing_bits(destination, dimension);
    shrunk
}

#[inline]
fn shrink_inline_words(words: &mut [usize], old_len: usize, new_len: usize) {
    debug_assert!(new_len <= old_len);
    let old_words = word_count(old_len);
    let new_words = word_count(new_len);
    if new_words < old_words {
        words[new_words..old_words].fill(0);
    }
    mask_trailing_bits(&mut words[..new_words], new_len);
}

pub(crate) fn mask_trailing_bits(blocks: &mut [usize], dimension: usize) {
    let mask = trailing_mask(dimension);
    if mask == usize::MAX {
        return;
    }
    if let Some(last) = blocks.last_mut() {
        *last &= mask;
    }
}

#[inline]
fn fmix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[inline]
fn hash_words_signature_u64(dimension: usize, words: &[usize]) -> u64 {
    let mask = trailing_mask(dimension);
    let mut h = 0xDEADBEEF_u64 ^ (dimension as u64).wrapping_mul(0x9e3779b97f4a7c15);
    let Some((&last, prefix)) = words.split_last() else {
        return h;
    };
    for &word in prefix {
        h ^= fmix64(word as u64);
        h = h.wrapping_mul(0x9e3779b97f4a7c15);
    }
    h ^= fmix64((last & mask) as u64);
    h = h.wrapping_mul(0x9e3779b97f4a7c15);
    h
}

#[inline]
fn hash_words_u64<H: BuildHasher>(dimension: usize, words: &[usize], hasher: &H) -> u64 {
    let mut state = hasher.build_hasher();
    dimension.hash(&mut state);
    let mask = trailing_mask(dimension);
    let Some((&last, prefix)) = words.split_last() else {
        return state.finish();
    };
    for &word in prefix {
        word.hash(&mut state);
    }
    (last & mask).hash(&mut state);
    state.finish()
}

#[inline]
fn first_unset_bit(blocks: &[usize], dimension: usize) -> Option<usize> {
    if dimension == 0 {
        return None;
    }
    let mask = trailing_mask(dimension);
    let last = blocks.len().saturating_sub(1);
    for (block_idx, &block) in blocks.iter().enumerate() {
        let mut inverted = !block;
        if block_idx == last {
            inverted &= mask;
        }
        if inverted != 0 {
            let bit = inverted.trailing_zeros() as usize;
            let idx = block_idx * WORD_BITS + bit;
            if idx < dimension {
                return Some(idx);
            }
            return None;
        }
    }
    None
}

#[inline]
fn last_unset_bit(blocks: &[usize], dimension: usize) -> Option<usize> {
    if dimension == 0 {
        return None;
    }
    let mask = trailing_mask(dimension);
    let last = blocks.len().saturating_sub(1);
    for (block_idx, &block) in blocks.iter().enumerate().rev() {
        let mut inverted = !block;
        if block_idx == last {
            inverted &= mask;
        }
        if inverted != 0 {
            let bit = WORD_BITS - 1 - inverted.leading_zeros() as usize;
            let idx = block_idx * WORD_BITS + bit;
            if idx < dimension {
                return Some(idx);
            }
            return None;
        }
    }
    None
}

#[inline]
fn assert_index_in_range(kind: &'static str, idx: usize, len: usize) {
    assert!(
        idx < len,
        "{}: index {} out of bounds (len {})",
        kind,
        idx,
        len
    );
}

pub trait IndexId: Copy {
    fn from_index(index: usize) -> Self;
    fn as_index(self) -> usize;
}

impl IndexId for RowId {
    #[inline(always)]
    fn from_index(index: usize) -> Self {
        RowId::new(index)
    }

    #[inline(always)]
    fn as_index(self) -> usize {
        self.0
    }
}

impl IndexId for ColId {
    #[inline(always)]
    fn from_index(index: usize) -> Self {
        ColId::new(index)
    }

    #[inline(always)]
    fn as_index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BitStore {
    Inline([usize; INLINE_WORDS]),
    Heap(FixedBitSet),
}

#[derive(Debug, Eq, PartialEq)]
pub struct IdSet<Id> {
    len: usize,
    store: BitStore,
    _marker: PhantomData<Id>,
}

pub type RowSet = IdSet<RowId>;
pub type ColSet = IdSet<ColId>;

#[cfg(feature = "serde")]
impl<Id: IndexId> serde::Serialize for IdSet<Id> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;

        let mut tuple = serializer.serialize_tuple(2)?;
        tuple.serialize_element(&self.len)?;
        tuple.serialize_element(&self.to_indices())?;
        tuple.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, Id: IndexId> serde::Deserialize<'de> for IdSet<Id> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (len, members) =
            <(usize, Vec<usize>) as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self::from_indices(len, &members))
    }
}

impl<Id> AsRef<Self> for IdSet<Id> {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<Id> Clone for IdSet<Id> {
    fn clone(&self) -> Self {
        Self {
            len: self.len,
            store: self.store.clone(),
            _marker: PhantomData,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.copy_from(source);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IterMode {
    Ones,
    Zeroes,
}

pub struct SetRawIter<'a> {
    blocks: &'a [usize],
    dimension: usize,
    word_idx: usize,
    current_word: usize,
    mode: IterMode,
}

impl<'a> SetRawIter<'a> {
    #[inline]
    fn new(blocks: &'a [usize], dimension: usize, mode: IterMode) -> Self {
        Self {
            blocks,
            dimension,
            word_idx: 0,
            current_word: 0,
            mode,
        }
    }

    #[inline]
    pub fn complement(self) -> Self {
        let mode = match self.mode {
            IterMode::Ones => IterMode::Zeroes,
            IterMode::Zeroes => IterMode::Ones,
        };
        Self::new(self.blocks, self.dimension, mode)
    }
}

impl<'a> Iterator for SetRawIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.dimension == 0 {
            return None;
        }
        let mask = trailing_mask(self.dimension);
        let last = self.blocks.len().saturating_sub(1);
        loop {
            if self.current_word == 0 {
                if self.word_idx >= self.blocks.len() {
                    return None;
                }
                let mut word = self.blocks[self.word_idx];
                if self.mode == IterMode::Zeroes {
                    word = !word;
                }
                if self.word_idx == last {
                    word &= mask;
                }
                self.current_word = word;
                if self.current_word == 0 {
                    self.word_idx += 1;
                    continue;
                }
            }
            let bit = self.current_word.trailing_zeros() as usize;
            self.current_word &= self.current_word - 1;
            let idx = self.word_idx * WORD_BITS + bit;
            if self.current_word == 0 {
                self.word_idx += 1;
            }
            if idx < self.dimension {
                return Some(idx);
            }
        }
    }
}

pub struct SetIter<'a, Id: IndexId> {
    raw: SetRawIter<'a>,
    _marker: PhantomData<Id>,
}

impl<'a, Id: IndexId> SetIter<'a, Id> {
    #[inline]
    pub fn raw(self) -> SetRawIter<'a> {
        self.raw
    }

    #[inline]
    pub fn complement(self) -> Self {
        Self {
            raw: self.raw.complement(),
            _marker: PhantomData,
        }
    }
}

impl<'a, Id: IndexId> Iterator for SetIter<'a, Id> {
    type Item = Id;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next().map(Id::from_index)
    }
}

impl<Id> IdSet<Id> {
    pub fn new(dimension: usize) -> Self {
        if dimension <= inline_capacity() {
            Self {
                len: dimension,
                store: BitStore::Inline([0; INLINE_WORDS]),
                _marker: PhantomData,
            }
        } else {
            Self {
                len: dimension,
                store: BitStore::Heap(FixedBitSet::with_capacity(dimension)),
                _marker: PhantomData,
            }
        }
    }

    pub fn all(dimension: usize) -> Self {
        let mut set = Self::new(dimension);
        set.set_all(true);
        set
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn hash_signature_u64(&self) -> u64 {
        hash_words_signature_u64(self.len, self.bit_slice())
    }

    #[inline]
    pub fn hash_u64<H: BuildHasher>(&self, hasher: &H) -> u64 {
        hash_words_u64(self.len, self.bit_slice(), hasher)
    }

    #[inline]
    pub fn eq_masked(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        let wa = self.bit_slice();
        let wb = other.bit_slice();
        debug_assert_eq!(wa.len(), wb.len(), "IdSet::bit_slice length mismatch");
        let mask = trailing_mask(self.len);
        let Some((&last_a, prefix_a)) = wa.split_last() else {
            return true;
        };
        let Some((&last_b, prefix_b)) = wb.split_last() else {
            return false;
        };
        prefix_a == prefix_b && (last_a & mask) == (last_b & mask)
    }

    #[inline]
    pub fn cardinality_at_least(&self, target: usize) -> bool {
        if target == 0 {
            return true;
        }
        let words = self.bit_slice();
        let mask = trailing_mask(self.len);
        let Some((&last, prefix)) = words.split_last() else {
            return false;
        };
        let mut count = 0usize;
        for &word in prefix {
            count += word.count_ones() as usize;
            if count >= target {
                return true;
            }
        }
        count += (last & mask).count_ones() as usize;
        count >= target
    }

    pub fn resize(&mut self, dimension: usize) {
        if dimension == self.len {
            return;
        }

        if dimension <= inline_capacity() {
            match &mut self.store {
                BitStore::Inline(words) => {
                    if dimension < self.len {
                        shrink_inline_words(words, self.len, dimension);
                    }
                }
                BitStore::Heap(bits) => {
                    let mut words = [0usize; INLINE_WORDS];
                    let word_len = word_count(dimension);
                    let src = bits.as_slice();
                    words[..word_len].copy_from_slice(&src[..word_len]);
                    mask_trailing_bits(&mut words[..word_len], dimension);
                    self.store = BitStore::Inline(words);
                }
            }
            self.len = dimension;
            return;
        }

        match &mut self.store {
            BitStore::Heap(bits) => {
                if dimension > self.len {
                    bits.grow(dimension);
                } else {
                    *bits = shrink_bits(bits, dimension);
                }
                self.len = dimension;
            }
            BitStore::Inline(words) => {
                let mut bits = FixedBitSet::with_capacity(dimension);
                let dst = bits.as_mut_slice();
                let src_words = word_count(self.len);
                dst[..src_words].copy_from_slice(&words[..src_words]);
                self.store = BitStore::Heap(bits);
                self.len = dimension;
            }
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        self.resize(other.len);
        match (&mut self.store, &other.store) {
            (BitStore::Inline(dst), BitStore::Inline(src)) => {
                let words = word_count(self.len);
                dst[..words].copy_from_slice(&src[..words]);
                if words < INLINE_WORDS {
                    dst[words..].fill(0);
                }
            }
            (BitStore::Heap(dst), BitStore::Heap(src)) => {
                debug_assert_eq!(
                    dst.as_slice().len(),
                    src.as_slice().len(),
                    "IdSet::copy_from requires matching dimensions"
                );
                dst.as_mut_slice().copy_from_slice(src.as_slice());
            }
            _ => unreachable!("IdSet::copy_from requires normalized storage"),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.store {
            BitStore::Inline(words) => words[..word_count(self.len)].iter().all(|&w| w == 0),
            BitStore::Heap(bits) => bits.is_clear(),
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        match &mut self.store {
            BitStore::Inline(words) => words[..word_count(self.len)].fill(0),
            BitStore::Heap(bits) => bits.clear(),
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        debug_assert_eq!(self.len, other.len);
        let mut out = self.clone();
        out.intersection_inplace(other);
        out
    }

    pub fn intersection_inplace(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        match (&mut self.store, &other.store) {
            (BitStore::Inline(dst), BitStore::Inline(src)) => {
                let words = word_count(self.len);
                for (a, b) in dst[..words].iter_mut().zip(&src[..words]) {
                    *a &= *b;
                }
            }
            (BitStore::Heap(dst), BitStore::Heap(src)) => dst.intersect_with(src),
            _ => unreachable!("IdSet::intersection_inplace requires normalized storage"),
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        debug_assert_eq!(self.len, other.len);
        let mut out = self.clone();
        out.union_inplace(other);
        out
    }

    pub fn union_inplace(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        match (&mut self.store, &other.store) {
            (BitStore::Inline(dst), BitStore::Inline(src)) => {
                let words = word_count(self.len);
                for (a, b) in dst[..words].iter_mut().zip(&src[..words]) {
                    *a |= *b;
                }
                mask_trailing_bits(&mut dst[..words], self.len);
            }
            (BitStore::Heap(dst), BitStore::Heap(src)) => dst.union_with(src),
            _ => unreachable!("IdSet::union_inplace requires normalized storage"),
        }
    }

    pub fn difference(&self, other: &Self) -> Self {
        debug_assert_eq!(self.len, other.len);
        let mut out = self.clone();
        out.difference_inplace(other);
        out
    }

    pub fn difference_inplace(&mut self, other: &Self) {
        debug_assert_eq!(self.len, other.len);
        match (&mut self.store, &other.store) {
            (BitStore::Inline(dst), BitStore::Inline(src)) => {
                let words = word_count(self.len);
                for (a, b) in dst[..words].iter_mut().zip(&src[..words]) {
                    *a &= !*b;
                }
                mask_trailing_bits(&mut dst[..words], self.len);
            }
            (BitStore::Heap(dst), BitStore::Heap(src)) => dst.difference_with(src),
            _ => unreachable!("IdSet::difference_inplace requires normalized storage"),
        }
    }

    pub fn complement(&self) -> Self {
        if self.len == 0 {
            return Self::new(0);
        }
        let mut out = self.clone();
        out.toggle_all();
        out
    }

    pub fn subset_of(&self, other: &Self) -> bool {
        debug_assert_eq!(self.len, other.len);
        match (&self.store, &other.store) {
            (BitStore::Inline(a), BitStore::Inline(b)) => {
                let words = word_count(self.len);
                a[..words]
                    .iter()
                    .zip(&b[..words])
                    .all(|(x, y)| (x & !y) == 0)
            }
            (BitStore::Heap(a), BitStore::Heap(b)) => a.is_subset(b),
            _ => unreachable!("IdSet::subset_of requires normalized storage"),
        }
    }

    pub fn cardinality(&self) -> usize {
        match &self.store {
            BitStore::Inline(words) => words[..word_count(self.len)]
                .iter()
                .map(|w| w.count_ones() as usize)
                .sum(),
            BitStore::Heap(bits) => bits.count_ones(..),
        }
    }

    #[inline]
    pub fn bit_slice(&self) -> &[usize] {
        match &self.store {
            BitStore::Inline(words) => &words[..word_count(self.len)],
            BitStore::Heap(bits) => {
                let slice = bits.as_slice();
                &slice[..word_count(self.len)]
            }
        }
    }

    pub fn count_intersection(&self, other: &Self) -> usize {
        debug_assert_eq!(self.len, other.len);
        match (&self.store, &other.store) {
            (BitStore::Inline(a), BitStore::Inline(b)) => {
                let words = word_count(self.len);
                a[..words]
                    .iter()
                    .zip(&b[..words])
                    .map(|(x, y)| (x & y).count_ones() as usize)
                    .sum()
            }
            (BitStore::Heap(a), BitStore::Heap(b)) => a.intersection_count(b),
            _ => unreachable!("IdSet::count_intersection requires normalized storage"),
        }
    }

    fn raw_iter(&self, mode: IterMode) -> SetRawIter<'_> {
        SetRawIter::new(self.bit_slice(), self.len, mode)
    }

    fn set_all(&mut self, value: bool) {
        match &mut self.store {
            BitStore::Inline(words) => {
                let words_len = word_count(self.len);
                let fill = if value { usize::MAX } else { 0 };
                words[..words_len].fill(fill);
                if value {
                    mask_trailing_bits(&mut words[..words_len], self.len);
                }
            }
            BitStore::Heap(bits) => bits.set_range(.., value),
        }
    }

    fn toggle_all(&mut self) {
        match &mut self.store {
            BitStore::Inline(words) => {
                let words_len = word_count(self.len);
                for word in &mut words[..words_len] {
                    *word = !*word;
                }
                mask_trailing_bits(&mut words[..words_len], self.len);
            }
            BitStore::Heap(bits) => {
                bits.toggle_range(..);
                mask_trailing_bits(bits.as_mut_slice(), self.len);
            }
        }
    }
}

impl<Id: IndexId> IdSet<Id> {
    pub fn from_indices(dimension: usize, members: &[usize]) -> Self {
        let mut set = Self::new(dimension);
        for &m in members {
            set.insert(Id::from_index(m));
        }
        set
    }

    pub fn insert<T: Into<Id>>(&mut self, id: T) {
        let idx = id.into().as_index();
        assert_index_in_range("IdSet::insert", idx, self.len);
        match &mut self.store {
            BitStore::Inline(words) => words[idx / WORD_BITS] |= 1usize << (idx % WORD_BITS),
            BitStore::Heap(bits) => bits.insert(idx),
        }
    }

    pub fn remove<T: Into<Id>>(&mut self, id: T) {
        let idx = id.into().as_index();
        assert_index_in_range("IdSet::remove", idx, self.len);
        match &mut self.store {
            BitStore::Inline(words) => words[idx / WORD_BITS] &= !(1usize << (idx % WORD_BITS)),
            BitStore::Heap(bits) => bits.remove(idx),
        }
    }

    pub fn contains<T: Into<Id>>(&self, id: T) -> bool {
        let idx = id.into().as_index();
        assert_index_in_range("IdSet::contains", idx, self.len);
        match &self.store {
            BitStore::Inline(words) => {
                (words[idx / WORD_BITS] & (1usize << (idx % WORD_BITS))) != 0
            }
            BitStore::Heap(bits) => bits.contains(idx),
        }
    }

    pub fn iter(&self) -> SetIter<'_, Id> {
        SetIter {
            raw: self.raw_iter(IterMode::Ones),
            _marker: PhantomData,
        }
    }

    pub fn to_indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.cardinality());
        out.extend(self.iter().raw());
        out
    }

    pub fn first_unset(&self) -> Option<Id> {
        first_unset_bit(self.bit_slice(), self.len).map(Id::from_index)
    }

    pub fn last_unset(&self) -> Option<Id> {
        last_unset_bit(self.bit_slice(), self.len).map(Id::from_index)
    }
}

pub type RowIndex = Vec<isize>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowOrder {
    MaxIndex,
    MinIndex,
    MinCutoff,
    MaxCutoff,
    MixCutoff,
    LexMin,
    LexMax,
    RandomRow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepresentationKind {
    Inequality,
    Generator,
}

pub trait Representation: Debug {
    const KIND: RepresentationKind;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Inequality;

impl Representation for Inequality {
    const KIND: RepresentationKind = RepresentationKind::Inequality;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Generator;

impl Representation for Generator {
    const KIND: RepresentationKind = RepresentationKind::Generator;
}

pub trait DualRepresentation: Representation {
    type Dual: Representation;
}

impl DualRepresentation for Inequality {
    type Dual = Generator;
}

impl DualRepresentation for Generator {
    type Dual = Inequality;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Conversion {
    InequalityToGenerator,
    GeneratorToInequality,
    LpMax,
    LpMin,
    InteriorFind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IncidenceOutput {
    Off,
    Cardinality,
    Set,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdjacencyOutput {
    Off,
    List,
    Degree,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputationStatus {
    InProgress,
    AllFound,
    RegionEmpty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InequalityKind {
    Inequality,
    Equality,
    StrictInequality,
}

impl InequalityKind {
    #[inline(always)]
    pub fn violates_sign(self, sign: Sign, relaxed: bool) -> bool {
        match self {
            Self::Equality => sign != Sign::Zero,
            Self::Inequality => sign == Sign::Negative,
            Self::StrictInequality => {
                if relaxed {
                    sign != Sign::Positive
                } else {
                    sign == Sign::Negative
                }
            }
        }
    }

    #[inline(always)]
    pub fn weakly_violates_sign(self, sign: Sign, relaxed: bool) -> bool {
        match self {
            Self::Equality => sign != Sign::Zero,
            Self::Inequality => sign == Sign::Negative,
            Self::StrictInequality => {
                if relaxed {
                    false
                } else {
                    sign == Sign::Negative
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ColSet, RowSet};

    #[test]
    fn rowset_complement_iterator_covers_elements_across_blocks() {
        let mut set = RowSet::new(70);
        for idx in [0, 31, 63, 69] {
            set.insert(idx);
        }
        let members: Vec<_> = set.iter().map(usize::from).collect();
        assert_eq!(members, vec![0, 31, 63, 69]);

        let complement: Vec<_> = set.iter().complement().map(usize::from).collect();
        assert!(complement.iter().all(|&idx| !set.contains(idx)));

        let mut combined = members.clone();
        combined.extend_from_slice(&complement);
        combined.sort_unstable();
        assert_eq!(combined, (0..70).collect::<Vec<_>>());
    }

    #[test]
    fn resize_truncates_bits_past_new_length() {
        let mut set = RowSet::new(10);
        set.insert(0);
        set.insert(5);
        set.insert(9);

        set.resize(6);
        assert_eq!(set.len(), 6);
        let members: Vec<_> = set.iter().map(usize::from).collect();
        assert_eq!(members, vec![0, 5]);
        assert_eq!(set.cardinality(), 2);
    }

    #[test]
    fn colset_complement_matches_expected_members() {
        let mut set = ColSet::new(4);
        set.insert(1);
        set.insert(3);

        let members: Vec<_> = set.iter().map(usize::from).collect();
        assert_eq!(members, vec![1, 3]);

        let complement: Vec<_> = set.iter().complement().map(usize::from).collect();
        assert_eq!(complement, vec![0, 2]);
    }
    #[test]
    fn rowset_all_creates_full_set() {
        let set = RowSet::all(10);
        assert_eq!(set.len(), 10);
        assert_eq!(set.cardinality(), 10);
        for i in 0..10 {
            assert!(set.contains(i));
        }
    }

    #[test]
    fn unset_helpers_match_complement_iterators() {
        let mut set = RowSet::new(130);
        for idx in [0, 1, 64, 129] {
            set.insert(idx);
        }
        assert_eq!(
            set.first_unset().map(usize::from),
            set.iter().complement().next().map(usize::from)
        );
        assert_eq!(
            set.last_unset().map(usize::from),
            set.iter().complement().last().map(usize::from)
        );

        let full = RowSet::all(10);
        assert_eq!(full.first_unset(), None);
        assert_eq!(full.last_unset(), None);

        let empty = RowSet::new(10);
        assert_eq!(empty.first_unset().map(usize::from), Some(0));
        assert_eq!(empty.last_unset().map(usize::from), Some(9));
    }
}
