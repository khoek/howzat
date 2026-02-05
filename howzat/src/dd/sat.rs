use std::fmt;

const WORD_BITS: usize = usize::BITS as usize;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct SatRowId(usize);

impl SatRowId {
    #[inline(always)]
    pub(crate) fn as_index(self) -> usize {
        self.0
    }
}

impl From<usize> for SatRowId {
    #[inline(always)]
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<SatRowId> for usize {
    #[inline(always)]
    fn from(value: SatRowId) -> Self {
        value.0
    }
}

impl fmt::Debug for SatRowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SatRowId").field(&self.0).finish()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SatSet {
    words: Vec<usize>,
}

impl SatSet {
    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.words.clear();
    }

    #[inline]
    fn trim(&mut self) {
        while self.words.last().is_some_and(|w| *w == 0) {
            self.words.pop();
        }
    }

    #[inline(always)]
    pub(crate) fn copy_from(&mut self, other: &Self) {
        self.words.clear();
        self.words.extend_from_slice(&other.words);
    }

    #[inline]
    pub(crate) fn contains(&self, id: SatRowId) -> bool {
        let idx = id.as_index();
        let word = idx / WORD_BITS;
        let bit = idx % WORD_BITS;
        self.words
            .get(word)
            .is_some_and(|w| (*w & (1usize << bit)) != 0)
    }

    #[inline]
    pub(crate) fn insert(&mut self, id: SatRowId) {
        let idx = id.as_index();
        let word = idx / WORD_BITS;
        let bit = idx % WORD_BITS;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1usize << bit;
    }

    #[inline]
    pub(crate) fn intersection_inplace(&mut self, other: &Self) {
        let min_len = self.words.len().min(other.words.len());
        for i in 0..min_len {
            self.words[i] &= other.words[i];
        }
        for i in min_len..self.words.len() {
            self.words[i] = 0;
        }
        self.trim();
    }

    #[inline]
    pub(crate) fn intersection_inplace_and_count(&mut self, other: &Self) -> usize {
        let min_len = self.words.len().min(other.words.len());
        let mut count = 0usize;
        for i in 0..min_len {
            let word = self.words[i] & other.words[i];
            self.words[i] = word;
            count += word.count_ones() as usize;
        }
        for i in min_len..self.words.len() {
            self.words[i] = 0;
        }
        self.trim();
        count
    }

    #[inline]
    pub(crate) fn cardinality(&self) -> usize {
        let mut count = 0usize;
        for &word in &self.words {
            count += word.count_ones() as usize;
        }
        count
    }

    #[inline]
    pub(crate) fn cardinality_at_least(&self, target: usize) -> bool {
        if target == 0 {
            return true;
        }
        let mut count = 0usize;
        for &word in &self.words {
            count += word.count_ones() as usize;
            if count >= target {
                return true;
            }
        }
        false
    }

    #[inline]
    pub(crate) fn signature_u64(&self) -> u64 {
        #[inline(always)]
        fn mix64(mut z: u64) -> u64 {
            z ^= z >> 30;
            z = z.wrapping_mul(0xBF58476D1CE4E5B9);
            z ^= z >> 27;
            z = z.wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        let mut state = 0xDEADBEEFDEADBEEF_u64 ^ (self.words.len() as u64);
        for (idx, &word) in self.words.iter().enumerate() {
            state = state.wrapping_add((word as u64).wrapping_add((idx as u64).wrapping_mul(0x9E37)));
            state = mix64(state);
        }
        state
    }

    pub(crate) fn iter(&self) -> SatSetIter<'_> {
        SatSetIter::new(&self.words)
    }
}

#[doc(hidden)]
pub struct SatSetIter<'a> {
    words: &'a [usize],
    next_word_idx: usize,
    current_word_idx: usize,
    current_word: usize,
}

impl<'a> SatSetIter<'a> {
    fn new(words: &'a [usize]) -> Self {
        Self {
            words,
            next_word_idx: 0,
            current_word_idx: 0,
            current_word: 0,
        }
    }
}

impl<'a> Iterator for SatSetIter<'a> {
    type Item = SatRowId;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_word == 0 {
            let word = self.words.get(self.next_word_idx).copied()?;
            self.current_word_idx = self.next_word_idx;
            self.next_word_idx += 1;
            self.current_word = word;
        }
        let bit = self.current_word.trailing_zeros() as usize;
        self.current_word &= self.current_word - 1;
        Some(SatRowId(self.current_word_idx * WORD_BITS + bit))
    }
}
