use crate::geometrizable::Geometrizable;
use calculo::num::{Epsilon, Num};

/// Errors that can occur during drum promotion operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromotionError {
    /// The target skin is simplicial (has exactly dim+1 vertices), so no
    /// non-essential vertex exists to use as a lift vertex.
    SkinIsSimplicial {
        skin: DrumSkin,
        vertex_count: usize,
        base_dim: usize,
    },
    /// The target skin is not full-dimensional.
    SkinNotFullDimensional {
        skin: DrumSkin,
        actual_dim: usize,
        expected_dim: usize,
    },
    /// No suitable lift vertex could be found that keeps the skin full-dimensional.
    NoSuitableLiftVertex { skin: DrumSkin },
}

impl std::error::Error for PromotionError {}

impl std::fmt::Display for PromotionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SkinIsSimplicial {
                skin,
                vertex_count,
                base_dim,
            } => write!(
                f,
                "{skin:?} skin is simplicial ({vertex_count} vertices for dimension {base_dim})"
            ),
            Self::SkinNotFullDimensional {
                skin,
                actual_dim,
                expected_dim,
            } => write!(
                f,
                "{skin:?} skin is not full-dimensional (dim={actual_dim}, expected={expected_dim})"
            ),
            Self::NoSuitableLiftVertex { skin } => {
                write!(f, "no suitable lift vertex found on {skin:?} skin")
            }
        }
    }
}

/// Which base facet ("skin") of a drum a vertex belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrumSkin {
    Top,
    Bot,
}

/// Parameters for a single Santos/Williamson "drum promotion" step.
///
/// In a promotion step, we:
/// - one-point suspend (split) a vertex `v` on one skin into two vertices `v±`,
/// - perturb (lift) a vertex `a` on the opposite skin along the same new axis.
///
/// This produces a new drum in one higher base dimension with one additional vertex.
#[derive(Debug, Clone)]
pub struct DrumPromotion<N: Num> {
    /// Which skin the split vertex `v` is taken from.
    pub split_skin: DrumSkin,
    /// Index of the split vertex `v` within that skin's vertex list.
    pub split_vertex: usize,
    /// Index of the lifted vertex `a` within the *opposite* skin's vertex list.
    pub lift_vertex: usize,
    /// Height `h` for the one-point suspension, creating `v±` at `±h` in the new coordinate.
    pub suspension_height: N,
    /// Height `δ` for lifting vertex `a` in the new coordinate.
    pub lift_height: N,
}

impl<N: Num> DrumPromotion<N> {
    pub fn new(split_skin: DrumSkin, split_vertex: usize, lift_vertex: usize) -> Self {
        Self {
            split_skin,
            split_vertex,
            lift_vertex,
            suspension_height: N::one(),
            lift_height: N::one().ref_div(&N::from_u64(2)),
        }
    }

    /// Choose a lift vertex automatically so that the lift side remains full-dimensional.
    ///
    /// Returns an error if no suitable lift vertex can be found (e.g., skin is simplicial).
    pub fn auto(
        bases: &DrumBases<N>,
        split_skin: DrumSkin,
        split_vertex: usize,
    ) -> Result<Self, PromotionError> {
        let eps = N::default_eps();
        let lift_skin = match split_skin {
            DrumSkin::Top => DrumSkin::Bot,
            DrumSkin::Bot => DrumSkin::Top,
        };
        let lift_vertices = match split_skin {
            DrumSkin::Top => bases.bot(),
            DrumSkin::Bot => bases.top(),
        };

        let lift_vertex =
            find_nonessential_vertex(lift_vertices, bases.base_dim(), lift_skin, &eps)?;
        Ok(Self::new(split_skin, split_vertex, lift_vertex))
    }

    pub fn with_heights(mut self, suspension_height: N, lift_height: N) -> Self {
        self.suspension_height = suspension_height;
        self.lift_height = lift_height;
        self
    }
}

/// Base vertex sets for a drum (top and bot skins) in the drum's base dimension.
#[derive(Debug, Clone)]
pub struct DrumBases<N: Num> {
    top: Vec<Vec<N>>,
    bot: Vec<Vec<N>>,
    base_dim: usize,
}

impl<N: Num> DrumBases<N> {
    pub fn new(top: Vec<Vec<N>>, bot: Vec<Vec<N>>) -> Self {
        assert!(
            !top.is_empty() && !bot.is_empty(),
            "drum bases must include at least one top and one bot vertex"
        );

        let base_dim = top[0].len();
        assert!(
            base_dim > 0,
            "drum base vertices must have positive dimension"
        );
        assert!(
            top.iter().all(|v| v.len() == base_dim) && bot.iter().all(|v| v.len() == base_dim),
            "drum bases must have consistent ambient dimension"
        );

        Self { top, bot, base_dim }
    }

    pub fn base_dim(&self) -> usize {
        self.base_dim
    }

    pub fn degenerate_with_eps(&self, eps: &impl Epsilon<N>) -> bool {
        affine_dimension(&self.top, eps) != self.base_dim
            || affine_dimension(&self.bot, eps) != self.base_dim
    }

    pub fn degenerate(&self) -> bool {
        let eps = N::default_eps();
        self.degenerate_with_eps(&eps)
    }

    /// Drum ambient dimension (one more than the base dimension).
    pub fn drum_dim(&self) -> usize {
        self.base_dim + 1
    }

    pub fn num_vertices(&self) -> usize {
        self.top.len() + self.bot.len()
    }

    pub fn top(&self) -> &[Vec<N>] {
        &self.top
    }

    pub fn bot(&self) -> &[Vec<N>] {
        &self.bot
    }

    /// Promote this drum by one base dimension, consuming `self`.
    ///
    /// # Panics
    ///
    /// Panics if promotion parameters are invalid or promotion fails to produce full-dimensional bases.
    pub fn promote(self, promotion: DrumPromotion<N>) -> Self {
        assert!(
            promotion.suspension_height != N::zero(),
            "promotion suspension_height must be nonzero"
        );
        assert!(
            promotion.lift_height != N::zero(),
            "promotion lift_height must be nonzero"
        );

        let required_for_lift = self.drum_dim() + 1;
        let (mut top, mut bot) = (self.top, self.bot);
        let new_base_dim = self.base_dim + 1;

        match promotion.split_skin {
            DrumSkin::Top => {
                assert!(
                    promotion.split_vertex < top.len(),
                    "split_vertex {} out of range for top skin (len={})",
                    promotion.split_vertex,
                    top.len()
                );
                assert!(
                    promotion.lift_vertex < bot.len(),
                    "lift_vertex {} out of range for bot skin (len={})",
                    promotion.lift_vertex,
                    bot.len()
                );
                assert!(
                    bot.len() >= required_for_lift,
                    "cannot promote: bot skin has {} vertices but needs at least {} \
                     to become full-dimensional after lift",
                    bot.len(),
                    required_for_lift
                );

                top = promote_split_vertices(
                    top,
                    promotion.split_vertex,
                    &promotion.suspension_height,
                );
                bot = promote_lift_vertices(bot, promotion.lift_vertex, &promotion.lift_height);
            }
            DrumSkin::Bot => {
                assert!(
                    promotion.split_vertex < bot.len(),
                    "split_vertex {} out of range for bot skin (len={})",
                    promotion.split_vertex,
                    bot.len()
                );
                assert!(
                    promotion.lift_vertex < top.len(),
                    "lift_vertex {} out of range for top skin (len={})",
                    promotion.lift_vertex,
                    top.len()
                );
                assert!(
                    top.len() >= required_for_lift,
                    "cannot promote: top skin has {} vertices but needs at least {} \
                     to become full-dimensional after lift",
                    top.len(),
                    required_for_lift
                );

                bot = promote_split_vertices(
                    bot,
                    promotion.split_vertex,
                    &promotion.suspension_height,
                );
                top = promote_lift_vertices(top, promotion.lift_vertex, &promotion.lift_height);
            }
        }

        let eps = N::default_eps();
        let top_dim = affine_dimension(&top, &eps);
        let bot_dim = affine_dimension(&bot, &eps);
        assert!(
            top_dim == new_base_dim && bot_dim == new_base_dim,
            "promotion did not produce full-dimensional bases \
             (top_dim={top_dim}, bot_dim={bot_dim}, expected={new_base_dim})"
        );

        Self {
            top,
            bot,
            base_dim: new_base_dim,
        }
    }

    /// Promote this drum by one base dimension without consuming it.
    pub fn promoted(&self, promotion: DrumPromotion<N>) -> Self {
        Self {
            top: self.top.clone(),
            bot: self.bot.clone(),
            base_dim: self.base_dim,
        }
        .promote(promotion)
    }
}

/// Prismatoid obtained by lifting two parallel base vertex sets into one higher dimension.
///
/// All vertices lie in one of two facets (the bases). This type does not compute widths or
/// facet graphs; it only produces geometry that can be handed to a backend solver.
#[derive(Debug, Clone)]
pub struct Drum<N: Num> {
    bases: DrumBases<N>,
}

impl<N: Num> Drum<N> {
    pub fn new(top: Vec<Vec<N>>, bot: Vec<Vec<N>>) -> Self {
        Self::from_bases(DrumBases::new(top, bot))
    }

    pub fn from_bases(bases: DrumBases<N>) -> Self {
        Self { bases }
    }

    pub fn bases(&self) -> &DrumBases<N> {
        &self.bases
    }

    pub fn base_dim(&self) -> usize {
        self.bases.base_dim()
    }

    pub fn drum_dim(&self) -> usize {
        self.bases.drum_dim()
    }

    pub fn num_vertices(&self) -> usize {
        self.bases.num_vertices()
    }

    pub fn degenerate_with_eps(&self, eps: &impl Epsilon<N>) -> bool {
        self.bases.degenerate_with_eps(eps)
    }

    pub fn degenerate(&self) -> bool {
        self.bases.degenerate()
    }

    /// Perform a single Santos/Williamson drum promotion step and return the promoted bases.
    pub fn promote_bases(&self, promotion: DrumPromotion<N>) -> DrumBases<N> {
        self.bases.promoted(promotion)
    }

    /// Perform a single Santos/Williamson drum promotion step and return the promoted drum.
    pub fn promote(&self, promotion: DrumPromotion<N>) -> Self {
        Self::from_bases(self.promote_bases(promotion))
    }
}

impl<N: Num> Geometrizable for Drum<N> {
    type N = N;

    fn into_vertices(self) -> Vec<Vec<Self::N>> {
        let mut vertices = embed_with_height(self.bases.top(), &N::one());
        vertices.extend(embed_with_height(self.bases.bot(), &N::zero()));
        vertices
    }
}

fn embed_with_height<N: Num>(vertices: &[Vec<N>], height: &N) -> Vec<Vec<N>> {
    vertices
        .iter()
        .map(|v| {
            let mut lifted = v.clone();
            lifted.push(height.clone());
            lifted
        })
        .collect()
}

fn affine_dimension<N: Num>(points: &[Vec<N>], eps: &impl Epsilon<N>) -> usize {
    if points.is_empty() {
        return 0;
    }
    let dim = points[0].len();
    if dim == 0 {
        return 0;
    }

    assert!(
        points.iter().all(|p| p.len() == dim),
        "cannot compute affine dimension: inconsistent point dimensions"
    );

    if points.len() == 1 {
        return 0;
    }

    let base = &points[0];
    let rows: Vec<Vec<N>> = points
        .iter()
        .skip(1)
        .map(|p| {
            p.iter()
                .zip(base.iter())
                .map(|(a, b)| a.ref_sub(b))
                .collect()
        })
        .collect();

    matrix_rank(&rows, eps)
}

fn affine_dimension_excluding<N: Num>(
    points: &[Vec<N>],
    exclude: usize,
    eps: &impl Epsilon<N>,
) -> usize {
    if points.is_empty() {
        return 0;
    }
    assert!(
        exclude < points.len(),
        "cannot compute affine dimension excluding index {exclude}: out of range (n={})",
        points.len()
    );

    let dim = points[0].len();
    if dim == 0 {
        return 0;
    }
    assert!(
        points.iter().all(|p| p.len() == dim),
        "cannot compute affine dimension: inconsistent point dimensions"
    );

    if points.len() <= 2 {
        return 0;
    }

    let base_idx = if exclude != 0 { 0 } else { 1 };
    let base = &points[base_idx];

    let mut rows: Vec<Vec<N>> = Vec::with_capacity(points.len() - 2);
    for (idx, p) in points.iter().enumerate() {
        if idx == exclude || idx == base_idx {
            continue;
        }
        let row = p
            .iter()
            .zip(base.iter())
            .map(|(a, b)| a.ref_sub(b))
            .collect();
        rows.push(row);
    }

    matrix_rank(&rows, eps)
}

fn matrix_rank<N: Num>(rows: &[Vec<N>], eps: &impl Epsilon<N>) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let cols = rows[0].len();
    if cols == 0 {
        return 0;
    }
    assert!(
        rows.iter().skip(1).all(|r| r.len() == cols),
        "rank matrix has inconsistent row lengths"
    );

    #[inline(always)]
    fn swap_rows_in_flat<N: Num>(data: &mut [N], width: usize, r1: usize, r2: usize) {
        debug_assert!(width > 0, "swap_rows_in_flat called with width=0");
        if r1 == r2 {
            return;
        }
        let start1 = r1 * width;
        let start2 = r2 * width;
        debug_assert!(start1 + width <= data.len(), "row 1 out of bounds");
        debug_assert!(start2 + width <= data.len(), "row 2 out of bounds");
        unsafe {
            let ptr = data.as_mut_ptr();
            std::ptr::swap_nonoverlapping(ptr.add(start1), ptr.add(start2), width);
        }
    }

    let m = rows.len();
    let n = cols;
    let width = n;
    let mut a = vec![N::zero(); m * n];
    for (i, row) in rows.iter().enumerate() {
        let start = i * width;
        a[start..start + width].clone_from_slice(row);
    }

    let mut rank = 0usize;
    let mut row = 0usize;
    for col in 0..n {
        let mut pivot_row = None;
        let mut best_abs = None;
        for r in row..m {
            let val = a[r * width + col].abs();
            if eps.is_zero(&val) {
                continue;
            }
            let better = best_abs
                .as_ref()
                .is_none_or(|b| val.partial_cmp(b).is_some_and(|o| o.is_gt()));
            if better {
                pivot_row = Some(r);
                best_abs = Some(val);
            }
        }
        let Some(piv) = pivot_row else { continue };
        if piv != row {
            swap_rows_in_flat(&mut a, width, row, piv);
        }
        let pivot_val = a[row * width + col].clone();
        let inv_pivot = N::one().ref_div(&pivot_val);
        for r in (row + 1)..m {
            let rstart = r * width;
            if eps.is_zero(&a[rstart + col]) {
                continue;
            }
            let factor = a[rstart + col].ref_mul(&inv_pivot);
            for c in col..n {
                let tmp = factor.ref_mul(&a[row * width + c]);
                let idx = rstart + c;
                a[idx] = a[idx].ref_sub(&tmp);
            }
        }
        rank += 1;
        row += 1;
        if row == m {
            break;
        }
    }

    rank
}

fn find_nonessential_vertex<N: Num>(
    vertices: &[Vec<N>],
    base_dim: usize,
    skin: DrumSkin,
    eps: &impl Epsilon<N>,
) -> Result<usize, PromotionError> {
    if vertices.len() <= base_dim + 1 {
        return Err(PromotionError::SkinIsSimplicial {
            skin,
            vertex_count: vertices.len(),
            base_dim,
        });
    }

    let dim = affine_dimension(vertices, eps);
    if dim != base_dim {
        return Err(PromotionError::SkinNotFullDimensional {
            skin,
            actual_dim: dim,
            expected_dim: base_dim,
        });
    }

    for idx in 0..vertices.len() {
        let dim_without = affine_dimension_excluding(vertices, idx, eps);
        if dim_without == base_dim {
            return Ok(idx);
        }
    }

    Err(PromotionError::NoSuitableLiftVertex { skin })
}

fn promote_split_vertices<N: Num>(
    vertices: Vec<Vec<N>>,
    split_vertex: usize,
    suspension_height: &N,
) -> Vec<Vec<N>> {
    let abs_h = suspension_height.abs();

    let mut promoted = Vec::with_capacity(vertices.len() + 1);
    for (idx, mut v) in vertices.into_iter().enumerate() {
        v.push(N::zero());
        if idx == split_vertex {
            let mut plus = v.clone();
            let last = plus.len() - 1;
            plus[last] = abs_h.clone();
            v[last] = abs_h.ref_neg();
            promoted.push(v);
            promoted.push(plus);
        } else {
            promoted.push(v);
        }
    }
    promoted
}

fn promote_lift_vertices<N: Num>(
    vertices: Vec<Vec<N>>,
    lift_vertex: usize,
    lift_height: &N,
) -> Vec<Vec<N>> {
    vertices
        .into_iter()
        .enumerate()
        .map(|(idx, mut v)| {
            v.push(if idx == lift_vertex {
                lift_height.clone()
            } else {
                N::zero()
            });
            v
        })
        .collect()
}
