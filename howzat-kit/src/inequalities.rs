use anyhow::{anyhow, ensure};
use calculo::num::Num;
use howzat::matrix::{LpMatrix, LpMatrixBuilder};
use hullabaloo::types::Inequality;

pub(crate) trait InequalitiesF64 {
    fn facet_count(&self) -> usize;
    fn dim(&self) -> usize;
    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_;
}

pub(crate) trait InequalitiesI64 {
    fn facet_count(&self) -> usize;
    fn dim(&self) -> usize;
    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_;
}

pub(crate) trait HowzatInequalities {
    fn build_howzat_inequality_matrix<N: Num>(
        &self,
    ) -> Result<LpMatrix<N, Inequality>, anyhow::Error>;
}

fn build_howzat_inequalities_f64<N: Num, I: InequalitiesF64 + ?Sized>(
    inequalities: &I,
) -> Result<LpMatrix<N, Inequality>, anyhow::Error> {
    let facet_count = inequalities.facet_count();
    ensure!(facet_count > 0, "howzat needs at least one inequality");
    let dim = inequalities.dim();
    ensure!(dim > 0, "howzat inequalities must have positive dimension");

    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("howzat inequality dimension too large"))?;
    let total = facet_count
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("howzat inequality matrix too large"))?;

    let mut data = Vec::with_capacity(total);
    for (row, coeffs) in inequalities.rows().enumerate() {
        ensure!(
            coeffs.len() == cols,
            "howzat inequalities must have consistent width (expected {cols}, got {} at row {row})",
            coeffs.len()
        );
        for &value in coeffs {
            ensure!(value.is_finite(), "non-finite inequality coefficient {value}");
            data.push(
                N::try_from_f64(value).ok_or_else(|| anyhow!("non-finite inequality coefficient {value}"))?,
            );
        }
    }

    Ok(LpMatrixBuilder::<N, Inequality>::from_flat(facet_count, cols, data).build())
}

fn build_howzat_inequalities_i64<N: Num, I: InequalitiesI64 + ?Sized>(
    inequalities: &I,
) -> Result<LpMatrix<N, Inequality>, anyhow::Error> {
    let facet_count = inequalities.facet_count();
    ensure!(facet_count > 0, "howzat needs at least one inequality");
    let dim = inequalities.dim();
    ensure!(dim > 0, "howzat inequalities must have positive dimension");

    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("howzat inequality dimension too large"))?;
    let total = facet_count
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("howzat inequality matrix too large"))?;

    let mut data = Vec::with_capacity(total);
    for (row, coeffs) in inequalities.rows().enumerate() {
        ensure!(
            coeffs.len() == cols,
            "howzat inequalities must have consistent width (expected {cols}, got {} at row {row})",
            coeffs.len()
        );
        for &value in coeffs {
            let mut out = N::from_u64(value.unsigned_abs());
            if value.is_negative() {
                out = -out;
            }
            data.push(out);
        }
    }

    Ok(LpMatrixBuilder::<N, Inequality>::from_flat(facet_count, cols, data).build())
}

impl InequalitiesF64 for [Vec<f64>] {
    fn facet_count(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        self.first().map_or(0, |row| row.len().saturating_sub(1))
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.iter().map(Vec::as_slice)
    }
}

impl HowzatInequalities for [Vec<f64>] {
    fn build_howzat_inequality_matrix<N: Num>(
        &self,
    ) -> Result<LpMatrix<N, Inequality>, anyhow::Error> {
        build_howzat_inequalities_f64(self)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RowMajorInequalities<'a> {
    coeffs: &'a [f64],
    facet_count: usize,
    dim: usize,
}

impl<'a> RowMajorInequalities<'a> {
    pub(crate) fn new(coeffs: &'a [f64], facet_count: usize, dim: usize) -> Result<Self, anyhow::Error> {
        ensure!(facet_count > 0, "need at least one inequality");
        ensure!(dim > 0, "need positive inequality dimension");
        let cols = dim
            .checked_add(1)
            .ok_or_else(|| anyhow!("inequality dimension too large"))?;
        ensure!(
            coeffs.len() == facet_count.saturating_mul(cols),
            "expected {facet_count}x{cols} coeffs but got {}",
            coeffs.len()
        );
        Ok(Self {
            coeffs,
            facet_count,
            dim,
        })
    }
}

impl InequalitiesF64 for RowMajorInequalities<'_> {
    fn facet_count(&self) -> usize {
        self.facet_count
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_ {
        let cols = self.dim + 1;
        self.coeffs.chunks_exact(cols)
    }
}

impl HowzatInequalities for RowMajorInequalities<'_> {
    fn build_howzat_inequality_matrix<N: Num>(
        &self,
    ) -> Result<LpMatrix<N, Inequality>, anyhow::Error> {
        build_howzat_inequalities_f64(self)
    }
}

impl InequalitiesI64 for [Vec<i64>] {
    fn facet_count(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        self.first().map_or(0, |row| row.len().saturating_sub(1))
    }

    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_ {
        self.iter().map(Vec::as_slice)
    }
}

impl HowzatInequalities for [Vec<i64>] {
    fn build_howzat_inequality_matrix<N: Num>(
        &self,
    ) -> Result<LpMatrix<N, Inequality>, anyhow::Error> {
        build_howzat_inequalities_i64(self)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RowMajorInequalitiesI64<'a> {
    coeffs: &'a [i64],
    facet_count: usize,
    dim: usize,
}

impl<'a> RowMajorInequalitiesI64<'a> {
    pub(crate) fn new(coeffs: &'a [i64], facet_count: usize, dim: usize) -> Result<Self, anyhow::Error> {
        ensure!(facet_count > 0, "need at least one inequality");
        ensure!(dim > 0, "need positive inequality dimension");
        let cols = dim
            .checked_add(1)
            .ok_or_else(|| anyhow!("inequality dimension too large"))?;
        ensure!(
            coeffs.len() == facet_count.saturating_mul(cols),
            "expected {facet_count}x{cols} coeffs but got {}",
            coeffs.len()
        );
        Ok(Self {
            coeffs,
            facet_count,
            dim,
        })
    }
}

impl InequalitiesI64 for RowMajorInequalitiesI64<'_> {
    fn facet_count(&self) -> usize {
        self.facet_count
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_ {
        let cols = self.dim + 1;
        self.coeffs.chunks_exact(cols)
    }
}

impl HowzatInequalities for RowMajorInequalitiesI64<'_> {
    fn build_howzat_inequality_matrix<N: Num>(
        &self,
    ) -> Result<LpMatrix<N, Inequality>, anyhow::Error> {
        build_howzat_inequalities_i64(self)
    }
}

