use anyhow::{anyhow, ensure};
use calculo::num::Num;
use howzat::matrix::{LpMatrix, LpMatrixBuilder};
use hullabaloo::types::Generator;

pub(crate) trait VerticesF64 {
    fn vertex_count(&self) -> usize;
    fn dim(&self) -> usize;
    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_;
}

pub(crate) trait VerticesI64 {
    fn vertex_count(&self) -> usize;
    fn dim(&self) -> usize;
    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_;
}

pub(crate) trait HowzatVertices {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error>;
}

fn build_howzat_matrix_f64<N: Num, V: VerticesF64 + ?Sized>(
    vertices: &V,
) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
    let vertex_count = vertices.vertex_count();
    ensure!(vertex_count > 0, "howzat needs at least one vertex");
    let dim = vertices.dim();
    ensure!(dim > 0, "howzat vertices must have positive dimension");

    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("howzat vertex dimension too large"))?;
    let total = vertex_count
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("howzat generator matrix too large"))?;

    let mut data = Vec::with_capacity(total);
    for (row, coords) in vertices.rows().enumerate() {
        ensure!(
            coords.len() == dim,
            "howzat vertices must have consistent dimension (expected {dim}, got {} at row {row})",
            coords.len()
        );

        data.push(N::one());
        for &value in coords {
            data.push(
                N::try_from_f64(value).ok_or_else(|| anyhow!("non-finite vertex coordinate {value}"))?,
            );
        }
    }

    Ok(LpMatrixBuilder::<N, Generator>::from_flat(vertex_count, cols, data).build())
}

fn build_howzat_matrix_i64<N: Num, V: VerticesI64 + ?Sized>(
    vertices: &V,
) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
    let vertex_count = vertices.vertex_count();
    ensure!(vertex_count > 0, "howzat needs at least one vertex");
    let dim = vertices.dim();
    ensure!(dim > 0, "howzat vertices must have positive dimension");

    let cols = dim
        .checked_add(1)
        .ok_or_else(|| anyhow!("howzat vertex dimension too large"))?;
    let total = vertex_count
        .checked_mul(cols)
        .ok_or_else(|| anyhow!("howzat generator matrix too large"))?;

    let mut data = Vec::with_capacity(total);
    for (row, coords) in vertices.rows().enumerate() {
        ensure!(
            coords.len() == dim,
            "howzat vertices must have consistent dimension (expected {dim}, got {} at row {row})",
            coords.len()
        );

        data.push(N::one());
        for &value in coords {
            let mut out = N::from_u64(value.unsigned_abs());
            if value.is_negative() {
                out = -out;
            }
            data.push(out);
        }
    }

    Ok(LpMatrixBuilder::<N, Generator>::from_flat(vertex_count, cols, data).build())
}

impl VerticesF64 for [Vec<f64>] {
    fn vertex_count(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.iter().map(Vec::as_slice)
    }
}

impl HowzatVertices for [Vec<f64>] {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        build_howzat_matrix_f64(self)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct HomogeneousGeneratorRowsF64<'a> {
    rows: &'a [Vec<f64>],
    row_count: usize,
    cols: usize,
}

impl<'a> HomogeneousGeneratorRowsF64<'a> {
    pub(crate) fn new(rows: &'a [Vec<f64>]) -> Result<Self, anyhow::Error> {
        let row_count = rows.len();
        ensure!(row_count > 0, "need at least one generator row");
        let cols = rows.first().map_or(0, Vec::len);
        ensure!(cols > 1, "need generator width at least 2");
        for (row, generator) in rows.iter().enumerate() {
            ensure!(
                generator.len() == cols,
                "howzat generator matrix must have consistent width (expected {cols}, got {} at row {row})",
                generator.len()
            );
        }
        Ok(Self {
            rows,
            row_count,
            cols,
        })
    }
}

impl VerticesF64 for HomogeneousGeneratorRowsF64<'_> {
    fn vertex_count(&self) -> usize {
        self.row_count
    }

    fn dim(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.rows.iter().map(Vec::as_slice)
    }
}

impl HowzatVertices for HomogeneousGeneratorRowsF64<'_> {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        let row_count = self.row_count;
        let cols = self.cols;
        let total = row_count
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("howzat generator matrix too large"))?;

        let mut data = Vec::with_capacity(total);
        for &value in self.rows().flatten() {
            data.push(
                N::try_from_f64(value)
                    .ok_or_else(|| anyhow!("non-finite generator coefficient {value}"))?,
            );
        }

        Ok(LpMatrixBuilder::<N, Generator>::from_flat(row_count, cols, data).build())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RowMajorVertices<'a> {
    coords: &'a [f64],
    vertex_count: usize,
    dim: usize,
}

impl<'a> RowMajorVertices<'a> {
    pub(crate) fn new(
        coords: &'a [f64],
        vertex_count: usize,
        dim: usize,
    ) -> Result<Self, anyhow::Error> {
        ensure!(vertex_count > 0, "need at least one vertex");
        ensure!(dim > 0, "need positive vertex dimension");
        ensure!(
            coords.len() == vertex_count.saturating_mul(dim),
            "expected {vertex_count}x{dim} coords but got {}",
            coords.len()
        );
        Ok(Self {
            coords,
            vertex_count,
            dim,
        })
    }
}

impl VerticesF64 for RowMajorVertices<'_> {
    fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.coords.chunks_exact(self.dim)
    }
}

impl HowzatVertices for RowMajorVertices<'_> {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        build_howzat_matrix_f64(self)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RowMajorHomogeneousGenerators<'a> {
    coords: &'a [f64],
    row_count: usize,
    cols: usize,
}

impl<'a> RowMajorHomogeneousGenerators<'a> {
    pub(crate) fn new(coords: &'a [f64], row_count: usize, cols: usize) -> Result<Self, anyhow::Error> {
        ensure!(row_count > 0, "need at least one generator row");
        ensure!(cols > 1, "need generator width at least 2");
        ensure!(
            coords.len() == row_count.saturating_mul(cols),
            "expected {row_count}x{cols} generator entries but got {}",
            coords.len()
        );
        Ok(Self {
            coords,
            row_count,
            cols,
        })
    }
}

impl VerticesF64 for RowMajorHomogeneousGenerators<'_> {
    fn vertex_count(&self) -> usize {
        self.row_count
    }

    fn dim(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> impl Iterator<Item = &[f64]> + '_ {
        self.coords.chunks_exact(self.cols)
    }
}

impl HowzatVertices for RowMajorHomogeneousGenerators<'_> {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        let row_count = self.row_count;
        let cols = self.cols;
        let total = row_count
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("howzat generator matrix too large"))?;

        let mut data = Vec::with_capacity(total);
        for (row, generator) in self.rows().enumerate() {
            ensure!(
                generator.len() == cols,
                "howzat generator matrix must have consistent width (expected {cols}, got {} at row {row})",
                generator.len()
            );
            for &value in generator {
                data.push(
                    N::try_from_f64(value)
                        .ok_or_else(|| anyhow!("non-finite generator coefficient {value}"))?,
                );
            }
        }

        Ok(LpMatrixBuilder::<N, Generator>::from_flat(row_count, cols, data).build())
    }
}

impl VerticesI64 for [Vec<i64>] {
    fn vertex_count(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        self.first().map_or(0, Vec::len)
    }

    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_ {
        self.iter().map(Vec::as_slice)
    }
}

impl HowzatVertices for [Vec<i64>] {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        build_howzat_matrix_i64(self)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct HomogeneousGeneratorRowsI64<'a> {
    rows: &'a [Vec<i64>],
    row_count: usize,
    cols: usize,
}

impl<'a> HomogeneousGeneratorRowsI64<'a> {
    pub(crate) fn new(rows: &'a [Vec<i64>]) -> Result<Self, anyhow::Error> {
        let row_count = rows.len();
        ensure!(row_count > 0, "need at least one generator row");
        let cols = rows.first().map_or(0, Vec::len);
        ensure!(cols > 1, "need generator width at least 2");
        for (row, generator) in rows.iter().enumerate() {
            ensure!(
                generator.len() == cols,
                "howzat generator matrix must have consistent width (expected {cols}, got {} at row {row})",
                generator.len()
            );
        }
        Ok(Self {
            rows,
            row_count,
            cols,
        })
    }
}

impl VerticesI64 for HomogeneousGeneratorRowsI64<'_> {
    fn vertex_count(&self) -> usize {
        self.row_count
    }

    fn dim(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_ {
        self.rows.iter().map(Vec::as_slice)
    }
}

impl HowzatVertices for HomogeneousGeneratorRowsI64<'_> {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        let row_count = self.row_count;
        let cols = self.cols;
        let total = row_count
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("howzat generator matrix too large"))?;

        let mut data = Vec::with_capacity(total);
        for &value in self.rows().flatten() {
            let mut out = N::from_u64(value.unsigned_abs());
            if value.is_negative() {
                out = -out;
            }
            data.push(out);
        }

        Ok(LpMatrixBuilder::<N, Generator>::from_flat(row_count, cols, data).build())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RowMajorVerticesI64<'a> {
    coords: &'a [i64],
    vertex_count: usize,
    dim: usize,
}

impl<'a> RowMajorVerticesI64<'a> {
    pub(crate) fn new(
        coords: &'a [i64],
        vertex_count: usize,
        dim: usize,
    ) -> Result<Self, anyhow::Error> {
        ensure!(vertex_count > 0, "need at least one vertex");
        ensure!(dim > 0, "need positive vertex dimension");
        ensure!(
            coords.len() == vertex_count.saturating_mul(dim),
            "expected {vertex_count}x{dim} coords but got {}",
            coords.len()
        );
        Ok(Self {
            coords,
            vertex_count,
            dim,
        })
    }
}

impl VerticesI64 for RowMajorVerticesI64<'_> {
    fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_ {
        self.coords.chunks_exact(self.dim)
    }
}

impl HowzatVertices for RowMajorVerticesI64<'_> {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        build_howzat_matrix_i64(self)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RowMajorHomogeneousGeneratorsI64<'a> {
    coords: &'a [i64],
    row_count: usize,
    cols: usize,
}

impl<'a> RowMajorHomogeneousGeneratorsI64<'a> {
    pub(crate) fn new(coords: &'a [i64], row_count: usize, cols: usize) -> Result<Self, anyhow::Error> {
        ensure!(row_count > 0, "need at least one generator row");
        ensure!(cols > 1, "need generator width at least 2");
        ensure!(
            coords.len() == row_count.saturating_mul(cols),
            "expected {row_count}x{cols} generator entries but got {}",
            coords.len()
        );
        Ok(Self {
            coords,
            row_count,
            cols,
        })
    }
}

impl VerticesI64 for RowMajorHomogeneousGeneratorsI64<'_> {
    fn vertex_count(&self) -> usize {
        self.row_count
    }

    fn dim(&self) -> usize {
        self.cols
    }

    fn rows(&self) -> impl Iterator<Item = &[i64]> + '_ {
        self.coords.chunks_exact(self.cols)
    }
}

impl HowzatVertices for RowMajorHomogeneousGeneratorsI64<'_> {
    fn build_howzat_generator_matrix<N: Num>(&self) -> Result<LpMatrix<N, Generator>, anyhow::Error> {
        let row_count = self.row_count;
        let cols = self.cols;
        let total = row_count
            .checked_mul(cols)
            .ok_or_else(|| anyhow!("howzat generator matrix too large"))?;

        let mut data = Vec::with_capacity(total);
        for (row, generator) in self.rows().enumerate() {
            ensure!(
                generator.len() == cols,
                "howzat generator matrix must have consistent width (expected {cols}, got {} at row {row})",
                generator.len()
            );
            for &value in generator {
                let mut out = N::from_u64(value.unsigned_abs());
                if value.is_negative() {
                    out = -out;
                }
                data.push(out);
            }
        }

        Ok(LpMatrixBuilder::<N, Generator>::from_flat(row_count, cols, data).build())
    }
}
