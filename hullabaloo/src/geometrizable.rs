use crate::matrix::Matrix;
use calculo::num::Num;

pub trait Geometrizable: Clone {
    type N: Num;

    fn into_vertices(self) -> Vec<Vec<Self::N>>;

    /// Convert vertices into a generator matrix (rows are [1, x1, x2, ...]).
    ///
    /// # Panics
    ///
    /// Panics if there are no vertices, vertex dimension is zero, or dimensions are inconsistent.
    fn into_matrix(self) -> Matrix<Self::N>
    where
        Self: Sized,
    {
        let vertices = self.into_vertices();
        let first = vertices
            .first()
            .expect("conversion requires at least one vertex");

        let dim = first.len();
        assert!(dim > 0, "conversion requires positive vertex dimension");
        assert!(
            vertices.iter().skip(1).all(|v| v.len() == dim),
            "conversion requires consistent vertex dimensions"
        );

        let generator_rows: Vec<Vec<Self::N>> = vertices
            .into_iter()
            .map(|coords| {
                let mut row = Vec::with_capacity(coords.len() + 1);
                row.push(Self::N::one());
                row.extend(coords);
                row
            })
            .collect();

        Matrix::from_rows(generator_rows)
    }
}

impl<N: Num> Geometrizable for Vec<Vec<N>> {
    type N = N;

    fn into_vertices(self) -> Vec<Vec<Self::N>> {
        self
    }
}
