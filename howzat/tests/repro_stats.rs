#[cfg(test)]
mod tests {
    use calculo::num::Num;
    use howzat::dd::ConeOptions;
    use howzat::matrix::LpMatrixBuilder;
    use howzat::polyhedron::{Polyhedron, PolyhedronOptions};
    use hullabaloo::types::IncidenceOutput;

    #[test]
    fn test_simplex_stats() {
        // Define a 1D simplex (segment) [0, 1] in R^1.
        // Inequalities: x >= 0, x <= 1.
        // Homogenized (x, x0): x >= 0, x - x0 <= 0 => -x + x0 >= 0.
        // Matrix rows:
        // 1. 1*x + 0*x0 >= 0  -> [0, 1] (if col 0 is x0) or [1, 0] (if col 1 is x0)
        // cddlib usually puts constant at index 0.
        // So x0 is col 0.
        // x >= 0  => 0*x0 + 1*x >= 0 -> [0, 1]
        // 1 - x >= 0 => 1*x0 - 1*x >= 0 -> [1, -1]

        let rows = vec![vec![0.0, 1.0], vec![1.0, -1.0]];

        let mat = LpMatrixBuilder::from_rows(rows).build();

        let eps = f64::default_eps();
        let poly_options = PolyhedronOptions {
            output_incidence: IncidenceOutput::Set,
            input_incidence: IncidenceOutput::Set,
            ..PolyhedronOptions::default()
        };
        let poly = Polyhedron::builder(mat)
            .cone_options(ConeOptions::default())
            .polyhedron_options(poly_options)
            .run_dd_with_eps(eps)
            .unwrap();

        let _ = poly.dimension();
        let _ = poly.output_size();
        let _ = poly.linearity_dimension();

        // Expected for polytope: Dim 1, Vertices 2 (0 and 1).
        // Expected for cone: Dim 2, Generators 2 ((1,0) is ray? No, (1,0) is x=0, x0=1 => 0. (1,1) is x=1, x0=1 => 1).
        // Wait, x0=0 => x>=0, -x>=0 => x=0. Ray (0, 1)? No, (0, x) -> (0, 1) is x=1, x0=0.
        // Wait, x0=0 => x>=0, -x>=0 => x=0. Ray (0, x) -> (0, 1) is x=1, x0=0.
        // 0*0 + 1*1 = 1 >= 0.
        // 1*0 - 1*1 = -1 < 0. Not feasible.
        // So x0=0 implies x=0. Ray (0,0) is trivial.
        // So only vertices.

        // Let's inspect the output rays.
        let output = poly.output();
        let _ = output.row_count();
        for i in 0..output.row_count() {
            let row = output.row(i);
            let _ = row.iter().cloned().collect::<Vec<_>>();
        }

        let _ = poly.homogeneous();
        let _ = poly.equality_kinds();
        let _ = poly.redundant_rows();
        let _ = poly.dominant_rows();
        let _ = poly.incidence();
    }
}
