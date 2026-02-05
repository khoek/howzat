use hullabaloo::LinearOrientedMatroid;

#[test]
fn triangle_linear_matroid_charpoly() {
    let columns = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let om = LinearOrientedMatroid::from_columns(columns);

    assert_eq!(om.rank(), 2);
    assert_eq!(om.num_elements(), 3);

    let chi = om.characteristic_polynomial().expect("char poly");
    assert_eq!(chi.coefficients, vec![2, -3, 1]);
}

#[test]
fn square_facets_charpoly() {
    let verts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let poly = cddlib_rs::Polyhedron::<f64>::from_vertices(&verts).expect("square polytope");
    let facets = poly.facets().expect("square facets");

    let mut columns = Vec::with_capacity(facets.rows());
    for row in 0..facets.rows() {
        let mut v = Vec::with_capacity(facets.cols());
        for col in 0..facets.cols() {
            v.push(facets.get_real(row, col));
        }
        columns.push(v);
    }
    let om = LinearOrientedMatroid::from_columns(columns);

    assert_eq!(om.num_elements(), facets.rows());
    assert!(om.rank() >= 2 && om.rank() <= 3);

    let chi = om.characteristic_polynomial().expect("char poly");

    assert_eq!(chi.coefficients, vec![-3, 6, -4, 1]);
}
