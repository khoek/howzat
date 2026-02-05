#[path = "support/common.rs"]
mod common;

use std::path::Path;

use calculo::num::Num;
use howzat::dd::{BasisInitialization, ConeOptions, EnumerationMode};
use howzat::polyhedron::{Polyhedron, PolyhedronOptions, PolyhedronOutput};
use hullabaloo::set_family::SetFamily;
use hullabaloo::types::{
    AdjacencyOutput, Generator, IncidenceOutput, Inequality, Representation, RepresentationKind,
    RowId,
};
use rstest::rstest;

use common::{TestNumber, format_matrix, normalize_matrix, parse_cdd_file, parse_set_family_file};

fn options_for(rep: RepresentationKind) -> ConeOptions {
    let _ = rep;
    let mut builder = ConeOptions::builder();
    builder.enumeration_mode(EnumerationMode::Exact);
    builder.basis_initialization(BasisInitialization::Top);
    builder.finish().expect("conversion options")
}

fn convert_ok(path: &str, _expect_out: RepresentationKind) {
    let parsed = parse_cdd_file(&std::path::PathBuf::from(path)).expect("parse input");
    let options = options_for(parsed.representation);
    let eps = TestNumber::default_eps();
    match parsed.matrix {
        common::ParsedMatrix::Inequality(mat) => {
            Polyhedron::builder(mat)
                .cone_options(options.clone())
                .run_dd_with_eps(eps)
                .expect("convert");
        }
        common::ParsedMatrix::Generator(mat) => {
            PolyhedronOutput::<_, Generator>::builder(mat)
                .cone_options(options.clone())
                .run_dd_with_eps(eps)
                .expect("convert");
        }
    }
}

fn assert_matrix_matches<R: Representation>(
    label: &str,
    expected: &howzat::matrix::LpMatrix<TestNumber, R>,
    actual: &howzat::matrix::LpMatrix<TestNumber, R>,
) {
    let norm_expected = normalize_matrix(expected);
    let norm_actual = normalize_matrix(actual);
    let tol = 1.0e-6;
    let approx_equal = norm_expected.row_count() == norm_actual.row_count()
        && norm_expected.col_count() == norm_actual.col_count()
        && norm_expected.linearity() == norm_actual.linearity()
        && {
            let mut used = vec![false; norm_actual.row_count()];
            norm_expected.rows().all(|erow| {
                norm_actual
                    .rows()
                    .enumerate()
                    .find(|(idx, arow)| {
                        !used[*idx]
                            && erow
                                .iter()
                                .zip(arow.iter())
                                .all(|(x, y)| (*x - *y).abs() <= tol)
                    })
                    .map(|(idx, _)| {
                        used[idx] = true;
                        true
                    })
                    .unwrap_or(false)
            })
        };
    if !approx_equal {
        let fmt_expected = format_matrix(&norm_expected);
        let fmt_actual = format_matrix(&norm_actual);
        assert_eq!(
            fmt_expected, fmt_actual,
            "matrix mismatch for {label}\nexpected:\n{fmt_expected}\nactual:\n{fmt_actual}"
        );
    }
}

fn assert_set_family_matches(label: &str, expected: &SetFamily, actual: &SetFamily) {
    if expected.family_size() != actual.family_size() {
        panic!("set family mismatch for {label}");
    }
    let canon = |sf: &SetFamily| {
        let mut out: Vec<Vec<usize>> = sf
            .sets()
            .iter()
            .map(|s| {
                let mut elems: Vec<usize> = s.iter().map(|r| r.as_index()).collect();
                elems.sort_unstable();
                elems
            })
            .collect();
        out.sort();
        out
    };
    let exp = canon(expected);
    let act = canon(actual);
    if exp != act {
        let first_diff = exp
            .iter()
            .zip(act.iter())
            .find(|(e, a)| e != a)
            .map(|(e, a)| format!("expected {:?} got {:?}", e, a))
            .unwrap_or_else(|| "different cardinalities".to_string());
        panic!("set family mismatch for {label}: {first_diff}");
    }
}

fn row_permutation<R: Representation>(
    expected: &howzat::matrix::LpMatrix<TestNumber, R>,
    actual: &howzat::matrix::LpMatrix<TestNumber, R>,
    tol: TestNumber,
) -> Option<Vec<usize>> {
    if expected.row_count() != actual.row_count() || expected.col_count() != actual.col_count() {
        return None;
    }

    let proportional = |a: &[TestNumber], b: &[TestNumber]| -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut scale: Option<TestNumber> = None;
        for (x, y) in a.iter().zip(b.iter()) {
            if y.abs() <= tol {
                if x.abs() > tol {
                    return false;
                }
                continue;
            }
            let candidate = x / y;
            if let Some(s) = scale {
                if (candidate - s).abs() > tol {
                    return false;
                }
            } else {
                scale = Some(candidate);
            }
        }
        true
    };

    let mut used = vec![false; expected.row_count()];
    let mut perm = vec![usize::MAX; actual.row_count()];
    for (a_idx, a_row) in actual.rows().enumerate() {
        let mut found = None;
        for (e_idx, e_row) in expected.rows().enumerate() {
            if used[e_idx] {
                continue;
            }
            if proportional(a_row, e_row) {
                found = Some(e_idx);
                break;
            }
        }
        let e_idx = found?;
        used[e_idx] = true;
        perm[a_idx] = e_idx;
    }
    Some(perm)
}

fn remap_set_family(family: &SetFamily, perm: &[usize]) -> SetFamily {
    let capacity = family.set_capacity().max(perm.len());
    let mut builder = SetFamily::builder(family.family_size(), capacity);
    for set_idx in 0..family.family_size() {
        builder.clear_set(set_idx);
        if let Some(set) = family.set(set_idx) {
            for row in set.iter() {
                let mapped = perm.get(row.as_index()).copied().unwrap_or(row.as_index());
                builder.insert_into_set(set_idx, RowId::new(mapped));
            }
        }
    }
    builder.build()
}

fn reorder_set_family(family: &SetFamily, perm: &[usize]) -> SetFamily {
    let mut builder = SetFamily::builder(family.family_size(), family.set_capacity());
    for (actual_idx, mapped_idx) in perm.iter().copied().enumerate() {
        if mapped_idx >= family.family_size() {
            continue;
        }
        builder.clear_set(mapped_idx);
        if let Some(set) = family.set(actual_idx) {
            for row in set.iter() {
                builder.insert_into_set(mapped_idx, row);
            }
        }
    }
    builder.build()
}

#[rstest]
fn converts_additional_3d_h_inputs(
    #[values("grcubocta.ine", "rcubocta.ine", "rhomtria.ine", "hexocta.ine")] name: &str,
) {
    convert_ok(
        &format!("tests/data/examples-ine3d/{name}"),
        RepresentationKind::Generator,
    );
}

#[rstest]
fn converts_additional_generator_inputs(
    #[values(
        "ccc5.ext",
        "ccc6.ext",
        "ccp5.ext",
        "ccp6.ext",
        "cyclic12-6.ext",
        "cyclic14-8.ext",
        "cyclic16-10.ext",
        "irbox20-4.ext",
        "reg24-5.ext"
    )]
    name: &str,
) {
    convert_ok(
        &format!("tests/data/examples-ext/{name}"),
        RepresentationKind::Inequality,
    );
}

#[rstest]
fn converts_additional_h_inputs(#[values("cube10.ine", "cube12.ine", "reg24-5.ine")] name: &str) {
    convert_ok(
        &format!("tests/data/examples-ine/{name}"),
        RepresentationKind::Generator,
    );
}

fn load_expected_generator_output(base: &str, dir: &str) -> common::ParsedInput {
    let ext_path = format!("{dir}/{base}.ext");
    if Path::new(&ext_path).exists() {
        parse_cdd_file(&std::path::PathBuf::from(ext_path))
            .expect("parse expected generator output")
    } else if Path::new("tests/data/cddexec/ext-from-ine")
        .join(format!("{base}.ext"))
        .exists()
    {
        parse_cdd_file(&Path::new("tests/data/cddexec/ext-from-ine").join(format!("{base}.ext")))
            .expect("parse generated generator output")
    } else {
        panic!("missing expected generator output for {base}");
    }
}

fn load_expected_inequality_output(base: &str, dir: &str) -> common::ParsedInput {
    let ine_path = format!("{dir}/{base}.ine");
    if Path::new(&ine_path).exists() {
        parse_cdd_file(&std::path::PathBuf::from(ine_path))
            .expect("parse expected inequality output")
    } else if Path::new("tests/data/cddexec/ine-from-ext")
        .join(format!("{base}.ine"))
        .exists()
    {
        parse_cdd_file(&Path::new("tests/data/cddexec/ine-from-ext").join(format!("{base}.ine")))
            .expect("parse generated inequality output")
    } else {
        panic!("missing expected inequality output for {base}");
    }
}

#[rstest]
#[ignore = "FIXME currently broken"]
fn inequality_inputs_match_cdd_aux(
    #[values("grcubocta", "rcubocta", "rhomtria", "hexocta")] name: &str,
) {
    let base = format!("tests/data/examples-ine3d/{name}");
    let parsed =
        parse_cdd_file(&std::path::PathBuf::from(format!("{base}.ine"))).expect("parse input");
    let eps = TestNumber::default_eps();
    let poly = PolyhedronOutput::<_, Inequality>::builder(
        parsed
            .matrix
            .as_inequality()
            .expect("inequality input")
            .clone(),
    )
    .cone_options(options_for(parsed.representation))
    .polyhedron_options(PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        input_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::List,
        input_adjacency: AdjacencyOutput::List,
        ..PolyhedronOptions::default()
    })
    .run_dd_with_eps(eps)
    .expect("convert");

    let expected_output = parse_cdd_file(&std::path::PathBuf::from(format!("{base}.ext")))
        .expect("parse expected generator output")
        .matrix
        .as_generator()
        .expect("generator matrix from ext")
        .clone();
    let perm = row_permutation(&expected_output, poly.output(), 1.0e-6)
        .unwrap_or_else(|| (0..poly.output().row_count()).collect());
    assert_matrix_matches(name, &expected_output, poly.output());

    let expected_incidence =
        parse_set_family_file(Path::new(&format!("{base}.ecd"))).expect("parse ecd");
    let actual_incidence_remap =
        reorder_set_family(poly.incidence().expect("output incidence"), &perm);
    assert_set_family_matches(
        &format!("{name} output incidence"),
        &expected_incidence,
        &actual_incidence_remap,
    );

    let expected_input_incidence =
        parse_set_family_file(Path::new(&format!("{base}.icd"))).expect("parse icd");
    let actual_input_incidence =
        remap_set_family(poly.input_incidence().expect("input incidence"), &perm);
    assert_set_family_matches(
        &format!("{name} input incidence"),
        &expected_input_incidence,
        &actual_input_incidence,
    );

    let expected_output_adjacency =
        parse_set_family_file(Path::new(&format!("{base}.ead"))).expect("parse ead");
    let actual_output_adjacency =
        remap_set_family(poly.adjacency().expect("output adjacency"), &perm);
    assert_set_family_matches(
        &format!("{name} output adjacency"),
        &expected_output_adjacency,
        &actual_output_adjacency,
    );

    let expected_input_adjacency =
        parse_set_family_file(Path::new(&format!("{base}.iad"))).expect("parse iad");
    assert_set_family_matches(
        &format!("{name} input adjacency"),
        &expected_input_adjacency,
        poly.input_adjacency().expect("input adjacency"),
    );
}

#[rstest]
#[ignore = "FIXME currently broken"]
fn generator_inputs_match_cdd_aux(
    #[values("grcubocta", "rcubocta", "rhomtria", "hexocta")] name: &str,
) {
    let base = format!("tests/data/examples-ine3d/{name}");
    let parsed =
        parse_cdd_file(&std::path::PathBuf::from(format!("{base}.ext"))).expect("parse input ext");
    let eps = TestNumber::default_eps();
    let poly = PolyhedronOutput::<_, Generator>::builder(
        parsed
            .matrix
            .as_generator()
            .expect("generator input")
            .clone(),
    )
    .cone_options(options_for(parsed.representation))
    .polyhedron_options(PolyhedronOptions {
        output_incidence: IncidenceOutput::Set,
        input_incidence: IncidenceOutput::Set,
        output_adjacency: AdjacencyOutput::List,
        input_adjacency: AdjacencyOutput::List,
        ..PolyhedronOptions::default()
    })
    .run_dd_with_eps(eps)
    .expect("convert");

    let expected_output = parse_cdd_file(&std::path::PathBuf::from(format!("{base}.ine")))
        .expect("parse expected inequality output")
        .matrix
        .as_inequality()
        .expect("inequality matrix from ine")
        .clone();
    assert_matrix_matches(name, &expected_output, poly.output());

    let expected_incidence =
        parse_set_family_file(Path::new(&format!("{base}.icd"))).expect("parse icd");
    assert_set_family_matches(
        &format!("{name} output incidence"),
        &expected_incidence,
        poly.incidence().expect("output incidence"),
    );

    let expected_input_incidence =
        parse_set_family_file(Path::new(&format!("{base}.ecd"))).expect("parse ecd");
    assert_set_family_matches(
        &format!("{name} input incidence"),
        &expected_input_incidence,
        poly.input_incidence().expect("input incidence"),
    );

    let expected_output_adjacency =
        parse_set_family_file(Path::new(&format!("{base}.iad"))).expect("parse iad");
    assert_set_family_matches(
        &format!("{name} output adjacency"),
        &expected_output_adjacency,
        poly.adjacency().expect("output adjacency"),
    );

    let expected_input_adjacency =
        parse_set_family_file(Path::new(&format!("{base}.ead"))).expect("parse ead");
    assert_set_family_matches(
        &format!("{name} input adjacency"),
        &expected_input_adjacency,
        poly.input_adjacency().expect("input adjacency"),
    );
}

#[rstest]
#[ignore = "FIXME currently broken"]
fn generator_examples_match_cddexec(
    #[values(
        "ccc5",
        "ccc6",
        "ccp5",
        "ccp6",
        "cyclic12-6",
        "cyclic14-8",
        "cyclic16-10",
        "irbox20-4",
        "reg24-5"
    )]
    name: &str,
) {
    let ext_path = format!("tests/data/examples-ext/{name}.ext");
    let parsed =
        parse_cdd_file(&std::path::PathBuf::from(&ext_path)).expect("parse generator input");
    let eps = TestNumber::default_eps();
    let poly = PolyhedronOutput::<_, Generator>::builder(
        parsed
            .matrix
            .as_generator()
            .expect("generator input")
            .clone(),
    )
    .cone_options(options_for(parsed.representation))
    .polyhedron_options(PolyhedronOptions::default())
    .run_dd_with_eps(eps)
    .expect("convert");

    let expected = load_expected_inequality_output(name, "tests/data/examples-ine");
    let expected_matrix = expected
        .matrix
        .as_inequality()
        .expect("expected inequality matrix")
        .clone();
    assert_matrix_matches(name, &expected_matrix, poly.output());
}

#[rstest]
#[ignore = "FIXME currently broken"]
fn inequality_examples_match_cddexec(#[values("cube10", "cube12", "reg24-5")] name: &str) {
    let ine_path = format!("tests/data/examples-ine/{name}.ine");
    let parsed =
        parse_cdd_file(&std::path::PathBuf::from(&ine_path)).expect("parse inequality input");
    let eps = TestNumber::default_eps();
    let poly = PolyhedronOutput::<_, Inequality>::builder(
        parsed
            .matrix
            .as_inequality()
            .expect("inequality input")
            .clone(),
    )
    .cone_options(options_for(parsed.representation))
    .polyhedron_options(PolyhedronOptions::default())
    .run_dd_with_eps(eps)
    .expect("convert");

    let expected = load_expected_generator_output(name, "tests/data/examples-ext");
    let expected_matrix = expected
        .matrix
        .as_generator()
        .expect("expected generator matrix")
        .clone();
    assert_matrix_matches(name, &expected_matrix, poly.output());
}
