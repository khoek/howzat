#[path = "support/common.rs"]
mod common;

use common::parse_cdd_file;

#[test]
fn parses_projection_directives() {
    let proj1 = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/project1.ine",
    ))
    .expect("parse project1");
    assert!(
        proj1.project.is_some(),
        "project1.ine should contain projection metadata"
    );

    let proj2 = parse_cdd_file(&std::path::PathBuf::from(
        "tests/data/examples/project2.ine",
    ))
    .expect("parse project2");
    assert!(
        proj2.project.is_some(),
        "project2.ine should contain projection metadata"
    );
}
