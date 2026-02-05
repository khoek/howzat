#[path = "support/common.rs"]
mod common;

use common::parse_cdd_file;

#[test]
fn parse_projection_results() {
    for name in ["project1res.ine", "project2res.ine"] {
        parse_cdd_file(&std::path::PathBuf::from(format!(
            "tests/data/examples/{}",
            name
        )))
        .expect("parse projection result");
    }
}
