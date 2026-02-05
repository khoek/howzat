use hullabaloo::types::{ColSet, RowSet};

#[test]
#[should_panic(expected = "IdSet::insert")]
fn rowset_insert_out_of_range_panics() {
    let mut set = RowSet::new(3);
    set.insert(3);
}

#[test]
#[should_panic(expected = "IdSet::remove")]
fn rowset_remove_out_of_range_panics() {
    let mut set = RowSet::new(3);
    set.remove(3);
}

#[test]
#[should_panic(expected = "IdSet::contains")]
fn rowset_contains_out_of_range_panics() {
    let set = RowSet::new(3);
    let _ = set.contains(3);
}

#[test]
#[should_panic(expected = "IdSet::insert")]
fn colset_insert_out_of_range_panics() {
    let mut set = ColSet::new(3);
    set.insert(3);
}

#[test]
#[should_panic(expected = "IdSet::remove")]
fn colset_remove_out_of_range_panics() {
    let mut set = ColSet::new(3);
    set.remove(3);
}

#[test]
#[should_panic(expected = "IdSet::contains")]
fn colset_contains_out_of_range_panics() {
    let set = ColSet::new(3);
    let _ = set.contains(3);
}
