#![allow(dead_code)]

use std::collections::VecDeque;

pub const SANTOS_TOP_BASE: [[f64; 4]; 24] = [
    [18.0, 0.0, 0.0, 0.0],
    [-18.0, 0.0, 0.0, 0.0],
    [0.0, 18.0, 0.0, 0.0],
    [0.0, -18.0, 0.0, 0.0],
    [0.0, 0.0, 45.0, 0.0],
    [0.0, 0.0, -45.0, 0.0],
    [0.0, 0.0, 0.0, 45.0],
    [0.0, 0.0, 0.0, -45.0],
    [15.0, 15.0, 0.0, 0.0],
    [-15.0, 15.0, 0.0, 0.0],
    [15.0, -15.0, 0.0, 0.0],
    [-15.0, -15.0, 0.0, 0.0],
    [0.0, 0.0, 30.0, 30.0],
    [0.0, 0.0, -30.0, 30.0],
    [0.0, 0.0, 30.0, -30.0],
    [0.0, 0.0, -30.0, -30.0],
    [0.0, 10.0, 40.0, 0.0],
    [0.0, -10.0, 40.0, 0.0],
    [0.0, 10.0, -40.0, 0.0],
    [0.0, -10.0, -40.0, 0.0],
    [10.0, 0.0, 0.0, 40.0],
    [-10.0, 0.0, 0.0, 40.0],
    [10.0, 0.0, 0.0, -40.0],
    [-10.0, 0.0, 0.0, -40.0],
];

pub const SANTOS_BOT_BASE: [[f64; 4]; 24] = [
    [0.0, 0.0, 0.0, 18.0],
    [0.0, 0.0, 0.0, -18.0],
    [0.0, 0.0, 18.0, 0.0],
    [0.0, 0.0, -18.0, 0.0],
    [45.0, 0.0, 0.0, 0.0],
    [-45.0, 0.0, 0.0, 0.0],
    [0.0, 45.0, 0.0, 0.0],
    [0.0, -45.0, 0.0, 0.0],
    [0.0, 0.0, 15.0, 15.0],
    [0.0, 0.0, 15.0, -15.0],
    [0.0, 0.0, -15.0, 15.0],
    [0.0, 0.0, -15.0, -15.0],
    [30.0, 30.0, 0.0, 0.0],
    [-30.0, 30.0, 0.0, 0.0],
    [30.0, -30.0, 0.0, 0.0],
    [-30.0, -30.0, 0.0, 0.0],
    [40.0, 0.0, 10.0, 0.0],
    [40.0, 0.0, -10.0, 0.0],
    [-40.0, 0.0, 10.0, 0.0],
    [-40.0, 0.0, -10.0, 0.0],
    [0.0, 40.0, 0.0, 10.0],
    [0.0, 40.0, 0.0, -10.0],
    [0.0, -40.0, 0.0, 10.0],
    [0.0, -40.0, 0.0, -10.0],
];

pub fn santos_bases_as_vecs() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let top = SANTOS_TOP_BASE.iter().map(|v| v.to_vec()).collect();
    let bot = SANTOS_BOT_BASE.iter().map(|v| v.to_vec()).collect();
    (top, bot)
}

const WILLIAMSON_K1_MOTIF: [[f64; 5]; 4] = [
    [0.0, 0.0, 3.0, 3.0, 1.0],
    [98.0, 0.0, 1.0, 0.0, 1.0],
    [100.0, 0.0, 0.0, 0.0, 1.0],
    [75.0, 75.0, 0.0, 0.0, 1.0],
];

const WILLIAMSON_K4_MOTIF: [[f64; 5]; 7] = [
    [0.0, 0.0, 3.0, 3.0, 1.0],
    [98.0, 0.0, 1.0, 0.0, 1.0],
    [100.0, 0.0, 0.0, 0.0, 1.0],
    [92.625_394_156_403_93, 23.393_453_858_754_036, 0.0, 0.0, 1.0],
    [85.693_360_076_337_29, 45.047_070_135_648_63, 0.0, 0.0, 1.0],
    [79.594_659_910_781_46, 62.140_358_293_966_72, 0.0, 0.0, 1.0],
    [75.0, 75.0, 0.0, 0.0, 1.0],
];

fn williamson_bases_from_motif(motif: &[[f64; 5]]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut orbit: Vec<[f64; 5]> = Vec::new();
    let mut queue: VecDeque<[f64; 5]> = VecDeque::new();

    for &v in motif {
        if !orbit.iter().any(|seen| seen == &v) {
            orbit.push(v);
            queue.push_back(v);
        }
    }

    while let Some(v) = queue.pop_front() {
        let epsilons = (0..4).map(|i| {
            let mut flipped = v;
            flipped[i] = -flipped[i];
            flipped
        });
        let tau = [v[2], v[3], v[1], v[0], -v[4]];

        for candidate in epsilons.chain(std::iter::once(tau)) {
            if orbit.iter().any(|seen| seen == &candidate) {
                continue;
            }
            orbit.push(candidate);
            queue.push_back(candidate);
        }
    }

    let mut top = Vec::new();
    let mut bot = Vec::new();
    for v in orbit {
        let base = vec![v[0], v[1], v[2], v[3]];
        if v[4].is_sign_positive() {
            top.push(base);
        } else {
            bot.push(base);
        }
    }
    (top, bot)
}

pub fn williamson_k1_bases_as_vecs() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    williamson_bases_from_motif(&WILLIAMSON_K1_MOTIF)
}

pub fn williamson_k4_bases_as_vecs() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    williamson_bases_from_motif(&WILLIAMSON_K4_MOTIF)
}
