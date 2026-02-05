use hullabaloo::{Drum, DrumBases, DrumPromotion, DrumSkin};

#[path = "support/backend.rs"]
mod backend;
mod fixtures;

#[test]
fn drum_bases_promotion_updates_coordinates_split_top() {
    let top = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
    ];
    let bot = vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![2.0, 2.0],
        vec![0.0, 2.0],
    ];
    let bases = DrumBases::new(top.clone(), bot.clone());

    let promotion = DrumPromotion::new(DrumSkin::Top, 1, 2).with_heights(2.0, 3.0);
    let promoted = bases.promoted(promotion);

    assert_eq!(promoted.base_dim(), 3);
    assert_eq!(promoted.top().len(), top.len() + 1);
    assert_eq!(promoted.bot().len(), bot.len());

    for (idx, v) in promoted.bot().iter().enumerate() {
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], bot[idx][0]);
        assert_eq!(v[1], bot[idx][1]);
        let expected = if idx == 2 { 3.0 } else { 0.0 };
        assert_eq!(v[2], expected);
    }

    for (idx, v) in promoted.top().iter().enumerate() {
        assert_eq!(v.len(), 3);
        let expected_xy = if idx <= 1 {
            &top[idx]
        } else if idx == 2 {
            &top[1]
        } else {
            &top[idx - 1]
        };
        assert_eq!(v[0], expected_xy[0]);
        assert_eq!(v[1], expected_xy[1]);
        let expected_z = match idx {
            1 => -2.0,
            2 => 2.0,
            _ => 0.0,
        };
        assert_eq!(v[2], expected_z);
    }
}

#[test]
fn drum_bases_promotion_updates_coordinates_split_bot() {
    let top = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
    ];
    let bot = vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![2.0, 2.0],
        vec![0.0, 2.0],
    ];
    let bases = DrumBases::new(top.clone(), bot.clone());

    let promotion = DrumPromotion::new(DrumSkin::Bot, 0, 3).with_heights(2.0, -7.0);
    let promoted = bases.promoted(promotion);

    assert_eq!(promoted.base_dim(), 3);
    assert_eq!(promoted.top().len(), top.len());
    assert_eq!(promoted.bot().len(), bot.len() + 1);

    for (idx, v) in promoted.top().iter().enumerate() {
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], top[idx][0]);
        assert_eq!(v[1], top[idx][1]);
        let expected = if idx == 3 { -7.0 } else { 0.0 };
        assert_eq!(v[2], expected);
    }

    for (idx, v) in promoted.bot().iter().enumerate() {
        assert_eq!(v.len(), 3);
        let expected_xy = if idx < 2 { &bot[0] } else { &bot[idx - 1] };
        assert_eq!(v[0], expected_xy[0]);
        assert_eq!(v[1], expected_xy[1]);
        let expected_z = match idx {
            0 => -2.0,
            1 => 2.0,
            _ => 0.0,
        };
        assert_eq!(v[2], expected_z);
    }
}

#[test]
fn drum_promotion_auto_avoids_degenerate_lift_choice() {
    let top = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
    ];
    let bot = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![2.0, 0.0],
        vec![0.0, 1.0],
    ];
    let bases = DrumBases::new(top, bot);

    // Note: bad promotion would now panic, so we only test the good case
    let good = DrumPromotion::auto(&bases, DrumSkin::Top, 0).expect("auto promotion");
    bases.promoted(good);
}

#[test]
fn santos_two_step_promotion_increases_width_by_one_each_step() {
    let (top, bot) = fixtures::santos_bases_as_vecs();
    let drum = Drum::<f64>::new(top.clone(), bot.clone());
    assert_eq!(
        backend::drum_width(drum.clone(), top.len(), bot.len()).expect("initial width"),
        6
    );

    let suspension_height = 1.0;
    let lift_height = 1.0e-3;

    let promotion0 = DrumPromotion::auto(drum.bases(), DrumSkin::Top, 0)
        .expect("auto promotion")
        .with_heights(suspension_height, lift_height);
    let drum1 = drum.promote(promotion0);
    assert_eq!(
        backend::drum_width(
            drum1.clone(),
            drum1.bases().top().len(),
            drum1.bases().bot().len()
        )
        .expect("width after first promotion"),
        7
    );

    let promotion1 = DrumPromotion::auto(drum1.bases(), DrumSkin::Bot, 0)
        .expect("auto promotion")
        .with_heights(suspension_height, lift_height);
    let drum2 = drum1.promote(promotion1);
    assert_eq!(
        backend::drum_width(
            drum2.clone(),
            drum2.bases().top().len(),
            drum2.bases().bot().len()
        )
        .expect("width after second promotion"),
        8
    );
}

#[test]
fn santos_bases_can_be_promoted_to_asimpliciality_zero_sizes() {
    let (top, bot) = fixtures::santos_bases_as_vecs();
    let mut bases = DrumBases::new(top, bot);

    for step in 0..38usize {
        let split_skin = if step % 2 == 0 {
            DrumSkin::Top
        } else {
            DrumSkin::Bot
        };
        let promotion = DrumPromotion::auto(&bases, split_skin, 0)
            .expect("auto promotion")
            .with_heights(1.0, 0.5);
        bases = bases.promote(promotion);
    }

    assert_eq!(bases.base_dim(), 42);
    assert_eq!(bases.drum_dim(), 43);
    assert_eq!(bases.top().len(), 43);
    assert_eq!(bases.bot().len(), 43);
    assert_eq!(bases.num_vertices(), 86);
}

#[test]
#[ignore]
fn santos_promotion_width_probe() {
    let (top, bot) = fixtures::santos_bases_as_vecs();
    let mut drum = Drum::<f64>::new(top.clone(), bot.clone());

    let mut widths = vec![
        backend::drum_width(
            drum.clone(),
            drum.bases().top().len(),
            drum.bases().bot().len(),
        )
        .expect("width"),
    ];
    let suspension_height = 1.0;
    let lift_height = 1.0e-3;

    for step in 0..12usize {
        let split_skin = if step % 2 == 0 {
            DrumSkin::Top
        } else {
            DrumSkin::Bot
        };

        let promotion = DrumPromotion::auto(drum.bases(), split_skin, 0)
            .expect("auto promotion")
            .with_heights(suspension_height, lift_height);

        drum = drum.promote(promotion);

        widths.push(
            backend::drum_width(
                drum.clone(),
                drum.bases().top().len(),
                drum.bases().bot().len(),
            )
            .expect("width"),
        );
        eprintln!(
            "step={step} dim={} vertices={} width={}",
            drum.drum_dim(),
            drum.num_vertices(),
            widths.last().unwrap()
        );
    }

    eprintln!("widths={widths:?}");
}

#[test]
#[ignore]
fn santos_promotion_lift_height_sweep() {
    let lift_heights = [1e-1, 1e-2, 1e-3];
    let suspension_height = 1.0;
    let max_steps = 4usize;

    for lift_height in lift_heights {
        let (top, bot) = fixtures::santos_bases_as_vecs();
        let mut drum = Drum::<f64>::new(top, bot);

        for step in 0..max_steps {
            let split_skin = if step % 2 == 0 {
                DrumSkin::Top
            } else {
                DrumSkin::Bot
            };

            let promotion = DrumPromotion::auto(drum.bases(), split_skin, 0)
                .expect("auto promotion")
                .with_heights(suspension_height, lift_height);

            drum = drum.promote(promotion);
        }

        eprintln!("lift_height={lift_height:.1e} ok");
    }
}

#[test]
#[ignore]
fn williamson_k1_promotion_width_probe() {
    let (top, bot) = fixtures::williamson_k1_bases_as_vecs();
    let mut drum = Drum::<f64>::new(top, bot);
    let mut widths = vec![
        backend::drum_width(
            drum.clone(),
            drum.bases().top().len(),
            drum.bases().bot().len(),
        )
        .expect("initial width"),
    ];

    let suspension_height = 1.0;
    let lift_height = 1.0e-3;
    let max_steps = 20usize;

    for step in 0..max_steps {
        let split_skin = if step % 2 == 0 {
            DrumSkin::Top
        } else {
            DrumSkin::Bot
        };

        let promotion = DrumPromotion::auto(drum.bases(), split_skin, 0)
            .expect("auto promotion")
            .with_heights(suspension_height, lift_height);

        drum = drum.promote(promotion);

        widths.push(
            backend::drum_width(
                drum.clone(),
                drum.bases().top().len(),
                drum.bases().bot().len(),
            )
            .expect("width"),
        );
        eprintln!(
            "step={step} dim={} vertices={} width={}",
            drum.drum_dim(),
            drum.num_vertices(),
            widths.last().unwrap()
        );
    }

    eprintln!("widths={widths:?}");
}

#[test]
#[ignore]
fn williamson_k1_lift_height_sweep() {
    let lift_heights = [1e-1, 1e-2, 1e-3];
    let suspension_height = 1.0;
    let max_steps = 4usize;

    for lift_height in lift_heights {
        let (top, bot) = fixtures::williamson_k1_bases_as_vecs();
        let mut drum = Drum::<f64>::new(top, bot);

        for step in 0..max_steps {
            let split_skin = if step % 2 == 0 {
                DrumSkin::Top
            } else {
                DrumSkin::Bot
            };

            let promotion = DrumPromotion::auto(drum.bases(), split_skin, 0)
                .expect("auto promotion")
                .with_heights(suspension_height, lift_height);

            drum = drum.promote(promotion);
        }

        eprintln!("lift_height={lift_height:.1e} ok");
    }
}

#[test]
#[ignore]
fn santos_hirsch_attempt_lift_height_1e_1() {
    let (top, bot) = fixtures::santos_bases_as_vecs();
    let mut drum = Drum::<f64>::new(top, bot);

    let suspension_height = 1.0;
    let lift_height = 1.0e-1;

    for step in 0..38usize {
        let split_skin = if step % 2 == 0 {
            DrumSkin::Top
        } else {
            DrumSkin::Bot
        };

        let promotion = DrumPromotion::auto(drum.bases(), split_skin, 0)
            .expect("auto promotion")
            .with_heights(suspension_height, lift_height);

        drum = drum.promote(promotion);

        if step % 4 == 0 {
            eprintln!(
                "step={step} dim={} vertices={}",
                drum.drum_dim(),
                drum.num_vertices()
            );
        }
    }
}

#[test]
fn unit_square_drum_two_promotions_increase_width() {
    let square = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
        vec![0.0, 1.0],
    ];
    let drum = Drum::<f64>::new(square.clone(), square);
    assert_eq!(
        backend::drum_width(
            drum.clone(),
            drum.bases().top().len(),
            drum.bases().bot().len()
        )
        .expect("initial width"),
        2
    );
    let suspension_height = 1.0;
    let lift_height = 1.0e-3;

    let promotion0 = DrumPromotion::auto(drum.bases(), DrumSkin::Top, 0)
        .expect("auto promotion")
        .with_heights(suspension_height, lift_height);
    let drum1 = drum.promote(promotion0);
    assert_eq!(
        backend::drum_width(
            drum1.clone(),
            drum1.bases().top().len(),
            drum1.bases().bot().len()
        )
        .expect("width after first promotion"),
        3
    );
    assert_eq!(drum1.drum_dim(), 4);
    assert_eq!(drum1.num_vertices(), 9);

    let promotion1 = DrumPromotion::auto(drum1.bases(), DrumSkin::Bot, 0)
        .expect("auto promotion")
        .with_heights(suspension_height, lift_height);
    let drum2 = drum1.promote(promotion1);
    assert_eq!(
        backend::drum_width(
            drum2.clone(),
            drum2.bases().top().len(),
            drum2.bases().bot().len()
        )
        .expect("width after second promotion"),
        4
    );
    assert_eq!(drum2.drum_dim(), 5);
    assert_eq!(drum2.num_vertices(), 10);
    // Note: Further promotions would return PromotionError due to simpliciality constraints
}
