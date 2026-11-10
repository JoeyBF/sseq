//! Prints the first $\tau$-Bockstein differential's ranks on $\mathrm{Ext}_P(Q)$ from the untwisted resolution.

use std::sync::Arc;

use ext::{
    algebraic_novikov::unit_module,
    chain_complex::ChainComplex,
    tau_lift::{Bar, Resolved, TauLift},
};
use sseq::coordinates::Bidegree;

/// Usage: `tau_lift <max_h> <max_t>`; prints `stem,s,weight,dim E_1,rank d_1` by source.
fn main() {
    let args: Vec<i32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (max_h, max_t) = (args[0], args[1]);
    let start = std::time::Instant::now();
    if let Some(top) = args.get(2) {
        // The bar resolution through filtration `top`: print m_3 and m_4 on basis elements.
        let bar = Bar::new(*top, max_t / 2);
        let lift = TauLift::new(&bar, max_h, max_t);
        eprintln!("lifted the bar resolution: {:?}", start.elapsed());
        for (g, g2) in lift.delta1_terms() {
            let z = &bar.gens[g.0 as usize][g.1.0 as usize][g.1.1 as usize];
            let x = &bar.gens[g2.0 as usize][g2.1.0 as usize][g2.1.1 as usize];
            let ps = |v: &Vec<u128>| {
                v.iter()
                    .map(|&r| format!("{:?}", ext::tau_lift::unpack_r(r)))
                    .collect::<Vec<_>>()
                    .join("|")
            };
            println!(
                "[{}] K={:?} <- [{}] K'={:?}",
                ps(z),
                &g.2[..5],
                ps(x),
                &g2.2[..5]
            );
        }
        return;
    }
    let res = ext::nassau::Resolution::new(Arc::new(unit_module()));
    if std::env::var("KOSZUL").is_ok() {
        res.compute_through_bidegree(Bidegree::s_t(max_h + 1, max_t / 2));
        let lift = ext::koszul_lift::KoszulLift::new(&Resolved(&res), max_h, max_t);
        eprintln!("Koszul gauge: {:?}", start.elapsed());
        eprintln!(
            "δ_0 δ_1 + δ_1 δ_0 fails on {} cochains",
            lift.check_cocycle()
        );
        if std::env::var("CLASSICAL").is_ok() {
            let delta = lift.all_orders();
            eprintln!("all orders: {:?}", start.elapsed());
            for ((n, s), d) in lift.classical(&delta) {
                println!("{n},{s},{d}");
            }
        } else {
            for ((n, s, w), (dim, rank)) in lift.pages() {
                println!("{n},{s},{w},{dim},{rank}");
            }
        }
        eprintln!("total {:?}", start.elapsed());
        return;
    }
    res.compute_through_bidegree(Bidegree::s_t(max_h, max_t / 2));
    let lift = TauLift::new(&Resolved(&res), max_h, max_t);
    eprintln!("lifted: {:?}", start.elapsed());
    lift.check_square_zero();
    eprintln!("D^2 = 0 checked");
    eprintln!("δ_1 shapes (Δa, Δk): {:?}", lift.delta1_shapes());
    eprintln!(
        "δ_1 backbone pairs, local: {:?}",
        lift.delta1_locality(&res)
    );
    if std::env::var("TAU_TERMS").is_ok() {
        for (g, g2) in lift.delta1_terms() {
            let k: Vec<_> = g.2.iter().take(5).collect();
            let k2: Vec<_> = g2.2.iter().take(5).collect();
            eprintln!(
                "z=({}, {:?}) K={k:?}  <-  x=({}, {:?}) K'={k2:?}",
                g.0, g.1, g2.0, g2.1
            );
        }
    }
    let (pages, shift) = lift.pages();
    eprintln!("d_1 weight shift: {shift:?}; total {:?}", start.elapsed());
    for ((n, s, w), (dim, rank)) in pages {
        println!("{n},{s},{w},{dim},{rank}");
    }
}
