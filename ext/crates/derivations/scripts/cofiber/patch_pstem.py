"""P-stem truncation of the untwisted tensor resolution: keep the generators (y, m) of Q_s whose
P-generator y has stem |y| - s <= pstem_max (a subcomplex: each component of the differential goes
to a y' of P-stem at most that of the source), so Q is computed by stem, not by internal degree."""
p = "ext/src/ext_algebra/tensor_resolution.rs"
s = open(p).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("""    lock: Mutex<()>,
}""", """    lock: Mutex<()>,
    /// keep only generators (y, m) with P-stem(y) <= this (see `set_pstem_max`)
    pstem_max: std::sync::atomic::AtomicI32,
}""")
rep("""            lock: Mutex::new(()),
        }""", """            lock: Mutex::new(()),
            pstem_max: std::sync::atomic::AtomicI32::new(i32::MAX),
        }""")
rep("""    /// Build $Q_s$ (free modules and differentials) for all `s ≤ s_max` through internal degree""",
    """    /// Truncate by the stem of the P-generator: only the generators $(y, m)$ of $Q_s$ with
    /// $|y| - s \\le n$ are kept. Each component of $\\partial_Q$ goes from $(x, m)$ to a $(y, m')$
    /// with $|x| - |y| \\ge 1$, so the P-stem never increases and the kept generators span a
    /// subcomplex, which agrees with $Q$ in Q-stems up to $n$; $P$ and the quasi-inverses are then
    /// only needed by stem.
    pub fn set_pstem_max(&self, n: i32) {
        self.pstem_max.store(n, std::sync::atomic::Ordering::Relaxed);
    }

    fn kept(&self, s: i32, gen_deg: i32) -> bool {
        gen_deg - s <= self.pstem_max.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Build $Q_s$ (free modules and differentials) for all `s ≤ s_max` through internal degree""")
rep("""        self.resolution
            .compute_through_bidegree(Bidegree::s_t(s_max, t_max));""",
    """        let pmax = self.pstem_max.load(std::sync::atomic::Ordering::Relaxed);
        if pmax == i32::MAX {
            self.resolution
                .compute_through_bidegree(Bidegree::s_t(s_max, t_max));
        } else {
            self.resolution
                .compute_through_stem(Bidegree::n_s(pmax + 1, s_max));
        }""")
# generator counts
i = s.index("                let count: usize = p_s")
j = s.index(".sum();", i)
old = s[i:j + len(".sum();")]
new = """                let count: usize = p_s
                    .iter_gens(d)
                    .filter(|&(d_i, _)| self.kept(s, d_i))
                    .map(|(d_i, _)| self.module.dimension(d - d_i))
                    .sum();"""
s = s.replace(old, new)
rep("""            self.differentials[s as usize].compute_auxiliary_data_through_degree(t_max);""",
    """            let t_aux = if pmax == i32::MAX {
                t_max
            } else {
                t_max.min(pmax + s + 2 + self.module.max_degree().unwrap_or(0) + 2)
            };
            self.differentials[s as usize].compute_auxiliary_data_through_degree(t_aux);""")
rep("""        for (d_i, i) in p_s.iter_gens(degree) {""",
    """        for (d_i, i) in p_s.iter_gens(degree) {
            if !self.kept(s, d_i) {
                continue;
            }""")
rep("""        for (dk, k) in p_s.iter_gens(gen_deg) {""",
    """        for (dk, k) in p_s.iter_gens(gen_deg) {
            if !self.kept(s, dk) {
                continue;
            }""")
open(p, "w").write(s)
print("patched")
