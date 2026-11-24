#!/bin/bash
# Localizes the stem-200 d2 contradictions: one module's exact d2 file at a time, and all with COF_D2=0.
cd $CLAUDE_JOB_DIR/tmp
B=/wsu/home/hd/hd72/hd7264/Documents/sseq/.claude/worktrees/derivations-global/ext/target/release/derivations
R=/rs/rs_grp_csht/resolutions/sphere_gpu_n400/ring_n400_nsweep
for M in C2 Ceta Cnu Csigma; do
  sed "/^module $M S/!s|md110/d2n200_[A-Za-z0-9]*\.csv|-|" md110/free200.dg > md110/iso_$M.dg
  nice -n 19 $B $R 200 100 --gens-only --diagram md110/iso_$M.dg --from-e2 --pages 2 --secondary secprod/easy3_n200.out > md110/iso_$M.out 2> md110/iso_$M.log &
done
COF_D2=0 nice -n 19 $B $R 200 100 --gens-only --diagram md110/free200.dg --from-e2 --pages 2 --secondary secprod/easy3_n200.out > md110/iso_nocof.out 2> md110/iso_nocof.log &
wait
for f in C2 Ceta Cnu Csigma nocof; do echo "$f: $(grep -m1 'd2 solution space' md110/iso_$f.log)"; done
