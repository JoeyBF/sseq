#!/bin/bash
# Bisects Ceta's stem-200 d2 contradictions by truncating its d2 file at a stem.
cd $CLAUDE_JOB_DIR/tmp
B=/wsu/home/hd/hd72/hd7264/Documents/sseq/.claude/worktrees/derivations-global/ext/target/release/derivations
R=/rs/rs_grp_csht/resolutions/sphere_gpu_n400/ring_n400_nsweep
for c in "$@"; do
  awk -F, -v c=$c 'NR==1 || $1<=c' md110/d2n200_Ceta.csv > md110/bis_$c.csv
  sed "s|md110/d2n200_Ceta.csv|md110/bis_$c.csv|" md110/iso_Ceta.dg > md110/bis_$c.dg
  COF_D2=0 nice -n 19 $B $R 200 100 --gens-only --diagram md110/bis_$c.dg --from-e2 --pages 2 --secondary secprod/easy3_n200.out > md110/bis_$c.out 2> md110/bis_$c.log &
done
wait
for c in "$@"; do echo "$c: $(grep -m1 'd2 solution space' md110/bis_$c.log)"; done
