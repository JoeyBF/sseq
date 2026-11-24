#!/bin/bash
# module_d2 blocks for the four two-cell complexes through stem 110, from the secprod store.
T=$CLAUDE_JOB_DIR/tmp
cd $T/conesrc/steenrod_modules
for M in C2 Ceta Cnu Csigma; do
  /usr/bin/time -f "$M: %e s %M KB" env RUST_LOG=info EXT_NASSAU_NO_SAVE_QI=1 MD_SPEC=$T/specs/$M.spec MD_MAX_N=200 MD_MAX_S=100 \
    nice -n 19 $T/target-cone/release/examples/module_d2 \
    < <(printf 'S_2\n/tmp/hd7264_secprod_store\n200\n100\n') > $T/md110/n200_$M.csv 2> $T/md110/n200_$M.log
  tail -1 $T/md110/n200_$M.log
done
