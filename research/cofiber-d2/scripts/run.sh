#!/bin/bash
# module_d2 blocks for the four two-cell complexes through stem 110, from the secprod store.
T=$CLAUDE_JOB_DIR/tmp
cd $T/conesrc/steenrod_modules
for M in C2 Ceta Cnu Csigma; do
  /usr/bin/time -f "$M: %e s %M KB" env RUST_LOG=info EXT_NASSAU_NO_SAVE_QI=1 MD_SPEC=$T/specs/$M.spec MD_MAX_N=112 MD_MAX_S=70 \
    nice -n 19 $T/target-cone/release/examples/module_d2 \
    < <(printf 'S_2\n/tmp/hd7264_secprod_store\n112\n70\n') > $T/md110/$M.csv 2> $T/md110/$M.log
  tail -1 $T/md110/$M.log
done
