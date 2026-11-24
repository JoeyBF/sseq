#!/bin/bash
# cone_d2 for C2, Ceta, Cnu, Csigma (j = 0..3) through stem $1 (s <= $2), on the recheck store.
T=$CLAUDE_JOB_DIR/tmp
cd $T/conesrc/steenrod_modules
N=$1; S=$2
for p in 0:C2 1:Ceta 2:Cnu 3:Csigma; do
  j=${p%%:*}; M=${p##*:}
  printf 'S_2\n/tmp/hd7264_recheck_store\n%s\n%s\n' $((N + 10)) $((S + 4)) | /usr/bin/time -f "$M: %e s %M KB" \
    env RUST_LOG=info EXT_NASSAU_NO_SAVE_QI=1 CD_J=$j CD_NAME=$M CD_MAX_N=$N CD_MAX_S=$S CD_MAP=theta CD_VERIFY=1 \
    nice -n 19 $T/target-cone/release/examples/cone_d2 > $T/cd/th${N}_$M.csv 2> $T/cd/th${N}_$M.log &
done
wait
for M in C2 Ceta Cnu Csigma; do grep -E "nonzero rows|^C|^[A-Z][a-z]*: [0-9.]+ s" $T/cd/th${N}_$M.log | tail -2; done
