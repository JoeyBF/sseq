#!/bin/bash
# cone_d2 in the delta' mode for one complex through stem $3 (s <= $4): run1.sh <j> <name> <N> <S>
T=$CLAUDE_JOB_DIR/tmp
cd $T/conesrc/steenrod_modules
j=$1; M=$2; N=$3; S=$4
printf 'S_2\n/tmp/hd7264_recheck_store\n%s\n%s\n' $N $((S + 4)) | /usr/bin/time -v \
  env RUST_LOG=info EXT_NASSAU_NO_SAVE_QI=1 CD_J=$j CD_NAME=$M CD_MAX_N=$N CD_MAX_S=$S CD_MAP=theta CD_VERIFY=1 \
  nice -n 19 $T/target-cone/release/examples/cone_d2 > $T/cd/th${N}_$M.csv 2> $T/cd/th${N}_$M.log
grep -hE "first step|nonzero rows|panicked|Elapsed|Maximum res" $T/cd/th${N}_$M.log
