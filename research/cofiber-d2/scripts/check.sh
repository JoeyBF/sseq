#!/bin/bash
# Merges cone_d2's (k,0) block with the sphere's diagonal blocks, converts to Ext-level d2 files,
# and compares with the tensor truth, module_d2 and the easy values.
# usage: check.sh <tag> <max stem> <max f>
T=$CLAUDE_JOB_DIR/tmp
cd $T
tag=$1; N=$2; F=$3
for p in C2:1 Ceta:2 Cnu:4 Csigma:8; do
  M=${p%%:*}; k=${p##*:}
  head -1 md110/n200_$M.csv > cd/blk${tag}_$M.csv
  awk -F, 'NR>1 && $5==$7' md110/n200_$M.csv >> cd/blk${tag}_$M.csv
  tail -n +2 cd/${tag}_$M.csv >> cd/blk${tag}_$M.csv
  python3 md110/to_d2.py /tmp/cof351/mods $M $k cd/blk${tag}_$M.csv cd/d2cone${tag}_$M.csv $N $F cd/hprod.csv 2>&1 | tail -1
  if [ -f cd/d2truth_$M.csv ]; then
    python3 cd/cmp.py "$M cone vs tensor truth" cd/d2cone${tag}_$M.csv cd/d2truth_$M.csv
  fi
  python3 cd/cmp.py "$M cone vs module_d2" cd/d2cone${tag}_$M.csv md110/d2n200_$M.csv
  python3 cd/cmp.py "$M cone vs easy" cd/d2cone${tag}_$M.csv cone/d2_${M}_easy_n200.csv
done
