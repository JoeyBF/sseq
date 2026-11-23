#!/bin/bash
# Fit the homogeneous (theta) directions of the twisted cobar equation against cell_truth data.
# usage: thetafit.sh <cell csv> <feat csv> <module> <max n> <max s>
cd "$(dirname "$0")"
B=../target-truth/release/examples/twisted_w
mapfile -t spec < <(NULL_OUT=null_$3.txt $B "$3" 2>/dev/null)
COMBOS=null_$3.txt DCHI=chi SHEAR=chi CHI=chi10.txt python3 cocfit.py "$1" "$2" "$3" "$4" "$5" "${spec[@]}" \
    | grep -v "undetermined per block"
