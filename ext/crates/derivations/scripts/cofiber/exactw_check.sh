#!/bin/bash
# exact_w solve + module_d2 prediction vs cell_truth; with NULL_OUT, fit homogeneous directions.
# usage: exactw_check.sh <truth csv> <feat csv> <module> <max n> <max s>
cd "$(dirname "$0")"
T=$(realpath ..)
B=$T/target-truth/release/examples/exact_w
cd "$T/truthsrc/ext"
mapfile -t spec < <(EXACT="$T/truth/oexact_$3.csv" NORM=${NORM_MODE:-nopath} SPEC_OUT="$T/specs/${3}_ex2.spec" \
    NULL_OUT="$T/truth/null_ex2_$3.txt" $B "$3" 2>"$T/truth/exw_$3.log")
cd "$T/truth"
echo -n "$(cat exw_$3.log) | truth: "
./md2check2.sh "$1" "$2" "$3" "$4" "$5" "$T/specs/${3}_ex2.spec" | sed "s/^$3: //"
echo -n "    theta fit: "
COMBOS=null_ex2_$3.txt DCHI=chi SHEAR=chi CHI=chi10.txt python3 cocfit.py "$1" "$2" "$3" "$4" "$5" "${spec[@]}" \
    | grep -v "undetermined" | tr '\n' ' ' | grep -o "rank.*"
echo
