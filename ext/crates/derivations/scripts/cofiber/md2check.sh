#!/bin/bash
# Class-level comparison of module_d2's prediction with cell_truth ground truth.
# usage: md2check.sh <truth csv> <feat csv (only for cocfit's bookkeeping)> <module> <max n> <max s>
cd "$(dirname "$0")"
T=$(realpath ..)
cd "$T/conesrc"
printf 'S_2\n/tmp/hd7264_secprod_store\n%d\n%d\n' $(( $4 + 24 )) $(( $5 + 8 )) \
    | MD_SPEC="$T/specs/$3.spec" MD_MAX_N=$(( $4 + 2 )) MD_MAX_S=$(( $5 + 2 )) EXT_NASSAU_NO_SAVE_QI=1 \
      "$T/target-cone/release/examples/module_d2" 2>/dev/null > "$T/truth/pred_$3.csv"
cd "$T/truth"
specs=()
for c in 0 1 2 3 4 5 6 7 8 9; do for b in 0 1 2 3 4 5 6 7 8 9; do
    [ "$c" -gt "$b" ] && specs+=("$c>$b:0")
done; done
echo -n "$3: "
ALT="pred_$3.csv" DCHI=chi CHI=chi10.txt python3 cocfit.py "$1" "$2" "$3" "$4" "$5" "${specs[@]}" \
    | head -1 | grep -o "nonzero.*"
