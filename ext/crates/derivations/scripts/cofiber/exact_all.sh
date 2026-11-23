#!/bin/bash
# Exact obstruction (bar_o) and the exact twisted equation (twisted_w EXACT=) for each module.
cd "$(dirname "$0")/../truthsrc/ext"
T=$(realpath ../..)
for m in "$@"; do
    top=$(python3 -c "import json;print(max(json.load(open('steenrod_modules/$m.json'))['gens'].values()))")
    timeout 3000 "$T/target-truth/release/examples/bar_o" "$m" $((top + 2)) > "$T/truth/oexact_$m.csv" 2>/dev/null
    r=$(EXACT="$T/truth/oexact_$m.csv" NORM=nopath SPEC_OUT="$T/specs/${m}_exact.spec" \
        timeout 3000 "$T/target-truth/release/examples/twisted_w" "$m" 2>&1 | grep unknowns)
    echo "$m: $r"
done
