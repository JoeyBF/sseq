#!/bin/bash
# Does walking the signatures in SCHEDULE-LEVEL order change the mathematics?
#
# This is the premise of the whole level-parallel lift, tested while everything is still SERIAL so
# there is no concurrency to confuse a failure with. The DAG says every dependency edge strictly
# increases the level, so a level-ordered walk still visits each signature after everything it
# depends on. If that is true on live data the Ext chart is unchanged; if it is false, this is
# where it shows up -- cheaply, and before any parallel code exists.
#
# Reuse cache OFF in both arms (measurement policy) so the only difference is the walk order.
set -e
N=${1:-150}
S=${2:-75}
W=/tmp/levelorder
BIN=/wsu/home/hd/hd72/hd7264/Documents/sseq/.claude/worktrees/census-sigs/ext/target/release/examples/resolve_through_stem

rm -rf "$W"; mkdir -p "$W"; cd "$W"
ml cuda/12.4 2>/dev/null || true
export CUDA_PATH=/wsu/el7/cuda/12.4 CUDA_HOME=$CUDA_PATH
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:/usr/lib64:$LD_LIBRARY_PATH"
export NASSAU_GPU=1 EXT_NASSAU_NO_SAVE_QI=1 NASSAU_GPU_CLEANUP_EVERY=0
export NASSAU_GPU_DEVICES=3 FP_CUDA_DEVICE=3 NASSAU_GPU_RESIDENT_MAX_DEGREE=200
export RAYON_NUM_THREADS=96
export NASSAU_SHIFT_REUSE=0
export RUST_LOG=warn

run () {
    local tag=$1 lvl=$2
    local t0 t1
    t0=$(date +%s.%N)
    ( if [ "$lvl" = 1 ]; then export NASSAU_SIG_LEVEL_ORDER=1; fi
      printf 'S_2\n\n%s\n%s\n' "$N" "$S" | "$BIN" > "$W/$tag.out" 2> "$W/$tag.log" )
    t1=$(date +%s.%N)
    echo "  $tag: $(echo "$t1 - $t0" | bc)s  sha $(sha256sum < "$W/$tag.out" | cut -c1-16)"
}

echo "=== stem $N max_s $S, reuse OFF, serial in both arms ==="
run odometer 0
run levelord 1

echo "=== chart diff ==="
if diff -q "$W/odometer.out" "$W/levelord.out" >/dev/null; then
    echo "  IDENTICAL ($(wc -l < "$W/odometer.out") lines) -- the level schedule is VALID on live data"
else
    echo "  *** DIFFERS -- a dependency edge does NOT increase the level; the schedule is unsound ***"
    diff "$W/odometer.out" "$W/levelord.out" | head -20
fi
grep -c panicked "$W"/*.log 2>/dev/null || true
