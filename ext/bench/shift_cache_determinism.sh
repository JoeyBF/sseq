#!/bin/bash
# Does the shift reuse cache ever change the MATHEMATICS?
#
# The claim in the code is that it cannot: "a miss merely rebuilds the entry, so eviction can never
# change results". That is only true if the cache KEY is complete -- and the sibling solver cache
# documents exactly this hazard, where keying by mask LENGTH rather than CONTENTS would be wrong.
# An incomplete key under a varying hit pattern gives INTERMITTENTLY wrong output, the worst
# failure mode there is, so this is worth testing rather than trusting.
#
# Two independent things are tested:
#   1. cache setting independence : 64GB vs 4GB vs OFF must agree
#   2. run-to-run determinism     : the SAME setting twice must agree, even though bidegree
#                                   scheduling (and hence the hit pattern) differs between runs
set -e
N=${1:-170}
S=${2:-85}
W=/tmp/determinism
BIN=/wsu/home/hd/hd72/hd7264/Documents/sseq/.claude/worktrees/census-sigs/ext/target/release/examples/resolve_through_stem

rm -rf "$W"; mkdir -p "$W"; cd "$W"
ml cuda/12.4 2>/dev/null || true
export CUDA_PATH=/wsu/el7/cuda/12.4 CUDA_HOME=$CUDA_PATH
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:/usr/lib64:$LD_LIBRARY_PATH"
export NASSAU_GPU=1 EXT_NASSAU_NO_SAVE_QI=1
export NASSAU_GPU_CLEANUP_EVERY=0 NASSAU_GPU_DEVICES=3 FP_CUDA_DEVICE=3
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-64}
export RUST_LOG=warn

run () {   # $1 tag, $2 cache setting
    local tag=$1 cache=$2
    if [ "$cache" = off ]; then
        NASSAU_SHIFT_REUSE=0 sh -c 'printf "S_2\n\n'"$N"'\n'"$S"'\n" | "$0"' "$BIN" \
            > "$W/$tag.out" 2> "$W/$tag.log"
    else
        NASSAU_SHIFT_CACHE_GB=$cache sh -c 'printf "S_2\n\n'"$N"'\n'"$S"'\n" | "$0"' "$BIN" \
            > "$W/$tag.out" 2> "$W/$tag.log"
    fi
    echo "  $tag done ($(wc -l < "$W/$tag.out") lines, sha $(sha256sum < "$W/$tag.out" | cut -c1-16))"
}

echo "=== stem $N, max_s $S ==="
run cache64_a 64
run cache64_b 64      # same setting, second run -- hit pattern will differ
run cache4    4
run cacheoff  off

echo
echo "=== 1. run-to-run determinism at a FIXED setting (64GB twice) ==="
if diff -q "$W/cache64_a.out" "$W/cache64_b.out" >/dev/null; then echo "  IDENTICAL"; else
  echo "  *** DIFFERS -- the cache makes results irreproducible ***"; diff "$W/cache64_a.out" "$W/cache64_b.out" | head; fi

echo "=== 2. cache-setting independence (64GB vs 4GB vs OFF) ==="
for t in cache4 cacheoff; do
  if diff -q "$W/cache64_a.out" "$W/$t.out" >/dev/null; then echo "  64GB == $t : IDENTICAL"; else
    echo "  *** 64GB != $t -- the cache CHANGES THE MATHEMATICS ***"; diff "$W/cache64_a.out" "$W/$t.out" | head; fi
done

echo "=== hit rates (confirming the arms really differed) ==="
grep -ho "cache hits=[0-9]*" "$W"/cache64_a.log "$W"/cache4.log "$W"/cacheoff.log 2>/dev/null | tail -3 || true
