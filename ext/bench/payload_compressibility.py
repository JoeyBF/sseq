#!/usr/bin/env python3
"""Structural compressibility of a captured frontier batch -- SAMPLED and vectorised.

Why it exists: in a distributed design the captured batch IS the wire payload, so its size sets how
many workers one submitter can feed. Measured on a real frontier batch (1.910 GB, 2,285,427
products, 445,486,632 terms, mean 194.9):

    headers      5.7% of file
    term lists  93.3% of file
    sorted ascending 100.0%   median max-delta 7
    varint(delta) -> 4.0x on term lists ;  dedup (54.1% distinct) -> 1.85x   [product 7.4x]

    zstd -3 : 1.910 GB -> 0.271 GB = 7.04x       lz4 -1 : -> 0.908 GB = 2.10x

CONCLUSION: just zstd the payload. A hand-rolled varint+dedup encoding reaches ~7.4x in theory,
which zstd -3 already delivers for no design work. At 7x, one unit is ~0.305 GB per ~21s of
compute = 0.116 Gb/s, so a 25 GbE submitter feeds ~215 L40S-equivalents and bandwidth stops being
the binding constraint.

Run it against a batch captured with NASSAU_CAPTURE_PRODUCTS (NASSAU_CAPTURE_NTH=1..3 at the
frontier -- the default of 200 counts TOP-LEVEL calls and is unreachable there).

The first version of this looped over all ~450M terms in Python and would have taken hours. The
quantities wanted here (mean term count, sortedness, delta magnitudes, dedup rate) are all
distributional, so a sample of products answers them to far better precision than the decisions
they inform.

Format, from capture_batch in milnor_gpu.rs (little-endian):
    magic "NASPROD1" | num_rows u64 | out_cols u64 | has_col_map u64 | col_map_len u64
    | col_map[len] u32 | num_products u64
    | per product: r_degree i64, s_degree i64, r_idx u64, row u64, out_offset u64, nterms u64,
                   terms[nterms] u32
"""
import sys, struct, collections
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/rs/rs_grp_csht/hd7264/capture/frontier_batch"
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000

with open(path, "rb") as f:
    buf = f.read()
mv = memoryview(buf)
assert bytes(mv[0:8]) == b"NASPROD1"
off = 8
num_rows, out_cols, has_cm, cm_len = struct.unpack_from("<QQQQ", mv, off)
off += 32 + cm_len * 4
(num_products,) = struct.unpack_from("<Q", mv, off)
off += 8
print(f"file {len(buf)/1e9:.3f} GB | rows={num_rows} out_cols={out_cols} "
      f"col_map={cm_len} products={num_products:,}")

# Pass 1: headers only, for ALL products (cheap: one 48-byte unpack each, no term touching).
unpack = struct.Struct("<qqQQQQ").unpack_from
starts = np.empty(num_products, dtype=np.int64)
nterms = np.empty(num_products, dtype=np.int64)
rows = np.empty(num_products, dtype=np.int64)
roff = np.empty(num_products, dtype=np.int64)
rkey = np.empty(num_products, dtype=np.int64)
p = off
for i in range(num_products):
    rd, sd, ridx, row, outo, nt = unpack(mv, p)
    p += 48
    starts[i] = p
    nterms[i] = nt
    rows[i] = row
    roff[i] = outo
    rkey[i] = (rd << 40) ^ ridx
    p += nt * 4
total_terms = int(nterms.sum())
print(f"terms {total_terms:,} (mean {total_terms/num_products:.1f}, "
      f"max {int(nterms.max())}, min {int(nterms.min())})")
print(f"distinct rows {len(np.unique(rows)):,} | distinct out_offset {len(np.unique(roff)):,} "
      f"| distinct (r_deg,r_idx) {len(np.unique(rkey)):,}")

hdr_bytes = num_products * 48
term_bytes = total_terms * 4
print(f"\n--- layout ---")
print(f"headers    {hdr_bytes/1e9:7.3f} GB  ({100*hdr_bytes/len(buf):.1f}%)")
print(f"term lists {term_bytes/1e9:7.3f} GB  ({100*term_bytes/len(buf):.1f}%)")

# Pass 2: sampled term analysis.
rng = np.random.default_rng(0)
idx = rng.choice(num_products, size=min(SAMPLE, num_products), replace=False)
n_sorted = 0
varint_bytes = 0
raw_bytes = 0
hashes = collections.Counter()
maxdelta = []
for i in idx:
    nt = int(nterms[i]); s = int(starts[i])
    if nt == 0:
        continue
    ts = np.frombuffer(buf, dtype=np.uint32, count=nt, offset=s).astype(np.int64)
    raw_bytes += nt * 4
    hashes[hash(ts.tobytes())] += 1
    d = np.diff(ts)
    asc = bool((d > 0).all())
    n_sorted += asc
    deltas = np.concatenate(([ts[0]], d)) if asc else ts
    maxdelta.append(int(deltas.max()))
    # varint: bytes = ceil(bits/7), min 1
    bits = np.maximum(1, np.ceil(np.log2(np.maximum(deltas, 1) + 1))).astype(np.int64)
    varint_bytes += int(np.ceil(bits / 7).clip(1).sum())

ns = len(idx)
uniq = len(hashes)
print(f"\n--- sampled {ns:,} products ---")
print(f"sorted ascending      {n_sorted:,}/{ns:,} ({100*n_sorted/ns:.1f}%)")
print(f"distinct term arrays  {uniq:,}/{ns:,} ({100*uniq/ns:.1f}%)  "
      f"-> dedup potential {ns/max(uniq,1):.2f}x")
print(f"median max-delta      {int(np.median(maxdelta))}")
print(f"varint(delta) bytes   {varint_bytes/max(raw_bytes,1):.3f} of raw "
      f"-> {raw_bytes/max(varint_bytes,1):.2f}x on term lists")

est_terms = term_bytes * varint_bytes / max(raw_bytes, 1)
est_hdr = num_products * 20          # r_deg u16+s_deg u16, r_idx u32, row u32, out_offset u32, nterms u32
est = est_hdr + est_terms
print(f"\n--- projected wire size ---")
print(f"tight headers + varint deltas = {est/1e9:.3f} GB  ({len(buf)/max(est,1):.2f}x smaller)")
