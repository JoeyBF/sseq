//! Regression test: a read concurrent with a write to the *same shard* must not
//! observe a torn shard. The sharding codec read-modify-writes the whole shard
//! file, so before `read()` took the per-shard lock, a read racing a write decoded
//! garbage ("Expected header with 1 elements, got N"). This surfaced when growing a
//! resolution box on a persisted store (parallel `ResolutionHomomorphism::extend_all`
//! reads and writes many bidegrees of one shard).

use std::sync::atomic::{AtomicUsize, Ordering};

use ext::save::{SaveKind, ZarrSaveStore};
use sseq::coordinates::Bidegree;

#[test]
fn concurrent_read_write_same_shard_is_not_torn() {
    let dir = std::env::temp_dir().join(format!("zarr-race-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    // Populate part of one 8×8 shard (n=0..4 at s=0) and persist it to disk.
    {
        let store = ZarrSaveStore::create(&dir).unwrap();
        for n in 0..4 {
            store
                .write(SaveKind::ChainMap, Bidegree::n_s(n, 0), &vec![n as u8 + 1; 32])
                .unwrap();
        }
    }

    // Reopen (so the shard is read back from disk), then hammer the same shard with
    // concurrent reads and read-modify-write writes.
    let store = ZarrSaveStore::create(&dir).unwrap();
    let store = &store;
    let read_errors = AtomicUsize::new(0);
    let read_errors = &read_errors;
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(move || {
                for round in 0..3000 {
                    // present (0..4), being-written (4..8), and absent (8..12) cells.
                    let n = round % 12;
                    if store.read(SaveKind::ChainMap, Bidegree::n_s(n, 0)).is_err() {
                        read_errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
        for w in 0..4 {
            scope.spawn(move || {
                for round in 0..800 {
                    let _ = store.write(
                        SaveKind::ChainMap,
                        Bidegree::n_s(4 + w, 0),
                        &vec![(round % 251) as u8; 32 + (round % 17)],
                    );
                }
            });
        }
    });

    assert_eq!(
        read_errors.load(Ordering::Relaxed),
        0,
        "reads raced with same-shard writes and observed a torn shard"
    );
    // Sanity: the four originally-written cells still read back correctly.
    for n in 0..4 {
        let got = store.read(SaveKind::ChainMap, Bidegree::n_s(n, 0)).unwrap();
        assert!(got.is_some(), "present cell (n={n}) went missing");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
