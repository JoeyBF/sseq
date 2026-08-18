//! Crash-safe and snapshot-safe writes for the on-disk save store.
//!
//! # Why
//!
//! [`zarrs_filesystem::FilesystemStore`] writes values **in place**: its `set_impl` opens the
//! target path with `.truncate(true)` and writes the bytes directly. Two consequences, both of
//! which bite a multi-week resolution:
//!
//! 1. **A crash mid-write leaves a truncated value.** The shard tier packs an `8x8` block of
//!    `(n, s)` into one key with a CRC32C over it, so a torn write does not lose one bidegree, it
//!    loses up to 64 — and [`crate::save::SaveStore::read`] propagates the CRC failure as an `Err`
//!    rather than treating the data as missing, so the next run fails to resume instead of
//!    recomputing.
//! 2. **A concurrent copy can observe a torn value**, which makes periodic snapshotting of a live
//!    save directory unsafe for exactly the same reason.
//!
//! # How
//!
//! Every write path in `zarrs` funnels through [`WritableStorageTraits::set`] — `set_partial_many`
//! is a read-modify-write that ends in `store.set(key, ..)` — so making `set` atomic makes the
//! whole store atomic. This adapter writes to a sibling temporary key and then uses the inner
//! store's [`AtomicRenameStorageTraits::rename`], which is `std::fs::rename` and therefore replaces
//! the destination atomically. A reader (or a snapshotting `rsync`) sees either the old value or
//! the new one, never a mixture.
//!
//! Temporary keys are suffixed with [`TMP_SUFFIX`] and are hidden from the listing traits, so a
//! crash that leaves one behind cannot be mistaken for a zarr key. [`AtomicWriteStore::new`] also
//! sweeps any stragglers left by a previous run.
//!
//! This does not remove the need to compute on node-local disk — that is a throughput choice, not a
//! safety one — but it does make the save directory safe to copy while it is being written.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use zarrs::storage::{
    AtomicRenameStorageTraits, Bytes, ListableStorageTraits, MaybeBytesIterator,
    OffsetBytesIterator, ReadableStorageTraits, ReadableWritableListableStorageTraits, StorageError,
    StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
    byte_range::ByteRangeIterator, store_set_partial_many,
};

/// Marks a key as an in-flight temporary. Chosen to be illegal as a zarr node name in practice and
/// easy to grep for on disk; keys carrying it are filtered out of every listing.
const TMP_SUFFIX: &str = ".__atomic_tmp";

/// Wraps a store so that [`WritableStorageTraits::set`] is atomic (write-to-temp then rename).
///
/// See the module documentation for why this is necessary.
pub struct AtomicWriteStore<S> {
    inner: Arc<S>,
    /// Disambiguates concurrent writes to the same key from different threads. Paired with the pid
    /// so that two processes sharing a save directory cannot collide on a temp name either.
    counter: AtomicU64,
    pid: u32,
}

impl<S> AtomicWriteStore<S>
where
    S: ReadableWritableListableStorageTraits + AtomicRenameStorageTraits + 'static,
{
    /// Wrap `inner`, sweeping any temporary keys left behind by a previous run.
    ///
    /// A leftover temp is by definition a write that never completed, so its contents are garbage;
    /// erasing it is always safe.
    pub fn new(inner: Arc<S>) -> Result<Self, StorageError> {
        for key in inner.list()? {
            if key.as_str().contains(TMP_SUFFIX) {
                inner.erase(&key)?;
            }
        }
        Ok(Self {
            inner,
            counter: AtomicU64::new(0),
            pid: std::process::id(),
        })
    }

    /// A unique sibling key for staging a write to `key`.
    ///
    /// The temp shares `key`'s parent directory, which keeps the rename within one filesystem —
    /// `std::fs::rename` is only atomic in that case.
    fn tmp_key(&self, key: &StoreKey) -> Result<StoreKey, StorageError> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        StoreKey::new(format!("{}{TMP_SUFFIX}.{}.{n}", key.as_str(), self.pid))
            .map_err(|e| StorageError::Other(e.to_string()))
    }
}

/// Drops any temporary keys from a listing so they are never mistaken for zarr nodes.
fn strip_tmp(keys: StoreKeys) -> StoreKeys {
    keys.into_iter()
        .filter(|k| !k.as_str().contains(TMP_SUFFIX))
        .collect()
}

impl<S> ReadableStorageTraits for AtomicWriteStore<S>
where
    S: ReadableWritableListableStorageTraits + AtomicRenameStorageTraits + 'static,
{
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl<S> ListableStorageTraits for AtomicWriteStore<S>
where
    S: ReadableWritableListableStorageTraits + AtomicRenameStorageTraits + 'static,
{
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Ok(strip_tmp(self.inner.list()?))
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        Ok(strip_tmp(self.inner.list_prefix(prefix)?))
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let kp = self.inner.list_dir(prefix)?;
        let (keys, prefixes) = (kp.keys().to_vec(), kp.prefixes().to_vec());
        Ok(StoreKeysPrefixes::new(strip_tmp(keys), prefixes))
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix)
    }
}

impl<S> WritableStorageTraits for AtomicWriteStore<S>
where
    S: ReadableWritableListableStorageTraits + AtomicRenameStorageTraits + 'static,
{
    /// The whole point of this adapter: stage into a sibling temp, then rename over the target.
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let tmp = self.tmp_key(key)?;
        self.inner.set(&tmp, value)?;
        match self.inner.rename(&tmp, key) {
            Ok(()) => Ok(()),
            Err(e) => {
                // Do not leave the staged value behind to be swept later — it would occupy space
                // for the rest of the run, and on a full filesystem that is how one failed write
                // becomes many.
                let _ = self.inner.erase(&tmp);
                Err(e)
            }
        }
    }

    /// Read-modify-write, ending in our atomic [`Self::set`]. This is how shard updates (an inner
    /// chunk written into an existing shard) become atomic too.
    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        store_set_partial_many(self, key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.inner.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.inner.erase_prefix(prefix)
    }

    /// `false`: partial writes must go through the read-modify-write path above, because a true
    /// partial write cannot be made atomic by rename.
    fn supports_set_partial(&self) -> bool {
        false
    }
}

impl<S> AtomicRenameStorageTraits for AtomicWriteStore<S>
where
    S: ReadableWritableListableStorageTraits + AtomicRenameStorageTraits + 'static,
{
    fn rename(&self, source: &StoreKey, destination: &StoreKey) -> Result<(), StorageError> {
        self.inner.rename(source, destination)
    }
}
