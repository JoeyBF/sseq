pub mod prelude {
    pub use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::prelude::*;

    pub trait MaybeParallelIterator: ParallelIterator {}

    pub trait MaybeIndexedParallelIterator: IndexedParallelIterator {}

    pub trait IntoMaybeParallelIterator: IntoParallelIterator + Sized {
        fn into_maybe_par_iter(self) -> Self::Iter {
            self.into_par_iter()
        }
    }

    pub trait IntoMaybeParallelRefMutIterator<'data>: IntoParallelRefMutIterator<'data> {
        fn maybe_par_iter_mut(&'data mut self) -> Self::Iter {
            self.par_iter_mut()
        }
    }

    pub type MaybeIterBridge<I> = rayon::iter::IterBridge<I>;

    pub trait MaybeParallelBridge: ParallelBridge {
        fn maybe_par_bridge(self) -> MaybeIterBridge<Self> {
            self.par_bridge()
        }
    }

    pub trait MaybeParallelSliceMut<T: Send>: ParallelSliceMut<T> {
        fn maybe_par_chunks_mut<'data>(
            &'data mut self,
            chunk_size: usize,
        ) -> impl MaybeIndexedParallelIterator<Item = &'data mut [T]>
        where
            T: 'data,
        {
            self.par_chunks_mut(chunk_size)
        }
    }

    // Implementations

    impl<I: ParallelIterator> MaybeParallelIterator for I {}

    impl<I: IndexedParallelIterator> MaybeIndexedParallelIterator for I {}

    impl<I: IntoParallelIterator> IntoMaybeParallelIterator for I {}

    impl<'data, I: 'data + ?Sized> IntoMaybeParallelRefMutIterator<'data> for I where
        &'data mut I: IntoMaybeParallelIterator
    {
    }

    impl<I: ParallelBridge> MaybeParallelBridge for I {}

    impl<T: Send> MaybeParallelSliceMut<T> for [T] {}
}

pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    rayon::join(oper_a, oper_b)
}

pub type Scope<'scope> = rayon::Scope<'scope>;

pub fn scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R + Send,
    R: Send,
{
    rayon::scope(op)
}

pub fn in_place_scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R,
{
    rayon::in_place_scope(op)
}

pub fn empty<T: Send>() -> rayon::iter::Empty<T> {
    rayon::iter::empty()
}

/// A private thread pool, so a caller can run work on threads that are NOT the global pool's.
///
/// The motivating case: a task that blocks (e.g. waiting on a GPU) must not be able to steal an
/// unrelated large job while it waits, which is how priority inversion happens. A private pool
/// bounds what its workers can pick up to the tasks submitted to it.
pub struct MaybeThreadPool(rayon::ThreadPool);

impl MaybeThreadPool {
    /// Build a pool with `num_threads` workers named `<name_prefix><i>`.
    pub fn new(num_threads: usize, name_prefix: &'static str) -> Self {
        Self(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .thread_name(move |i| format!("{name_prefix}{i}"))
                .build()
                .expect("failed to build a MaybeThreadPool"),
        )
    }

    /// Run `f` inside the pool; parallel iterators created within it use only this pool's workers.
    pub fn install<R: Send, F: FnOnce() -> R + Send>(&self, f: F) -> R {
        self.0.install(f)
    }
}
