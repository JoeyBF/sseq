pub mod prelude {
    pub trait MaybeParallelIterator: Iterator {}

    pub trait MaybeIndexedParallelIterator: Iterator {}

    pub trait IntoMaybeParallelIterator: IntoIterator {
        type Iter;

        fn into_maybe_par_iter(self) -> Self::Iter;
    }

    pub trait MaybeIntoParallelRefMutIterator<'data> {
        type Iter;

        fn maybe_par_iter_mut(&'data mut self) -> Self::Iter;
    }

    pub struct MaybeIterBridge<Iter>(Iter);

    pub trait MaybeParallelBridge: Sized {
        fn maybe_par_bridge(self) -> MaybeIterBridge<Self> {
            MaybeIterBridge(self)
        }
    }

    // Implementations

    impl<I: Iterator> MaybeParallelIterator for I {}

    impl<I: Iterator> MaybeIndexedParallelIterator for I {}

    impl<I: IntoIterator> IntoMaybeParallelIterator for I {
        type Iter = Self::IntoIter;

        fn into_maybe_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    impl<'data, I: 'data + ?Sized> MaybeIntoParallelRefMutIterator<'data> for I
    where
        &'data mut I: IntoIterator,
    {
        type Iter = <&'data mut I as IntoIterator>::IntoIter;

        fn maybe_par_iter_mut(&'data mut self) -> Self::Iter {
            self.into_iter()
        }
    }

    impl<Iter: Iterator> Iterator for MaybeIterBridge<Iter> {
        type Item = Iter::Item;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }
    }

    impl<T: Iterator> MaybeParallelBridge for T {}
}

#[allow(dead_code)]
pub struct Scope<'scope>(&'scope ());

impl<'scope> Scope<'scope> {
    pub fn spawn<BODY>(&self, body: BODY)
    where
        BODY: FnOnce(&Self) + Send + 'scope,
    {
        body(self)
    }
}

pub struct MaybeThreadPool;

impl MaybeThreadPool {
    pub fn in_place_scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&Scope<'scope>) -> R,
    {
        op(&Scope(&()))
    }
}

#[derive(Debug, Default)]
pub struct MaybeThreadPoolBuilder;

#[derive(Debug)]
pub struct MaybeThreadPoolBuildError;

impl MaybeThreadPool {
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        op()
    }
}

impl MaybeThreadPoolBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn num_threads(self, _num_threads: usize) -> Self {
        self
    }

    pub fn full_blocking(self) -> Self {
        self
    }

    pub fn build(self) -> Result<MaybeThreadPool, MaybeThreadPoolBuildError> {
        Ok(MaybeThreadPool)
    }
}

pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    (oper_a(), oper_b())
}

pub fn scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R + Send,
    R: Send,
{
    op(&Scope(&()))
}

pub fn in_place_scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R,
{
    op(&Scope(&()))
}
