use std::sync::Arc;

use algebra::module::Module;
use ext::{
    chain_complex::ChainComplex,
    resolution_with_products::{ResolutionEvent, ResolutionWithProducts},
};
use fp::prime::ValidPrime;
use serde_json::Value;
use sseq::coordinates::Bidegree;

use crate::actions::{Action, Message, SseqChoice};

pub fn resolution_event_to_message(event: ResolutionEvent, sseq: SseqChoice) -> Message {
    match event {
        ResolutionEvent::NewClass { b, dimension } => Message {
            recipients: vec![],
            sseq,
            action: Action::from(crate::actions::AddClass { b, num: dimension }),
        },
        ResolutionEvent::NewProduct {
            name,
            source_b,
            mult_b,
            left,
            matrix,
        } => Message {
            recipients: vec![],
            sseq,
            action: Action::from(crate::actions::AddProduct {
                name,
                source_b,
                mult_b,
                left,
                matrix,
            }),
        },
    }
}

/// Thin wrapper around [`ResolutionWithProducts`] that converts events to messages.
pub struct Resolution<CC: ChainComplex> {
    pub inner: ResolutionWithProducts<CC>,
    sender: crate::Sender,
    sseq: crate::actions::SseqChoice,
}

impl Resolution<ext::CCC> {
    pub fn new_from_json(
        json: Value,
        algebra_name: &str,
        sseq: crate::actions::SseqChoice,
        sender: crate::Sender,
    ) -> Option<Self> {
        let inner = ResolutionWithProducts::new_from_json(json, algebra_name)?;
        Some(Self {
            inner,
            sender,
            sseq,
        })
    }
}

impl<CC: ChainComplex> Resolution<CC> {
    pub fn compute_through_stem(&self, b: Bidegree) {
        let sender = &self.sender;
        let sseq = self.sseq;
        self.inner.compute_through_stem(b, |event| {
            sender
                .send(resolution_event_to_message(event, sseq))
                .unwrap();
        });
    }

    pub fn complex(&self) -> Arc<CC> {
        self.inner.complex()
    }
}

impl<CC: ChainComplex> Resolution<CC> {
    pub fn algebra(&self) -> Arc<<CC::Module as Module>::Algebra> {
        self.complex().algebra()
    }

    pub fn prime(&self) -> ValidPrime {
        self.inner.prime()
    }

    pub fn min_degree(&self) -> i32 {
        self.inner.min_degree()
    }

    pub fn set_unit_resolution_self(&mut self) {
        self.inner.set_unit_resolution_self();
    }

    pub fn set_unit_resolution(&mut self, unit_res: Self) {
        self.inner.set_unit_resolution(unit_res.inner);
    }
}
