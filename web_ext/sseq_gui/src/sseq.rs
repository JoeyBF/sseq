use fp::prime::ValidPrime;
use sseq::{
    Adams, SseqProfile,
    managed::{ManagedSseq, RefreshResult},
};

use crate::{Sender, actions::*};

/// Thin wrapper around [`ManagedSseq`] that converts refresh results to messages.
pub struct SseqWrapper<P: SseqProfile = Adams> {
    pub managed: ManagedSseq<P>,
    name: SseqChoice,

    /// If this is a positive number, then the spectral sequence will not re-compute classes and
    /// edges. See [`BlockRefresh`] for details.
    pub block_refresh: u32,
    sender: Option<Sender>,
}

impl<P: SseqProfile> SseqWrapper<P> {
    pub fn new(p: ValidPrime, name: SseqChoice, sender: Option<Sender>) -> Self {
        Self {
            managed: ManagedSseq::new(p),
            name,
            block_refresh: 0,
            sender,
        }
    }

    pub fn refresh(&mut self) {
        if self.block_refresh > 0 {
            return;
        }
        let result = self.managed.refresh();
        self.send_refresh_result(result);
    }

    fn send_refresh_result(&self, result: RefreshResult) {
        for diff_data in result.differentials {
            self.send(Message {
                recipients: vec![],
                sseq: self.name,
                action: Action::from(SetDifferential(diff_data)),
            });
        }
        for class_data in result.classes {
            self.send(Message {
                recipients: vec![],
                sseq: self.name,
                action: Action::from(SetClass(class_data)),
            });
        }
        for edge_data in result.edges {
            self.send(Message {
                recipients: vec![],
                sseq: self.name,
                action: Action::from(SetStructline(edge_data)),
            });
        }
    }

    fn send(&self, msg: Message) {
        if let Some(sender) = &self.sender {
            sender.send(msg).unwrap();
        }
    }
}
