//! # cuda-confidence-cascade
//!
//! Bayesian confidence propagation network for the Lucineer fleet.
//! Confidence flows through agents like voltage through a circuit.
//!
//! ```rust
//! use cuda_confidence_cascade::{Cascade, CascadeNode, ConfidenceUpdate};
//! use cuda_equipment::Confidence;
//!
//! let mut cascade = Cascade::new(0.85); // converge threshold
//! cascade.add_node("sensor_front", 1.0);
//! cascade.add_node("sensor_back", 1.0);
//! cascade.add_node("fusion", 0.5);
//! cascade.add_edge("sensor_front", "fusion", 0.7);
//! cascade.add_edge("sensor_back", "fusion", 0.7);
//!
//! cascade.update("sensor_front", Confidence::new(0.9));
//! cascade.update("sensor_back", Confidence::new(0.3));
//! cascade.propagate();
//! ```

pub use cuda_equipment::{Confidence, VesselId};

use std::collections::HashMap;

/// A node in the confidence cascade.
#[derive(Debug, Clone)]
pub struct CascadeNode {
    pub name: String,
    pub vessel_id: Option<VesselId>,
    pub confidence: Confidence,
    pub base_confidence: f64, // initial confidence before updates
    pub weight: f64,          // how much this node influences downstream
    pub updates: Vec<ConfidenceUpdate>,
    pub decay_rate: f64,
}

impl CascadeNode {
    pub fn new(name: &str, base_confidence: f64) -> Self {
        Self { name: name.to_string(), vessel_id: None,
            confidence: Confidence::new(base_confidence), base_confidence,
            weight: 1.0, updates: vec![], decay_rate: 0.98 }
    }

    pub fn with_vessel(mut self, id: u64) -> Self {
        self.vessel_id = Some(VesselId(id)); self
    }

    /// Apply a new observation, weighted by source reliability.
    pub fn receive_update(&mut self, source_name: &str, confidence: Confidence, reliability: f64) {
        let adjusted = Confidence::new(confidence.value() * reliability);
        self.updates.push(ConfidenceUpdate {
            source: source_name.to_string(), confidence: adjusted,
            previous: self.confidence, timestamp: now_ms(),
        });
        self.confidence = self.confidence.combine(adjusted);
    }

    /// Decay confidence toward base over time.
    pub fn decay(&mut self) {
        let decayed = Confidence::new(
            self.base_confidence + (self.confidence.value() - self.base_confidence) * self.decay_rate
        );
        self.confidence = decayed;
    }
}

/// An edge between cascade nodes with transmission weight.
#[derive(Debug, Clone)]
pub struct CascadeEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub bidirectional: bool,
}

/// A confidence update event for provenance.
#[derive(Debug, Clone)]
pub struct ConfidenceUpdate {
    pub source: String,
    pub confidence: Confidence,
    pub previous: Confidence,
    pub timestamp: u64,
}

/// Confidence cascade — propagates Bayesian confidence through a network.
pub struct Cascade {
    nodes: HashMap<String, CascadeNode>,
    edges: Vec<CascadeEdge>,
    converge_threshold: f64,
    max_rounds: u32,
    round: u32,
}

impl Cascade {
    pub fn new(converge_threshold: f64) -> Self {
        Self { nodes: HashMap::new(), edges: vec![],
            converge_threshold: converge_threshold.clamp(0.0, 1.0),
            max_rounds: 100, round: 0 }
    }

    pub fn add_node(&mut self, name: &str, base_confidence: f64) {
        self.nodes.insert(name.to_string(), CascadeNode::new(name, base_confidence));
    }

    pub fn add_edge(&mut self, from: &str, to: &str, weight: f64) {
        self.edges.push(CascadeEdge { from: from.to_string(), to: to.to_string(),
            weight: weight.clamp(0.0, 1.0), bidirectional: false });
    }

    pub fn add_bidirectional_edge(&mut self, a: &str, b: &str, weight: f64) {
        self.edges.push(CascadeEdge { from: a.to_string(), to: b.to_string(),
            weight: weight.clamp(0.0, 1.0), bidirectional: true });
    }

    /// Push an observation into a node.
    pub fn update(&mut self, node_name: &str, confidence: Confidence) {
        if let Some(node) = self.nodes.get_mut(node_name) {
            node.receive_update("external", confidence, 1.0);
        }
    }

    /// Push a weighted observation from one node to another.
    pub fn update_from(&mut self, source: &str, target: &str, confidence: Confidence, reliability: f64) {
        if let Some(node) = self.nodes.get_mut(target) {
            node.receive_update(source, confidence, reliability);
        }
    }

    /// Propagate confidence through all edges for one round.
    pub fn propagate(&mut self) -> PropagationResult {
        self.round += 1;
        let mut propagated = 0usize;
        let mut total_delta = 0.0f64;

        // Collect confidence changes (avoid borrow issues)
        let changes: Vec<(String, Confidence, f64)> = self.edges.iter().filter_map(|edge| {
            let from_node = self.nodes.get(&edge.from)?;
            let weight = edge.weight;
            Some((edge.to.clone(), from_node.confidence, weight))
        }).collect();

        // Apply changes
        for (target_name, from_conf, weight) in changes {
            if let Some(target_node) = self.nodes.get_mut(&target_name) {
                let prev = target_node.confidence;
                let incoming = Confidence::new(from_conf.value() * weight);
                target_node.confidence = target_node.confidence.combine(incoming);
                total_delta += (target_node.confidence.value() - prev.value()).abs();
                propagated += 1;
            }
        }

        // Apply bidirectional edges
        let bidi_changes: Vec<(String, Confidence, f64)> = self.edges.iter()
            .filter(|e| e.bidirectional).filter_map(|edge| {
            let to_node = self.nodes.get(&edge.to)?;
            Some((edge.from.clone(), to_node.confidence, edge.weight))
        }).collect();

        for (target_name, to_conf, weight) in bidi_changes {
            if let Some(target_node) = self.nodes.get_mut(&target_name) {
                let prev = target_node.confidence;
                let incoming = Confidence::new(to_conf.value() * weight);
                target_node.confidence = target_node.confidence.combine(incoming);
                total_delta += (target_node.confidence.value() - prev.value()).abs();
                propagated += 1;
            }
        }

        // Decay all nodes
        for node in self.nodes.values_mut() {
            node.decay();
        }

        let converged = total_delta < self.converge_threshold * 0.01
            || self.round >= self.max_rounds;

        PropagationResult { round: self.round, propagated, total_delta, converged }
    }

    /// Propagate until convergence or max rounds.
    pub fn propagate_until_converged(&mut self) -> PropagationResult {
        let mut last = PropagationResult { round: 0, propagated: 0, total_delta: 1.0, converged: false };
        loop {
            last = self.propagate();
            if last.converged { break; }
        }
        last
    }

    pub fn confidence(&self, node: &str) -> Option<Confidence> {
        self.nodes.get(node).map(|n| n.confidence)
    }

    pub fn node(&self, node: &str) -> Option<&CascadeNode> { self.nodes.get(node) }
    pub fn node_names(&self) -> Vec<String> { self.nodes.keys().cloned().collect() }
    pub fn round(&self) -> u32 { self.round }
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Summary of all node confidences.
    pub fn snapshot(&self) -> Vec<(&str, f64)> {
        self.nodes.iter().map(|(name, node)| (name.as_str(), node.confidence.value())).collect()
    }

    /// Find the least confident node (weakest link).
    pub fn weakest_node(&self) -> Option<(&str, f64)> {
        self.nodes.iter()
            .map(|(name, node)| (name.as_str(), node.confidence.value()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Find the most confident node.
    pub fn strongest_node(&self) -> Option<(&str, f64)> {
        self.nodes.iter()
            .map(|(name, node)| (name.as_str(), node.confidence.value()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct PropagationResult {
    pub round: u32,
    pub propagated: usize,
    pub total_delta: f64,
    pub converged: bool,
}

/// Confidence gate — only passes values above a threshold.
pub struct ConfidenceGate {
    pub threshold: f64,
    pub strict: bool, // if true, returns ZERO instead of discounted value
}

impl ConfidenceGate {
    pub fn new(threshold: f64) -> Self { Self { threshold, strict: false } }
    pub fn strict(threshold: f64) -> Self { Self { threshold, strict: true } }

    pub fn check(&self, confidence: Confidence) -> GatedConfidence {
        if confidence.value() >= self.threshold {
            GatedConfidence { passed: true, confidence }
        } else if self.strict {
            GatedConfidence { passed: false, confidence: Confidence::ZERO }
        } else {
            GatedConfidence { passed: false, confidence: confidence.discount(0.5) }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GatedConfidence {
    pub passed: bool,
    pub confidence: Confidence,
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_propagation() {
        let mut c = Cascade::new(0.01);
        c.add_node("source", 0.5);
        c.add_node("target", 0.5);
        c.add_edge("source", "target", 1.0);
        c.update("source", Confidence::SURE);
        let result = c.propagate();
        assert!(result.propagated >= 1);
        let target_conf = c.confidence("target").unwrap();
        assert!(target_conf.value() > 0.5); // target should have absorbed some confidence
    }

    #[test]
    fn test_convergence() {
        let mut c = Cascade::new(0.01);
        c.add_node("a", 0.9);
        c.add_node("b", 0.9);
        c.add_bidirectional_edge("a", "b", 0.5);
        let result = c.propagate_until_converged();
        assert!(result.converged);
    }

    #[test]
    fn test_multiple_sources() {
        let mut c = Cascade::new(0.01);
        c.add_node("s1", 0.5);
        c.add_node("s2", 0.5);
        c.add_node("fusion", 0.5);
        c.add_edge("s1", "fusion", 0.7);
        c.add_edge("s2", "fusion", 0.7);
        c.update("s1", Confidence::SURE);
        c.update("s2", Confidence::new(0.2));
        c.propagate();
        let fusion = c.confidence("fusion").unwrap();
        assert!(fusion.value() > 0.5);
    }

    #[test]
    fn test_decay() {
        let mut c = Cascade::new(0.01);
        c.add_node("sensor", 0.3);
        c.update("sensor", Confidence::SURE);
        c.propagate(); // triggers decay
        let conf = c.confidence("sensor").unwrap();
        // Should have decayed toward 0.3 base
        assert!(conf.value() < 1.0);
        assert!(conf.value() > 0.3);
    }

    #[test]
    fn test_weakest_strongest() {
        let mut c = Cascade::new(0.01);
        c.add_node("weak", 0.1);
        c.add_node("strong", 0.9);
        c.add_node("mid", 0.5);
        assert_eq!(c.weakest_node().unwrap().1, 0.1);
        assert_eq!(c.strongest_node().unwrap().1, 0.9);
    }

    #[test]
    fn test_confidence_gate() {
        let gate = ConfidenceGate::new(0.5);
        let passed = gate.check(Confidence::LIKELY);
        assert!(passed.passed);

        let failed = gate.check(Confidence::UNLIKELY);
        assert!(!failed.passed);
        assert!(failed.confidence.value() > 0.0); // non-strict: discounted

        let strict = ConfidenceGate::strict(0.5);
        let blocked = strict.check(Confidence::UNLIKELY);
        assert!(!blocked.passed);
        assert_eq!(blocked.confidence, Confidence::ZERO);
    }

    #[test]
    fn test_snapshot() {
        let mut c = Cascade::new(0.01);
        c.add_node("a", 0.7);
        c.add_node("b", 0.3);
        let snap = c.snapshot();
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn test_node_with_vessel() {
        let node = CascadeNode::new("scout", 0.8).with_vessel(42);
        assert!(node.vessel_id.is_some());
    }
}
