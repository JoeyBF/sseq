use std::{collections::BTreeSet, fmt::Display, io};

use serde_json::{Map, Value, json};

use crate::{
    charting::{Backend, Orientation},
    coordinates::{Bidegree, BidegreeGenerator},
};

/// A [`Backend`] that emits [SeqSee](https://github.com/JoeyBF/SeqSee) JSON.
///
/// SeqSee is a generic visualization tool for spectral sequences. It consumes a JSON file
/// conforming to its [schema] and produces a self-contained, interactive HTML chart. This backend
/// targets that schema so that the charts produced elsewhere in this crate can be rendered by
/// SeqSee.
///
/// The document is assembled in memory and written out when the backend is dropped, since the JSON
/// object must group all nodes together and all edges together, whereas the [`Backend`] callbacks
/// interleave them.
///
/// [schema]: https://raw.githubusercontent.com/JoeyBF/SeqSee/refs/heads/master/seqsee/input_schema.json
pub struct SeqSeeBackend<T: io::Write> {
    out: T,
    max: Bidegree,
    /// Map from node id to its SeqSee node object.
    nodes: Map<String, Value>,
    /// The list of SeqSee edge objects.
    edges: Vec<Value>,
    /// The set of style names referenced by edges or nodes. Each becomes an attribute alias in the
    /// header so that the objects referencing it validate against the schema.
    styles: BTreeSet<String>,
    /// Explicit attribute-alias definitions, `name -> attributes array`. These override the
    /// auto-generated `[]`/differential defaults for the same name at [`Drop`] time.
    attribute_defs: Map<String, Value>,
}

impl<T: io::Write> SeqSeeBackend<T> {
    pub fn new(out: T) -> Self {
        Self {
            out,
            max: Bidegree::zero(),
            nodes: Map::new(),
            edges: Vec::new(),
            styles: BTreeSet::new(),
            attribute_defs: Map::new(),
        }
    }

    /// Register a named attribute alias with an explicit spec, e.g.
    /// `define_attribute("tau3", json!([{ "color": "#d62728" }]))`.
    ///
    /// The `spec` is a SeqSee attributes array (each item a `{color, size, ...}` object). It appears
    /// under `header.aliases.attributes` and takes precedence over the auto-generated default for a
    /// name of the same key. Nodes and edges reference the alias by name via [`Self::styled_node`]
    /// and [`Backend::structline`].
    pub fn define_attribute(&mut self, name: &str, spec: Value) {
        self.attribute_defs.insert(name.to_string(), spec);
    }

    /// Emit a single node at bidegree `b`, position index `position`, carrying the named attribute
    /// aliases `attrs` (colors, sizes, ... registered via [`Self::define_attribute`]) and an
    /// optional `label`.
    ///
    /// Like [`Backend::node`], out-of-bounds nodes (beyond `max`) are silently skipped, and the node
    /// id uses the same `(x,y,idx)` format so structline endpoints stay in sync. Any referenced
    /// attribute name that was not explicitly defined is auto-registered as an empty alias.
    pub fn styled_node(
        &mut self,
        b: Bidegree,
        position: usize,
        attrs: &[String],
        label: Option<String>,
    ) -> Result<(), io::Error> {
        if b.x() > self.max.x() || b.y() > self.max.y() {
            return Ok(());
        }

        let id = format!("{:#}", BidegreeGenerator::new(b, position));
        let mut node = Map::new();
        node.insert("x".to_string(), json!(b.x()));
        node.insert("y".to_string(), json!(b.y()));
        node.insert("position".to_string(), json!(position));
        if !attrs.is_empty() {
            for a in attrs {
                self.styles.insert(a.clone());
            }
            node.insert("attributes".to_string(), json!(attrs));
        }
        if let Some(label) = label {
            node.insert("label".to_string(), Value::String(label));
        }
        self.nodes.insert(id, Value::Object(node));
        Ok(())
    }
}

/// Whether a style name denotes a differential, i.e. it is `d` followed by a page number.
///
/// Differentials are given a distinct color to match the other backends in this crate.
fn is_differential(style: &str) -> bool {
    style
        .strip_prefix('d')
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

impl<T: io::Write> Backend for SeqSeeBackend<T> {
    type Error = io::Error;

    const EXT: &'static str = "json";

    fn header(&mut self, max: Bidegree) -> Result<(), Self::Error> {
        self.max = max;
        Ok(())
    }

    // SeqSee draws its own grid based on the chart dimensions in the header, so there is nothing to
    // emit for the grid lines.
    fn line(&mut self, _start: Bidegree, _end: Bidegree, _style: &str) -> Result<(), Self::Error> {
        Ok(())
    }

    // SeqSee draws its own axis labels, so there is nothing to emit here.
    fn text(
        &mut self,
        _b: Bidegree,
        _content: impl Display,
        _orientation: Orientation,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn node(&mut self, b: Bidegree, n: usize) -> Result<(), Self::Error> {
        if n == 0 || b.x() > self.max.x() || b.y() > self.max.y() {
            return Ok(());
        }

        for k in 0..n {
            // The `{:#}` (alternate) form of the `Display` impl is `(x,y,idx)`. Reusing it here and
            // in `structline` keeps node ids and edge endpoints in sync.
            let id = format!("{:#}", BidegreeGenerator::new(b, k));
            self.nodes.insert(
                id,
                json!({
                    "x": b.x(),
                    "y": b.y(),
                    "position": k,
                }),
            );
        }
        Ok(())
    }

    fn structline(
        &mut self,
        source: BidegreeGenerator,
        target: BidegreeGenerator,
        style: Option<&str>,
    ) -> Result<(), Self::Error> {
        if source.x() > self.max.x()
            || source.y() > self.max.y()
            || target.x() > self.max.x()
            || target.y() > self.max.y()
        {
            return Ok(());
        }

        let mut edge = Map::new();
        edge.insert("source".to_string(), Value::String(format!("{source:#}")));
        edge.insert("target".to_string(), Value::String(format!("{target:#}")));
        if let Some(style) = style {
            self.styles.insert(style.to_string());
            edge.insert("attributes".to_string(), json!([style]));
        }
        self.edges.push(Value::Object(edge));
        Ok(())
    }
}

impl<T: io::Write> Drop for SeqSeeBackend<T> {
    fn drop(&mut self) {
        // Register every referenced style as an attribute alias so that the edges validate.
        let mut attributes = Map::new();
        for style in &self.styles {
            let attr = if is_differential(style) {
                json!([{ "color": "blue" }])
            } else {
                json!([])
            };
            attributes.insert(style.clone(), attr);
        }
        // Explicit definitions win over the auto-generated defaults above.
        for (name, spec) in &self.attribute_defs {
            attributes.insert(name.clone(), spec.clone());
        }

        let document = json!({
            "header": {
                "chart": {
                    "width": { "min": 0, "max": self.max.x() },
                    "height": { "min": 0, "max": self.max.y() },
                },
                "aliases": {
                    "attributes": attributes,
                },
            },
            "nodes": std::mem::take(&mut self.nodes),
            "edges": std::mem::take(&mut self.edges),
        });

        let _ = serde_json::to_writer_pretty(&mut self.out, &document);
        let _ = writeln!(self.out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seqsee_output() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut backend = SeqSeeBackend::new(&mut buf);
            // `init` draws the grid and axis labels via `line`/`text`, which are no-ops here.
            backend.init(Bidegree::x_y(2, 2)).unwrap();

            backend.node(Bidegree::x_y(0, 0), 1).unwrap();
            backend.node(Bidegree::x_y(0, 1), 1).unwrap();
            backend.node(Bidegree::x_y(1, 1), 1).unwrap();

            // A multiplication by `h0` (a filtration-one product).
            backend
                .structline(
                    BidegreeGenerator::new(Bidegree::x_y(0, 0), 0),
                    BidegreeGenerator::new(Bidegree::x_y(0, 1), 0),
                    Some("h0"),
                )
                .unwrap();
            // A `d2` differential.
            backend
                .structline(
                    BidegreeGenerator::new(Bidegree::x_y(1, 1), 0),
                    BidegreeGenerator::new(Bidegree::x_y(0, 1), 0),
                    Some("d2"),
                )
                .unwrap();
            // Out of bounds: dropped silently.
            backend.node(Bidegree::x_y(5, 5), 1).unwrap();
        }

        // Compare parsed values rather than the raw string so the assertion does not depend on
        // JSON object key ordering, which varies with whether serde_json's `preserve_order`
        // feature is enabled by feature unification in the surrounding build.
        let produced: Value = serde_json::from_slice(&buf).unwrap();
        let expected = json!({
            "header": {
                "chart": {
                    "width": { "min": 0, "max": 2 },
                    "height": { "min": 0, "max": 2 },
                },
                "aliases": {
                    "attributes": {
                        "h0": [],
                        "d2": [{ "color": "blue" }],
                    },
                },
            },
            "nodes": {
                "(0,0,0)": { "x": 0, "y": 0, "position": 0 },
                "(0,1,0)": { "x": 0, "y": 1, "position": 0 },
                "(1,1,0)": { "x": 1, "y": 1, "position": 0 },
            },
            "edges": [
                { "source": "(0,0,0)", "target": "(0,1,0)", "attributes": ["h0"] },
                { "source": "(1,1,0)", "target": "(0,1,0)", "attributes": ["d2"] },
            ],
        });
        assert_eq!(produced, expected);
    }

    #[test]
    fn test_styled_node_and_define_attribute() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut backend = SeqSeeBackend::new(&mut buf);
            backend.init(Bidegree::x_y(2, 2)).unwrap();

            // Explicit definition; overrides nothing, so appears verbatim.
            backend.define_attribute("tau2", json!([{ "color": "#d62728" }]));

            // Styled node referencing the explicit attr, plus a label.
            backend
                .styled_node(Bidegree::x_y(0, 0), 0, &["tau2".to_string()], Some("x".to_string()))
                .unwrap();
            // Styled node referencing an attr never explicitly defined: auto-registered as `[]`.
            backend
                .styled_node(Bidegree::x_y(1, 1), 0, &["free".to_string()], None)
                .unwrap();
            // Plain node, no attributes.
            backend.styled_node(Bidegree::x_y(0, 1), 0, &[], None).unwrap();
            // Out of bounds: dropped silently.
            backend
                .styled_node(Bidegree::x_y(9, 9), 0, &["free".to_string()], None)
                .unwrap();
        }

        let produced: Value = serde_json::from_slice(&buf).unwrap();
        let expected = json!({
            "header": {
                "chart": {
                    "width": { "min": 0, "max": 2 },
                    "height": { "min": 0, "max": 2 },
                },
                "aliases": {
                    "attributes": {
                        "tau2": [{ "color": "#d62728" }],
                        "free": [],
                    },
                },
            },
            "nodes": {
                "(0,0,0)": { "x": 0, "y": 0, "position": 0, "attributes": ["tau2"], "label": "x" },
                "(1,1,0)": { "x": 1, "y": 1, "position": 0, "attributes": ["free"] },
                "(0,1,0)": { "x": 0, "y": 1, "position": 0 },
            },
            "edges": [],
        });
        assert_eq!(produced, expected);
    }

    #[test]
    fn test_define_attribute_overrides_default() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut backend = SeqSeeBackend::new(&mut buf);
            backend.init(Bidegree::x_y(2, 2)).unwrap();
            // `d2` would default to blue; an explicit definition must win.
            backend.define_attribute("d2", json!([{ "color": "green" }]));
            backend
                .structline(
                    BidegreeGenerator::new(Bidegree::x_y(1, 1), 0),
                    BidegreeGenerator::new(Bidegree::x_y(0, 1), 0),
                    Some("d2"),
                )
                .unwrap();
        }
        let produced: Value = serde_json::from_slice(&buf).unwrap();
        assert_eq!(
            produced["header"]["aliases"]["attributes"]["d2"],
            json!([{ "color": "green" }])
        );
    }

    #[test]
    fn test_is_differential() {
        assert!(is_differential("d2"));
        assert!(is_differential("d17"));
        assert!(!is_differential("d"));
        assert!(!is_differential("h0"));
        assert!(!is_differential("delta"));
    }
}
