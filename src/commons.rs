use serde_json::{Map, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(super) type Result<T> = anyhow::Result<T>;
pub(super) type Metadata = HashMap<String, MetadataValue>;
pub(super) type ConfigurationJson = Map<String, Value>;
pub(super) type Metadatas = Vec<Metadata>;
pub(super) type Embedding = Vec<f32>;
pub(super) type Embeddings = Vec<Embedding>;
pub(super) type Documents<'a> = Vec<&'a str>;

/// Type-safe metadata values that match ChromaDB's supported types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::Str(s)
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::Str(s.to_string())
    }
}

impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        MetadataValue::Int(i)
    }
}

impl From<i32> for MetadataValue {
    fn from(i: i32) -> Self {
        MetadataValue::Int(i as i64)
    }
}

impl From<f64> for MetadataValue {
    fn from(f: f64) -> Self {
        MetadataValue::Float(f)
    }
}

impl From<f32> for MetadataValue {
    fn from(f: f32) -> Self {
        MetadataValue::Float(f as f64)
    }
}

impl From<bool> for MetadataValue {
    fn from(b: bool) -> Self {
        MetadataValue::Bool(b)
    }
}

impl From<MetadataValue> for Value {
    fn from(mv: MetadataValue) -> Self {
        match mv {
            MetadataValue::Str(s) => Value::String(s),
            MetadataValue::Int(i) => Value::Number(serde_json::Number::from(i)),
            MetadataValue::Float(f) => Value::Number(serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0))),
            MetadataValue::Bool(b) => Value::Bool(b),
            MetadataValue::Null => Value::Null,
        }
    }
}
