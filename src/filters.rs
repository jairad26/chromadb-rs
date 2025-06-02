use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::commons::MetadataValue;

/// A type-safe builder for ChromaDB metadata filters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetadataFilter {
    conditions: Value,
}

impl MetadataFilter {
    pub fn new(conditions: Value) -> Self {
        Self { conditions }
    }

    /// Combine multiple filters with AND logic
    pub fn and(filters: Vec<MetadataFilter>) -> Self {
        let conditions: Vec<Value> = filters
            .into_iter()
            .map(|f| f.conditions)
            .collect();
        
        Self {
            conditions: json!({ "$and": conditions }),
        }
    }

    /// Combine multiple filters with OR logic
    pub fn or(filters: Vec<MetadataFilter>) -> Self {
        let conditions: Vec<Value> = filters
            .into_iter()
            .map(|f| f.conditions)
            .collect();
        
        Self {
            conditions: json!({ "$or": conditions }),
        }
    }

    /// Convert to a JSON Value for use with ChromaDB API
    pub fn to_value(&self) -> Value {
        self.conditions.clone()
    }
}

impl Default for MetadataFilter {
    fn default() -> Self {
        Self {
            conditions: json!({}),
        }
    }
}

impl From<MetadataFilter> for Value {
    fn from(filter: MetadataFilter) -> Self {
        filter.conditions
    }
}

/// A type-safe builder for ChromaDB document filters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocumentFilter {
    conditions: Value,
}

impl DocumentFilter {
    fn new(conditions: Value) -> Self {
        Self { conditions }
    }


    /// Combine multiple document filters with AND logic (internal use)
    fn and(filters: Vec<DocumentFilter>) -> Self {
        let conditions: Vec<Value> = filters
            .into_iter()
            .map(|f| f.conditions)
            .collect();
        
        Self {
            conditions: json!({ "$and": conditions }),
        }
    }

    /// Combine multiple document filters with OR logic (internal use)
    fn or(filters: Vec<DocumentFilter>) -> Self {
        let conditions: Vec<Value> = filters
            .into_iter()
            .map(|f| f.conditions)
            .collect();
        
        Self {
            conditions: json!({ "$or": conditions }),
        }
    }

    /// Convert to a JSON Value for use with ChromaDB API
    pub fn to_value(&self) -> Value {
        self.conditions.clone()
    }

    /// Build from a raw JSON value (for advanced use cases)
    pub fn from_value(value: Value) -> Self {
        Self { conditions: value }
    }
}

impl Default for DocumentFilter {
    fn default() -> Self {
        Self {
            conditions: json!({}),
        }
    }
}

impl From<DocumentFilter> for Value {
    fn from(filter: DocumentFilter) -> Self {
        filter.conditions
    }
}

    pub fn eq<K: AsRef<str>, V: Into<MetadataValue>>(key: K, value: V) -> MetadataFilter {
        let condition = json!({ "$eq": Value::from(value.into()) });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn ne<K: AsRef<str>, V: Into<MetadataValue>>(key: K, value: V) -> MetadataFilter {
        let condition = json!({ "$ne": Value::from(value.into()) });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn gt<K: AsRef<str>, V: Into<MetadataValue>>(key: K, value: V) -> MetadataFilter {
        let condition = json!({ "$gt": Value::from(value.into()) });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn gte<K: AsRef<str>, V: Into<MetadataValue>>(key: K, value: V) -> MetadataFilter {
        let condition = json!({ "$gte": Value::from(value.into()) });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn lt<K: AsRef<str>, V: Into<MetadataValue>>(key: K, value: V) -> MetadataFilter {
        let condition = json!({ "$lt": Value::from(value.into()) });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn lte<K: AsRef<str>, V: Into<MetadataValue>>(key: K, value: V) -> MetadataFilter {
        let condition = json!({ "$lte": Value::from(value.into()) });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn in_list<K: AsRef<str>, V: Into<MetadataValue>>(key: K, values: Vec<V>) -> MetadataFilter {
        let values: Vec<Value> = values.into_iter().map(|v| Value::from(v.into())).collect();
        let condition = json!({ "$in": values });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn not_in_list<K: AsRef<str>, V: Into<MetadataValue>>(key: K, values: Vec<V>) -> MetadataFilter {
        let values: Vec<Value> = values.into_iter().map(|v| Value::from(v.into())).collect();
        let condition = json!({ "$nin": values });
        MetadataFilter::new(json!({ key.as_ref(): condition }))
    }

    pub fn and_metadata(filters: Vec<MetadataFilter>) -> MetadataFilter {
        MetadataFilter::and(filters)
    }

    pub fn or_metadata(filters: Vec<MetadataFilter>) -> MetadataFilter {
        MetadataFilter::or(filters)
    }

    pub fn contains<T: AsRef<str>>(text: T) -> DocumentFilter {
        DocumentFilter::new(json!({ "$contains": text.as_ref() }))
    }

    pub fn not_contains<T: AsRef<str>>(text: T) -> DocumentFilter {
        DocumentFilter::new(json!({ "$not_contains": text.as_ref() }))
    }

    pub fn regex<T: AsRef<str>>(pattern: T) -> DocumentFilter {
        DocumentFilter::new(json!({ "$regex": pattern.as_ref() }))
    }

    pub fn not_regex<T: AsRef<str>>(pattern: T) -> DocumentFilter {
        DocumentFilter::new(json!({ "$not_regex": pattern.as_ref() }))
    }

    pub fn and_doc(filters: Vec<DocumentFilter>) -> DocumentFilter {
        DocumentFilter::and(filters)
    }

    pub fn or_doc(filters: Vec<DocumentFilter>) -> DocumentFilter {
        DocumentFilter::or(filters)
    }


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_metadata_filter_equality() {
        let filter = eq("topic", "technology");
        assert_eq!(filter.to_value(), json!({"topic": {"$eq": "technology"}}));
    }

    #[test]
    fn test_metadata_filter_comparison() {
        let filter = gt("price", 10);
        assert_eq!(filter.to_value(), json!({"price": {"$gt": 10}}));
    }

    #[test]
    fn test_metadata_filter_range_with_and() {
        let filter = and_metadata(vec![
            eq("topic", "technology"),
            gt("price", 10),
            lt("price", 100)
        ]);
        
        assert_eq!(
            filter.to_value(),
            json!({
                "$and": [
                    {"topic": {"$eq": "technology"}},
                    {"price": {"$gt": 10}},
                    {"price": {"$lt": 100}}
                ]
            })
        );
    }

    #[test]
    fn test_metadata_filter_and() {
        let filter1 = eq("topic", "technology");
        let filter2 = gt("price", 10);
        let combined = and_metadata(vec![filter1, filter2]);
        
        assert_eq!(
            combined.to_value(),
            json!({
                "$and": [
                    {"topic": {"$eq": "technology"}},
                    {"price": {"$gt": 10}}
                ]
            })
        );
    }

    #[test]
    fn test_metadata_filter_or() {
        let filter1 = eq("topic", "technology");
        let filter2 = eq("topic", "science");
        let combined = or_metadata(vec![filter1, filter2]);
        
        assert_eq!(
            combined.to_value(),
            json!({
                "$or": [
                    {"topic": {"$eq": "technology"}},
                    {"topic": {"$eq": "science"}}
                ]
            })
        );
    }

    #[test]
    fn test_metadata_filter_in_list() {
        let filter = in_list("topic", vec!["technology", "science"]);
        assert_eq!(
            filter.to_value(),
            json!({"topic": {"$in": ["technology", "science"]}})
        );
    }

    #[test]
    fn test_metadata_filter_different_types() {
        let combined = and_metadata(vec![
            eq("title", "AI Research"),
            eq("rating", 4.5),
            eq("views", 1000i64),
            eq("published", true)
        ]);
        
        assert_eq!(
            combined.to_value(),
            json!({
                "$and": [
                    {"title": {"$eq": "AI Research"}},
                    {"rating": {"$eq": 4.5}},
                    {"views": {"$eq": 1000}},
                    {"published": {"$eq": true}}
                ]
            })
        );
    }

    #[test]
    fn test_document_filter_contains() {
        let filter = contains("hello world");
        assert_eq!(filter.to_value(), json!({"$contains": "hello world"}));
    }

    #[test]
    fn test_document_filter_not_contains() {
        let filter = not_contains("hello world");
        assert_eq!(filter.to_value(), json!({"$not_contains": "hello world"}));
    }

    #[test]
    fn test_document_filter_regex() {
        let filter = regex(r"\d{4}-\d{2}-\d{2}");
        assert_eq!(filter.to_value(), json!({"$regex": r"\d{4}-\d{2}-\d{2}"}));
    }

    #[test]
    fn test_document_filter_not_regex() {
        let filter = not_regex(r"\d+");
        assert_eq!(filter.to_value(), json!({"$not_regex": r"\d+"}));
    }

    #[test]
    fn test_document_filter_and() {
        let filter1 = contains("hello");
        let filter2 = contains("world");
        let combined = and_doc(vec![filter1, filter2]);
        
        assert_eq!(
            combined.to_value(),
            json!({
                "$and": [
                    {"$contains": "hello"},
                    {"$contains": "world"}
                ]
            })
        );
    }

    #[test]
    fn test_document_filter_or() {
        let filter1 = contains("hello");
        let filter2 = regex(r"\d+");
        let combined = or_doc(vec![filter1, filter2]);
        
        assert_eq!(
            combined.to_value(),
            json!({
                "$or": [
                    {"$contains": "hello"},
                    {"$regex": r"\d+"}
                ]
            })
        );
    }

    #[test]
    fn test_convenience_functions() {
        let filter = eq("topic", "technology");
        assert_eq!(filter.to_value(), json!({"topic": {"$eq": "technology"}}));
        
        let doc_filter = contains("hello");
        assert_eq!(doc_filter.to_value(), json!({"$contains": "hello"}));

        let regex_filter = regex(r"\w+@\w+\.\w+");
        assert_eq!(regex_filter.to_value(), json!({"$regex": r"\w+@\w+\.\w+"}));
    }

    #[test]
    fn test_complex_filter_combination() {
        let filter = and_metadata(vec![
            eq("status", "active"),
            or_metadata(vec![
                eq("category", "tech"),
                eq("category", "science")
            ]),
            gte("score", 85)
        ]);

        assert_eq!(
            filter.to_value(),
            json!({
                "$and": [
                    {"status": {"$eq": "active"}},
                    {
                        "$or": [
                            {"category": {"$eq": "tech"}},
                            {"category": {"$eq": "science"}}
                        ]
                    },
                    {"score": {"$gte": 85}}
                ]
            })
        );
    }

    #[test]
    fn test_in_list_with_mixed_types() {
        let filter = in_list("category", vec!["tech", "science"]);
        assert_eq!(
            filter.to_value(),
            json!({"category": {"$in": ["tech", "science"]}})
        );

        let filter = in_list("score", vec![85, 90, 95]);
        assert_eq!(
            filter.to_value(),
            json!({"score": {"$in": [85, 90, 95]}})
        );

        let filter = in_list("rating", vec![4.5, 3.2, 5.0]);
        assert_eq!(
            filter.to_value(),
            json!({"rating": {"$in": [4.5, 3.2, 5.0]}})
        );

        let filter = in_list("active", vec![true, false]);
        assert_eq!(
            filter.to_value(),
            json!({"active": {"$in": [true, false]}})
        );
    }

    #[test]
    fn test_not_in_list_function() {
        let filter = not_in_list("category", vec!["spam", "deleted"]);
        assert_eq!(
            filter.to_value(),
            json!({"category": {"$nin": ["spam", "deleted"]}})
        );
    }
} 