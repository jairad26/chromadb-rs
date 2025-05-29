use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::commons::Result;

/// Vector space distance metrics supported by ChromaDB
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Space {
    #[default]
    #[serde(rename = "l2")]
    L2,
    #[serde(rename = "cosine")]
    Cosine,
    #[serde(rename = "ip")]
    Ip,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateHNSWConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space: Option<Space>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_construction: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_neighbors: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_threshold: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resize_factor: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateSpannConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_nprobe: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_nprobe: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space: Option<Space>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_construction: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ef_search: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_neighbors: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reassign_neighbor_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_threshold: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merge_threshold: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CreateCollectionConfiguration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hnsw: Option<CreateHNSWConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spann: Option<CreateSpannConfiguration>,
}

impl CreateCollectionConfiguration {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn hnsw(mut self, hnsw: CreateHNSWConfiguration) -> Self {
        self.hnsw = Some(hnsw);
        self
    }

    pub fn spann(mut self, spann: CreateSpannConfiguration) -> Self {
        self.spann = Some(spann);
        self
    }

    pub fn to_configuration(&self) -> Result<Map<String, Value>> {
        let json = serde_json::to_value(self)?;
        match json {
            Value::Object(map) => Ok(map),
            _ => Ok(Map::new()),
        }
    }
}

impl CreateHNSWConfiguration {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn space(mut self, space: Space) -> Self {
        self.space = Some(space);
        self
    }

    pub fn ef_construction(mut self, ef_construction: i32) -> Self {
        self.ef_construction = Some(ef_construction);
        self
    }

    pub fn max_neighbors(mut self, max_neighbors: i32) -> Self {
        self.max_neighbors = Some(max_neighbors);
        self
    }

    pub fn ef_search(mut self, ef_search: i32) -> Self {
        self.ef_search = Some(ef_search);
        self
    }

    pub fn sync_threshold(mut self, sync_threshold: i32) -> Self {
        self.sync_threshold = Some(sync_threshold);
        self
    }

    pub fn resize_factor(mut self, resize_factor: f64) -> Self {
        self.resize_factor = Some(resize_factor);
        self
    }
}

impl CreateSpannConfiguration {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn search_nprobe(mut self, search_nprobe: i32) -> Self {
        self.search_nprobe = Some(search_nprobe);
        self
    }

    pub fn write_nprobe(mut self, write_nprobe: i32) -> Self {
        self.write_nprobe = Some(write_nprobe);
        self
    }

    pub fn space(mut self, space: Space) -> Self {
        self.space = Some(space);
        self
    }

    pub fn ef_construction(mut self, ef_construction: i32) -> Self {
        self.ef_construction = Some(ef_construction);
        self
    }

    pub fn ef_search(mut self, ef_search: i32) -> Self {
        self.ef_search = Some(ef_search);
        self
    }

    pub fn max_neighbors(mut self, max_neighbors: i32) -> Self {
        self.max_neighbors = Some(max_neighbors);
        self
    }

    pub fn reassign_neighbor_count(mut self, reassign_neighbor_count: i32) -> Self {
        self.reassign_neighbor_count = Some(reassign_neighbor_count);
        self
    }

    pub fn split_threshold(mut self, split_threshold: i32) -> Self {
        self.split_threshold = Some(split_threshold);
        self
    }

    pub fn merge_threshold(mut self, merge_threshold: i32) -> Self {
        self.merge_threshold = Some(merge_threshold);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_hnsw_configuration() {
        let config = CreateHNSWConfiguration::new()
            .space(Space::Cosine)
            .ef_construction(200)
            .max_neighbors(16)
            .ef_search(100)
            .sync_threshold(1000)
            .resize_factor(1.2);

        assert_eq!(config.space, Some(Space::Cosine));
        assert_eq!(config.ef_construction, Some(200));
        assert_eq!(config.max_neighbors, Some(16));
        assert_eq!(config.ef_search, Some(100));
        assert_eq!(config.sync_threshold, Some(1000));
        assert_eq!(config.resize_factor, Some(1.2));
    }

    #[test]
    fn test_create_spann_configuration() {
        let config = CreateSpannConfiguration::new()
            .space(Space::L2)
            .search_nprobe(10)
            .write_nprobe(20)
            .ef_construction(300)
            .ef_search(150)
            .max_neighbors(32)
            .reassign_neighbor_count(5)
            .split_threshold(2000)
            .merge_threshold(500);

        assert_eq!(config.space, Some(Space::L2));
        assert_eq!(config.search_nprobe, Some(10));
        assert_eq!(config.write_nprobe, Some(20));
        assert_eq!(config.ef_construction, Some(300));
        assert_eq!(config.ef_search, Some(150));
        assert_eq!(config.max_neighbors, Some(32));
        assert_eq!(config.reassign_neighbor_count, Some(5));
        assert_eq!(config.split_threshold, Some(2000));
        assert_eq!(config.merge_threshold, Some(500));
    }

    #[test]
    fn test_create_collection_configuration() {
        let hnsw_config = CreateHNSWConfiguration::new()
            .space(Space::Cosine)
            .ef_construction(200);

        let spann_config = CreateSpannConfiguration::new()
            .space(Space::L2)
            .search_nprobe(10);

        let config = CreateCollectionConfiguration::new()
            .hnsw(hnsw_config.clone())
            .spann(spann_config.clone());

        assert!(config.hnsw.is_some());
        assert!(config.spann.is_some());
        assert_eq!(config.hnsw.unwrap().space, Some(Space::Cosine));
        assert_eq!(config.spann.unwrap().space, Some(Space::L2));
    }

    #[test]
    fn test_to_configuration_conversion() {
        let hnsw_config = CreateHNSWConfiguration::new()
            .space(Space::Cosine)
            .ef_construction(200);

        let config = CreateCollectionConfiguration::new().hnsw(hnsw_config);

        let json_config = config.to_configuration().unwrap();
        assert!(json_config.contains_key("hnsw"));
        
        // Verify the nested structure is properly serialized
        if let Some(hnsw_value) = json_config.get("hnsw") {
            if let Some(hnsw_obj) = hnsw_value.as_object() {
                assert!(hnsw_obj.contains_key("space"));
                assert!(hnsw_obj.contains_key("ef_construction"));
                assert_eq!(hnsw_obj.get("space").unwrap().as_str(), Some("cosine"));
                assert_eq!(hnsw_obj.get("ef_construction").unwrap().as_i64(), Some(200));
            }
        }
    }

    #[test]
    fn test_space_serialization() {
        use serde_json;

        assert_eq!(serde_json::to_string(&Space::L2).unwrap(), "\"l2\"");
        assert_eq!(serde_json::to_string(&Space::Cosine).unwrap(), "\"cosine\"");
        assert_eq!(serde_json::to_string(&Space::Ip).unwrap(), "\"ip\"");
    }
} 