use anyhow::bail;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashSet, sync::Arc, vec};

use super::{
    api::APIClientAsync,
    commons::{ConfigurationJson, Documents, Embedding, Embeddings, Metadata, Metadatas, Result},
    embeddings::EmbeddingFunction,
    filters::{MetadataFilter, DocumentFilter},
};

/// A collection representation for interacting with the associated ChromaDB collection.
#[derive(Clone, Deserialize, Debug, Default)]
pub struct ChromaCollection {
    #[serde(skip)]
    pub(super) api: Arc<APIClientAsync>,
    pub(super) id: String,
    pub(super) metadata: Option<Metadata>,
    pub(super) name: String,
    pub(super) configuration_json: ConfigurationJson,
    pub(super) dimension: Option<usize>,
    pub(super) tenant: String,
    pub(super) database: String,
    #[serde(default)]
    pub(super) version: usize,
    #[serde(default)]
    pub(super) log_position: usize,
}

impl ChromaCollection {
    /// Get the UUID of the collection.
    pub fn id(&self) -> &str {
        self.id.as_ref()
    }

    /// Get the name of the collection.
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Get the metadata of the collection.
    pub fn metadata(&self) -> Option<&Metadata> {
        self.metadata.as_ref()
    }

    /// Get the configuration of the collection.
    pub fn configuration_json(&self) -> &ConfigurationJson {
        &self.configuration_json
    }

    /// Get the dimension of the collection.
    pub fn dimension(&self) -> Option<&usize> {
        self.dimension.as_ref()
    }

    /// Get the tenant of the collection.
    pub fn tenant(&self) -> &str {
        self.tenant.as_ref()
    }

    /// Get the database of the collection.
    pub fn database(&self) -> &str {
        self.database.as_ref()
    }

    /// Get the version of the collection.
    pub fn version(&self) -> usize {
        self.version
    }

    /// Get the log position of the collection.
    pub fn log_position(&self) -> usize {
        self.log_position
    }
    /// The total number of embeddings added to the database.
    pub async fn count(&self) -> Result<usize> {
        let path = format!("/collections/{}/count", self.id);
        let response = self.api.get_database(&path).await?;
        let count = response.json::<usize>().await?;
        Ok(count)
    }

    /// Modify the name/metadata of a collection.
    ///
    /// # Arguments
    ///
    /// * `name` - The new name of the collection. Must be unique.
    /// * `metadata` - The new metadata of the collection. Must be a JSON object with keys and values that are either numbers, strings or floats.
    ///
    /// # Errors
    ///
    /// * If the collection name is invalid
    pub async fn modify(&self, name: Option<&str>, metadata: Option<&Metadata>) -> Result<()> {
        let json_body = json!({
            "new_name": name,
            "new_metadata": metadata,
        });
        let path = format!("/collections/{}", self.id);
        self.api.put_database(&path, Some(json_body)).await?;
        Ok(())
    }

    /// Fork a collection
    /// 
    /// # Arguments
    /// 
    /// * `new_name` - The name of the new collection.
    /// 
    /// 
    pub async fn fork(&self, new_name: &str) -> Result<ChromaCollection> {
        let json_body = json!({
            "new_name": new_name,
        });
        let path = format!("/collections/{}/fork", self.id);
        let response = self.api.post_database(&path, Some(json_body)).await?;
        let mut collection = response.json::<ChromaCollection>().await?;
        collection.api = self.api.clone();
        Ok(collection)
    }

    /// Add embeddings to the data store. Ignore the insert if the ID already exists.
    ///
    /// # Arguments
    ///
    /// * `ids` - The ids to associate with the embeddings.
    /// * `embeddings` -  The embeddings to add. If None, embeddings will be computed based on the documents using the provided embedding_function. Optional.
    /// * `metadata` - The metadata to associate with the embeddings. When querying, you can filter on this metadata. Optional.
    /// * `documents` - The documents to associate with the embeddings. Optional.
    /// * `embedding_function` - The function to use to compute the embeddings. If None, embeddings must be provided. Optional.
    ///
    /// # Errors
    ///
    /// * If you don't provide either embeddings or documents
    /// * If the length of ids, embeddings, metadatas, or documents don't match
    /// * If you provide duplicates in ids, empty ids
    /// * If you provide documents and don't provide an embedding function when embeddings is None
    /// * If you provide an embedding function and don't provide documents
    /// * If you provide both embeddings and embedding_function
    ///
    pub async fn add<'a>(
        &self,
        collection_entries: CollectionEntries<'a>,
        embedding_function: Option<Box<dyn EmbeddingFunction>>,
    ) -> Result<Value> {
        let collection_entries = validate(true, collection_entries, embedding_function).await?;

        let CollectionEntries {
            ids,
            embeddings,
            metadatas,
            documents,
        } = collection_entries;

        let json_body = json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "documents": documents,
        });

        let path = format!("/collections/{}/add", self.id);
        let response = self.api.post_database(&path, Some(json_body)).await?;
        let response = response.json::<Value>().await?;

        Ok(response)
    }

    /// Add embeddings to the data store. Update the entry if an ID already exists.
    ///
    /// # Arguments
    ///
    /// * `ids` - The ids to associate with the embeddings.
    /// * `embeddings` -  The embeddings to add. If None, embeddings will be computed based on the documents using the provided embedding_function. Optional.
    /// * `metadata` - The metadata to associate with the embeddings. When querying, you can filter on this metadata. Optional.
    /// * `documents` - The documents to associate with the embeddings. Optional.
    /// * `embedding_function` - The function to use to compute the embeddings. If None, embeddings must be provided. Optional.
    ///
    /// # Errors
    ///
    /// * If you don't provide either embeddings or documents
    /// * If the length of ids, embeddings, metadatas, or documents don't match
    /// * If you provide duplicates in ids, empty ids
    /// * If you provide documents and don't provide an embedding function when embeddings is None
    /// * If you provide an embedding function and don't provide documents
    /// * If you provide both embeddings and embedding_function
    ///
    pub async fn upsert<'a>(
        &self,
        collection_entries: CollectionEntries<'a>,
        embedding_function: Option<Box<dyn EmbeddingFunction>>,
    ) -> Result<Value> {
        let collection_entries = validate(true, collection_entries, embedding_function).await?;

        let CollectionEntries {
            ids,
            embeddings,
            metadatas,
            documents,
        } = collection_entries;

        let json_body = json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "documents": documents,
        });

        let path = format!("/collections/{}/upsert", self.id);
        let response = self.api.post_database(&path, Some(json_body)).await?;
        let response = response.json::<Value>().await?;

        Ok(response)
    }

    /// Get embeddings and their associated data from the collection. If no ids or filter is provided returns all embeddings up to limit starting at offset.
    ///
    /// # Arguments
    ///
    /// * `ids` - The ids of the embeddings to get. Optional..
    /// * `where_metadata` - Used to filter results by metadata. E.g. `{ "$and": [{"foo": "bar"}, {"price": {"$gte": 4.20}}] }`. See <https://docs.trychroma.com/usage-guide#filtering-by-metadata> for more information on metadata filters. Optional.
    /// * `limit` - The maximum number of documents to return. Optional.
    /// * `offset` - The offset to start returning results from. Useful for paging results with limit. Optional.
    /// * `where_document` - Used to filter by the documents. E.g. {"$contains": "hello"}. See <https://docs.trychroma.com/usage-guide#filtering-by-document-contents> for more information on document content filters. Optional.
    /// * `include` - A list of what to include in the results. Can contain `"embeddings"`, `"metadatas"`, `"documents"`. Ids are always included. Defaults to `["metadatas", "documents"]`. Optional.
    ///
    pub async fn get(&self, get_options: GetOptions) -> Result<GetResult> {
        let GetOptions {
            ids,
            where_metadata,
            limit,
            offset,
            where_document,
            include,
        } = get_options;
        let mut json_body = json!({
            "ids": if !ids.is_empty() { Some(ids) } else { None },
            "where": where_metadata.map(|w| w.to_value()),
            "limit": limit,
            "offset": offset,
            "where_document": where_document.map(|w| w.to_value()),
            "include": include
        });

        json_body
            .as_object_mut()
            .unwrap()
            .retain(|_, v| !v.is_null());

        let path = format!("/collections/{}/get", self.id);
        let response = self.api.post_database(&path, Some(json_body)).await?;
        let get_result = response.json::<GetResult>().await?;
        Ok(get_result)
    }

    /// Update the embeddings, metadatas or documents for provided ids.
    ///
    /// # Arguments
    ///
    /// * `ids` - The ids to associate with the embeddings.
    /// * `embeddings` -  The embeddings to add. If None, embeddings will be computed based on the documents using the provided embedding_function. Optional.
    /// * `metadata` - The metadata to associate with the embeddings. When querying, you can filter on this metadata. Optional.
    /// * `documents` - The documents to associate with the embeddings. Optional.
    /// * `embedding_function` - The function to use to compute the embeddings. If None, embeddings must be provided. Optional.
    ///
    /// # Errors
    ///
    /// * If the length of ids, embeddings, metadatas, or documents don't match
    /// * If you provide duplicates in ids, empty ids
    /// * If you provide documents and don't provide an embedding function when embeddings is None
    /// * If you provide an embedding function and don't provide documents
    /// * If you provide both embeddings and embedding_function
    ///
    pub async fn update<'a>(
        &self,
        collection_entries: CollectionEntries<'a>,
        embedding_function: Option<Box<dyn EmbeddingFunction>>,
    ) -> Result<()> {
        let collection_entries = validate(false, collection_entries, embedding_function).await?;

        let CollectionEntries {
            ids,
            embeddings,
            metadatas,
            documents,
        } = collection_entries;

        let json_body = json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "documents": documents,
        });

        let path = format!("/collections/{}/update", self.id);
        let response = self.api.post_database(&path, Some(json_body)).await?;

        match response.error_for_status() {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    ///Get the n_results nearest neighbor embeddings for provided query_embeddings or query_texts.
    ///
    /// # Arguments
    ///
    /// * `query_embeddings` - The embeddings to get the closest neighbors of. Optional.
    /// * `query_texts` -  The document texts to get the closest neighbors of. Optional.
    /// * `n_results` - The number of neighbors to return for each query_embedding or query_texts. Optional.
    /// * `where_metadata` - Used to filter results by metadata. E.g. {"$and": ["color" : "red", "price": {"$gte": 4.20}]}. Optional.
    /// * `where_document` - Used to filter results by documents. E.g. {$contains: "some text"}. Optional.
    /// * `include` - A list of what to include in the results. Can contain "embeddings", "metadatas", "documents", "distances". Ids are always included. Defaults to ["metadatas", "documents", "distances"]. Optional.
    /// * `embedding_function` - The function to use to compute the embeddings. If None, embeddings must be provided. Optional.
    ///
    /// # Errors
    ///
    /// * If you don't provide either query_embeddings or query_texts
    /// * If you provide both query_embeddings and query_texts
    /// * If you provide query_texts and don't provide an embedding function when embeddings is None
    ///
    pub async fn query<'a>(
        &self,
        query_options: QueryOptions<'a>,
        embedding_function: Option<Box<dyn EmbeddingFunction>>,
    ) -> Result<QueryResult> {
        let QueryOptions {
            mut query_embeddings,
            query_texts,
            n_results,
            where_metadata,
            where_document,
            include,
        } = query_options;
        if query_embeddings.is_some() && query_texts.is_some() {
            bail!("You can only provide query_embeddings or query_texts, not both");
        } else if query_embeddings.is_none() && query_texts.is_none() {
            bail!("You must provide either query_embeddings or query_texts");
        } else if query_texts.is_some() && embedding_function.is_none() {
            bail!("You must provide an embedding function when providing query_texts");
        } else if query_embeddings.is_none() && embedding_function.is_some() {
            query_embeddings = Some(
                embedding_function
                    .unwrap()
                    .embed(query_texts.as_ref().unwrap())
                    .await?,
            );
        };

        let mut json_body = json!({
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where_metadata.map(|w| w.to_value()),
            "where_document": where_document.map(|w| w.to_value()),
            "include": include
        });

        json_body
            .as_object_mut()
            .unwrap()
            .retain(|_, v| !v.is_null());

        let path = format!("/collections/{}/query", self.id);
        let response = self.api.post_database(&path, Some(json_body)).await?;
        let query_result = response.json::<QueryResult>().await?;
        Ok(query_result)
    }

    ///Get the first entries in the collection up to the limit
    ///
    /// # Arguments
    ///
    /// * `limit` - The number of entries to return.
    ///
    pub async fn peek(&self, limit: usize) -> Result<GetResult> {
        let get_query = GetOptions {
            ids: vec![],
            where_metadata: None,
            limit: Some(limit),
            offset: None,
            where_document: None,
            include: None,
        };
        self.get(get_query).await
    }

    /// Delete the embeddings based on ids and/or a where filter. Deletes all the entries if None are provided
    ///
    /// # Arguments
    ///
    /// * `ids` - The ids of the embeddings to delete. Optional
    /// * `where_metadata` -  Used to filter deletion by metadata. Optional.
    /// * `where_document` - Used to filter the deletion by the document content. Optional.
    ///
    pub async fn delete(
        &self,
        ids: Option<Vec<&str>>,
        where_metadata: Option<MetadataFilter>,
        where_document: Option<DocumentFilter>,
    ) -> Result<()> {
        let json_body = json!({
            "ids": ids,
            "where": where_metadata.map(|w| w.to_value()),
            "where_document": where_document.map(|w| w.to_value()),
        });

        let path = format!("/collections/{}/delete", self.id);
        let response = self.api.post_database(&path, Some(json_body)).await?;

        match response.error_for_status() {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

#[derive(Deserialize, Debug)]
pub struct GetResult {
    pub ids: Vec<String>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
    pub embeddings: Option<Vec<Option<Embedding>>>,
}

#[derive(Serialize, Debug, Default)]
pub struct GetOptions {
    pub ids: Vec<String>,
    pub where_metadata: Option<MetadataFilter>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub where_document: Option<DocumentFilter>,
    pub include: Option<Vec<String>>,
}

impl GetOptions {
    /// Create a new GetOptions with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the IDs to filter by
    pub fn ids(mut self, ids: Vec<String>) -> Self {
        self.ids = ids;
        self
    }

    /// Set metadata filter using the type-safe MetadataFilter
    pub fn where_metadata(mut self, filter: MetadataFilter) -> Self {
        self.where_metadata = Some(filter);
        self
    }

    /// Set document filter using the type-safe DocumentFilter
    pub fn where_document(mut self, filter: DocumentFilter) -> Self {
        self.where_document = Some(filter);
        self
    }

    /// Set the limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set the offset
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Set what to include in the results
    pub fn include(mut self, include: Vec<String>) -> Self {
        self.include = Some(include);
        self
    }
}

#[derive(Serialize, Debug, Default)]
pub struct QueryOptions<'a> {
    pub query_embeddings: Option<Embeddings>,
    pub query_texts: Option<Vec<&'a str>>,
    pub n_results: Option<usize>,
    pub where_metadata: Option<MetadataFilter>,
    pub where_document: Option<DocumentFilter>,
    pub include: Option<Vec<&'a str>>,
}

impl<'a> QueryOptions<'a> {
    /// Create a new QueryOptions with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set query embeddings
    pub fn query_embeddings(mut self, embeddings: Embeddings) -> Self {
        self.query_embeddings = Some(embeddings);
        self
    }

    /// Set query texts
    pub fn query_texts(mut self, texts: Vec<&'a str>) -> Self {
        self.query_texts = Some(texts);
        self
    }

    /// Set the number of results to return
    pub fn n_results(mut self, n: usize) -> Self {
        self.n_results = Some(n);
        self
    }

    /// Set metadata filter using the type-safe MetadataFilter
    pub fn where_metadata(mut self, filter: MetadataFilter) -> Self {
        self.where_metadata = Some(filter);
        self
    }

    /// Set document filter using the type-safe DocumentFilter
    pub fn where_document(mut self, filter: DocumentFilter) -> Self {
        self.where_document = Some(filter);
        self
    }

    /// Set what to include in the results
    pub fn include(mut self, include: Vec<&'a str>) -> Self {
        self.include = Some(include);
        self
    }
}

#[derive(Deserialize, Debug)]
pub struct QueryResult {
    pub ids: Vec<Vec<String>>,
    pub metadatas: Option<Vec<Vec<Option<Metadata>>>>,
    pub documents: Option<Vec<Vec<String>>>,
    pub embeddings: Option<Vec<Vec<Embedding>>>,
    pub distances: Option<Vec<Vec<f32>>>,
}

#[derive(Serialize, Debug, Default)]
pub struct CollectionEntries<'a> {
    pub ids: Vec<&'a str>,
    pub metadatas: Option<Metadatas>,
    pub documents: Option<Documents<'a>>,
    pub embeddings: Option<Embeddings>,
}

impl<'a> CollectionEntries<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ids(mut self, ids: Vec<&'a str>) -> Self {
        self.ids = ids;
        self
    }

    pub fn metadatas(mut self, metadatas: Metadatas) -> Self {
        self.metadatas = Some(metadatas);
        self
    }

    pub fn documents(mut self, documents: Documents<'a>) -> Self {
        self.documents = Some(documents);
        self
    }

    pub fn embeddings(mut self, embeddings: Embeddings) -> Self {
        self.embeddings = Some(embeddings);
        self
    }
}

async fn validate(
    require_embeddings_or_documents: bool,
    collection_entries: CollectionEntries<'_>,
    embedding_function: Option<Box<dyn EmbeddingFunction>>,
) -> Result<CollectionEntries<'_>> {
    let CollectionEntries {
        ids,
        mut embeddings,
        metadatas,
        documents,
    } = collection_entries;
    if require_embeddings_or_documents && embeddings.is_none() && documents.is_none() {
        bail!("Embeddings and documents cannot both be None",);
    }

    if embeddings.is_none() && documents.is_some() && embedding_function.is_none() {
        bail!(
            "embedding_function cannot be None if documents are provided and embeddings are None",
        );
    }

    if embeddings.is_some() && embedding_function.is_some() {
        bail!("embedding_function should be None if embeddings are provided",);
    }

    if embeddings.is_none() && documents.is_some() && embedding_function.is_some() {
        embeddings = Some(
            embedding_function
                .unwrap()
                .embed(documents.as_ref().unwrap())
                .await?,
        );
    }

    for id in &ids {
        if id.is_empty() {
            bail!("Found empty string in IDs");
        }
    }

    if (embeddings.is_some() && embeddings.as_ref().unwrap().len() != ids.len())
        || (metadatas.is_some() && metadatas.as_ref().unwrap().len() != ids.len())
        || (documents.is_some() && documents.as_ref().unwrap().len() != ids.len())
    {
        bail!("IDs, embeddings, metadatas, and documents must all be the same length",);
    }

    let unique_ids: HashSet<_> = ids.iter().collect();
    if unique_ids.len() != ids.len() {
        let duplicate_ids: Vec<_> = ids
            .iter()
            .filter(|id| ids.iter().filter(|x| x == id).count() > 1)
            .collect();
        bail!(
            "Expected IDs to be unique, found duplicates for: {:?}",
            duplicate_ids
        );
    }
    Ok(CollectionEntries {
        ids,
        metadatas,
        documents,
        embeddings,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::{
        collection::{CollectionEntries, GetOptions, QueryOptions},
        commons::MetadataValue,
        embeddings::MockEmbeddingProvider,
        filters::*,
        ChromaClient,
    };

    const TEST_COLLECTION: &str = "21-recipies-for-octopus";

    #[tokio::test]
    async fn test_modify_collection() {
        let client = ChromaClient::new(Default::default());

        let collection = client
            .await
            .unwrap()
            .get_or_create_collection(TEST_COLLECTION, None, None)
            .await
            .unwrap();

        //Test for setting invalid collection name. Should fail.
        assert!(collection
            .modify(Some("new name for test collection"), None)
            .await
            .is_err());

        //Test for setting new metadata. Should pass.
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), MetadataValue::Str("test".to_string()));
        assert!(collection
            .modify(
                None,
                Some(&metadata)
            )
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_add_to_collection() {
        let client = ChromaClient::new(Default::default());

        let collection = client
            .await
            .unwrap()
            .get_or_create_collection(TEST_COLLECTION, None, None)
            .await
            .unwrap();

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test1"],
            metadatas: None,
            documents: None,
            embeddings: None,
        };

        let response = collection.add(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_err(),
            "Embeddings and documents cannot both be None"
        );

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.add(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_err(),
            "IDs, embeddings, metadatas, and documents must all be the same length"
        );

        let valid_collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.add(
            valid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_ok(),
            "IDs, embeddings, metadatas, and documents must all be the same length"
        );

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test1", ""],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.add(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(response.await.is_err(), "Empty IDs not allowed");

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test", "test"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
        };
        let response = collection.add(invalid_collection_entries, None);
        assert!(
            response.await.is_err(),
            "Expected IDs to be unique. Duplicates not allowed"
        );

        let collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.add(collection_entries, None);
        assert!(
            response.await.is_err(),
            "embedding_function cannot be None if documents are provided and embeddings are None"
        );

        let collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider)));
        assert!(
            response.await.is_ok(),
            "Embeddings are computed by the embedding_function if embeddings are None and documents are provided"
        );
    }

    #[tokio::test]
    async fn test_upsert_collection() {
        let client = ChromaClient::new(Default::default());

        let collection = client
            .await
            .unwrap()
            .get_or_create_collection(TEST_COLLECTION, None, None)
            .await
            .unwrap();

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test1"],
            metadatas: None,
            documents: None,
            embeddings: None,
        };

        let response = collection.upsert(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_err(),
            "Embeddings and documents cannot both be None"
        );

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.upsert(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_err(),
            "IDs, embeddings, metadatas, and documents must all be the same length"
        );

        let valid_collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.upsert(
            valid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_ok(),
            "IDs, embeddings, metadatas, and documents must all be the same length"
        );

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test1", ""],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.upsert(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(response.await.is_err(), "Empty IDs not allowed");

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test", "test"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
        };
        let response = collection.upsert(invalid_collection_entries, None);
        assert!(
            response.await.is_err(),
            "Expected IDs to be unique. Duplicates not allowed"
        );

        let collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.upsert(collection_entries, None);
        assert!(
            response.await.is_err(),
            "embedding_function cannot be None if documents are provided and embeddings are None"
        );

        let collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.upsert(collection_entries, Some(Box::new(MockEmbeddingProvider)));
        assert!(
            response.await.is_ok(),
            "Embeddings are computed by the embedding_function if embeddings are None and documents are provided"
        );
    }

    #[tokio::test]
    async fn test_get_all_embeddings_from_collection() {
        let client = ChromaClient::new(Default::default());

        let collection = client
            .await
            .unwrap()
            .get_or_create_collection(TEST_COLLECTION, None, None)
            .await
            .unwrap();

        let get_all_query = GetOptions {
            ids: vec![],
            where_metadata: None,
            limit: None,
            offset: None,
            where_document: None,
            include: None,
        };
        let get_all_result = collection.get(get_all_query).await.unwrap();

        assert_eq!(get_all_result.ids.len(), collection.count().await.unwrap());
    }

    #[tokio::test]
    async fn test_update_collection() {
        let client = ChromaClient::new(Default::default());

        let collection = client
            .await
            .unwrap()
            .get_or_create_collection(TEST_COLLECTION, None, None)
            .await
            .unwrap();

        let valid_collection_entries = CollectionEntries {
            ids: vec!["test1"],
            metadatas: None,
            documents: None,
            embeddings: None,
        };

        let response = collection
            .update(
                valid_collection_entries,
                Some(Box::new(MockEmbeddingProvider)),
            )
            .await;


        assert!(
            response.is_ok(),
            "Embeddings and documents can both be None"
        );

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.update(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_err(),
            "IDs, embeddings, metadatas, and documents must all be the same length"
        );

        let valid_collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.update(
            valid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(
            response.await.is_ok(),
            "IDs, embeddings, metadatas, and documents must all be the same length"
        );

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test1", ""],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.update(
            invalid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(response.await.is_err(), "Empty IDs not allowed");

        let invalid_collection_entries = CollectionEntries {
            ids: vec!["test", "test"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: Some(vec![vec![1.0, 2.0], vec![3.0, 4.0]]),
        };
        let response = collection.update(invalid_collection_entries, None);
        assert!(
            response.await.is_err(),
            "Expected IDs to be unique. Duplicates not allowed"
        );

        let collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.update(collection_entries, None);
        assert!(
            response.await.is_err(),
            "embedding_function cannot be None if documents are provided and embeddings are None"
        );

        let collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Document content 1", "Document content 2"]),
            embeddings: None,
        };
        let response = collection.update(collection_entries, Some(Box::new(MockEmbeddingProvider)));
        assert!(
            response.await.is_ok(),
            "Embeddings are computed by the embedding_function if embeddings are None and documents are provided"
        );
    }

    #[tokio::test]
    async fn test_query_collection() {
        let client = ChromaClient::new(Default::default());

        let collection = client
            .await
            .unwrap()
            .get_or_create_collection(TEST_COLLECTION, None, None)
            .await
            .unwrap();
        assert!(collection.count().await.is_ok());

        let query = QueryOptions {
            query_texts: None,
            query_embeddings: None,
            where_metadata: None,
            where_document: None,
            n_results: None,
            include: None,
        };
        let query_result = collection.query(query, None);
        assert!(
            query_result.await.is_err(),
            "query_texts and query_embeddings cannot both be None"
        );

        let query = QueryOptions {
            query_texts: Some(vec![
                "Writing tests help me find bugs",
                "Running them does not",
            ]),
            query_embeddings: None,
            where_metadata: None,
            where_document: None,
            n_results: None,
            include: None,
        };
        let query_result = collection.query(query, Some(Box::new(MockEmbeddingProvider)));
        assert!(
            query_result.await.is_ok(),
            "query_embeddings will be computed from query_texts if embedding_function is provided"
        );

        let query = QueryOptions {
            query_texts: Some(vec![
                "Writing tests help me find bugs",
                "Running them does not",
            ]),
            query_embeddings: Some(vec![vec![0.0_f32; 768], vec![0.0_f32; 768]]),
            where_metadata: None,
            where_document: None,
            n_results: None,
            include: None,
        };
        let query_result = collection.query(query, Some(Box::new(MockEmbeddingProvider)));
        assert!(
            query_result.await.is_err(),
            "Both query_embeddings and query_texts cannot be provided"
        );

        let query = QueryOptions {
            query_texts: None,
            query_embeddings: Some(vec![vec![0.0_f32; 768], vec![0.0_f32; 768]]),
            where_metadata: None,
            where_document: None,
            n_results: None,
            include: None,
        };
        let query_result = collection.query(query, None);
        assert!(
            query_result.await.is_ok(),
            "Use provided query_embeddings if embedding_function is None"
        );
    }

    #[tokio::test]
    async fn test_delete_from_collection() {
        let client = ChromaClient::new(Default::default());

        let collection = client
            .await
            .unwrap()
            .get_or_create_collection(TEST_COLLECTION, None, None)
            .await
            .unwrap();

        let valid_collection_entries = CollectionEntries {
            ids: vec!["123ABC"],
            metadatas: None,
            documents: Some(vec!["Document content 1"]),
            embeddings: None,
        };

        let response = collection.add(
            valid_collection_entries,
            Some(Box::new(MockEmbeddingProvider)),
        );
        assert!(response.await.is_ok());

        let response = collection.delete(Some(vec!["123ABC"]), None, None).await;

        assert!(response.is_ok(),);
    }

    #[tokio::test]
    async fn test_metadata_filtering_with_get() {
        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-metadata-filtering-unique";
        let collection = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Add test data with metadata
        let collection_entries = CollectionEntries {
            ids: vec!["doc1", "doc2", "doc3", "doc4"],
            metadatas: Some(vec![
                HashMap::from([("category".to_string(), "technology".into()), ("score".to_string(), 85.into())]),
                HashMap::from([("category".to_string(), "science".into()), ("score".to_string(), 92.into())]),
                HashMap::from([("category".to_string(), "technology".into()), ("score".to_string(), 78.into())]),
                HashMap::from([("category".to_string(), "medicine".into()), ("score".to_string(), 95.into())]),
            ]),
            documents: Some(vec![
                "This is about artificial intelligence",
                "Quantum physics research paper",
                "Machine learning algorithms",
                "Medical breakthrough in cancer research"
            ]),
            embeddings: None,
        };

        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        // Test equality filter
        let filter = eq("category", "technology");
        let get_options = GetOptions::new().where_metadata(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc1".to_string()));
        assert!(result.ids.contains(&"doc3".to_string()));

        // Test greater than filter
        let filter = gt("score", 90);
        let get_options = GetOptions::new().where_metadata(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc2".to_string()));
        assert!(result.ids.contains(&"doc4".to_string()));

        // Test range filter using $and (ChromaDB requirement)
        let filter = and_metadata(vec![
            gte("score", 80),
            lt("score", 95)
        ]);
        let get_options = GetOptions::new().where_metadata(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc1".to_string()));
        assert!(result.ids.contains(&"doc2".to_string()));

        // Test in_list filter
        let filter = in_list("category", vec!["science", "medicine"]);
        let get_options = GetOptions::new().where_metadata(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc2".to_string()));
        assert!(result.ids.contains(&"doc4".to_string()));

        // Test AND filter
        let filter = and_metadata(vec![
            eq("category", "technology"),
            gt("score", 80)
        ]);
        let get_options = GetOptions::new().where_metadata(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 1);
        assert!(result.ids.contains(&"doc1".to_string()));

        // Test OR filter
        let filter = or_metadata(vec![
            eq("category", "science"),
            gt("score", 90)
        ]);
        let get_options = GetOptions::new().where_metadata(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc2".to_string()));
        assert!(result.ids.contains(&"doc4".to_string()));

        // Clean up: delete the collection
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_document_filtering_with_get() {
        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-document-filtering-unique";
        let collection = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Add test data
        let collection_entries = CollectionEntries {
            ids: vec!["doc1", "doc2", "doc3", "doc4"],
            metadatas: None,
            documents: Some(vec![
                "Machine learning and artificial intelligence",
                "Deep learning neural networks research",
                "Quantum computing breakthrough",
                "Medical AI applications in healthcare"
            ]),
            embeddings: None,
        };

        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        // Test contains filter
        let filter = contains("learning");
        let get_options = GetOptions::new().where_document(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc1".to_string()));
        assert!(result.ids.contains(&"doc2".to_string()));

        // Test not_contains filter
        let filter = not_contains("Quantum");
        let get_options = GetOptions::new().where_document(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 3);
        assert!(!result.ids.contains(&"doc3".to_string()));

        // Test AND filter for documents
        let filter = and_doc(vec![
            contains("AI"),
            contains("applications")
        ]);
        let get_options = GetOptions::new().where_document(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 1);
        assert!(result.ids.contains(&"doc4".to_string()));

        // Test OR filter for documents
        let filter = or_doc(vec![
            contains("Quantum"),
            contains("neural")
        ]);
        let get_options = GetOptions::new().where_document(filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc2".to_string()));
        assert!(result.ids.contains(&"doc3".to_string()));

        // Clean up: delete the collection
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_combined_metadata_and_document_filtering() {
        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-combined-filtering-unique";
        let collection = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Add test data with both metadata and documents
        let collection_entries = CollectionEntries {
            ids: vec!["doc1", "doc2", "doc3", "doc4"],
            metadatas: Some(vec![
                HashMap::from([("category".to_string(), "AI".into()), ("year".to_string(), 2023.into())]),
                HashMap::from([("category".to_string(), "quantum".into()), ("year".to_string(), 2022.into())]),
                HashMap::from([("category".to_string(), "AI".into()), ("year".to_string(), 2024.into())]),
                HashMap::from([("category".to_string(), "medicine".into()), ("year".to_string(), 2023.into())]),
            ]),
            documents: Some(vec![
                "Machine learning research paper",
                "Quantum computing breakthrough",
                "Advanced AI neural networks",
                "Medical AI diagnostic tools"
            ]),
            embeddings: None,
        };

        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        // Test combined metadata and document filters
        let metadata_filter = eq("category", "AI");
        let document_filter = contains("neural");
        
        let get_options = GetOptions::new()
            .where_metadata(metadata_filter)
            .where_document(document_filter);
        
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 1);
        assert!(result.ids.contains(&"doc3".to_string()));

        // Test another combination
        let metadata_filter = gte("year", 2023);
        let document_filter = contains("AI");
        
        let get_options = GetOptions::new()
            .where_metadata(metadata_filter)
            .where_document(document_filter);
        
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2);
        assert!(result.ids.contains(&"doc3".to_string()));
        assert!(result.ids.contains(&"doc4".to_string()));

        // Clean up: delete the collection
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_query_with_metadata_filtering() {
        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-query-metadata-unique";
        let collection = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Add test data
        let collection_entries = CollectionEntries {
            ids: vec!["doc1", "doc2", "doc3"],
            metadatas: Some(vec![
                HashMap::from([("priority".to_string(), "high".into()), ("status".to_string(), "active".into())]),
                HashMap::from([("priority".to_string(), "low".into()), ("status".to_string(), "inactive".into())]),
                HashMap::from([("priority".to_string(), "high".into()), ("status".to_string(), "inactive".into())]),
            ]),
            documents: Some(vec![
                "Important machine learning project",
                "Basic data analysis task",
                "Critical AI research paper"
            ]),
            embeddings: None,
        };

        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        // Test query with metadata filter
        let metadata_filter = eq("priority", "high");
        let query_options = QueryOptions::new()
            .query_texts(vec!["machine learning"])
            .where_metadata(metadata_filter)
            .n_results(10);

        let result = collection.query(query_options, Some(Box::new(MockEmbeddingProvider))).await.unwrap();
        assert_eq!(result.ids[0].len(), 2); // Should return 2 high priority docs
        assert!(result.ids[0].contains(&"doc1".to_string()));
        assert!(result.ids[0].contains(&"doc3".to_string()));

        // Test query with AND metadata filter
        let metadata_filter = and_metadata(vec![
            eq("priority", "high"),
            eq("status", "active")
        ]);
        let query_options = QueryOptions::new()
            .query_texts(vec!["AI research"])
            .where_metadata(metadata_filter)
            .n_results(10);

        let result = collection.query(query_options, Some(Box::new(MockEmbeddingProvider))).await.unwrap();
        assert_eq!(result.ids[0].len(), 1); // Should return only doc1
        assert!(result.ids[0].contains(&"doc1".to_string()));

        // Clean up: delete the collection
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_query_with_document_filtering() {
        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-query-document-unique";
        let collection = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Add test data
        let collection_entries = CollectionEntries {
            ids: vec!["doc1", "doc2", "doc3"],
            metadatas: None,
            documents: Some(vec![
                "Machine learning algorithms for classification",
                "Deep learning neural network architectures",
                "Quantum machine learning applications"
            ]),
            embeddings: None,
        };

        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        // Test query with document filter
        let document_filter = contains("neural");
        let query_options = QueryOptions::new()
            .query_texts(vec!["deep learning"])
            .where_document(document_filter)
            .n_results(10);

        let result = collection.query(query_options, Some(Box::new(MockEmbeddingProvider))).await.unwrap();
        assert_eq!(result.ids[0].len(), 1); // Should return only doc2
        assert!(result.ids[0].contains(&"doc2".to_string()));

        // Test query with OR document filter
        let document_filter = or_doc(vec![
            contains("Quantum"),
            contains("classification")
        ]);
        let query_options = QueryOptions::new()
            .query_texts(vec!["machine learning"])
            .where_document(document_filter)
            .n_results(10);

        let result = collection.query(query_options, Some(Box::new(MockEmbeddingProvider))).await.unwrap();
        assert_eq!(result.ids[0].len(), 2); // Should return doc1 and doc3
        assert!(result.ids[0].contains(&"doc1".to_string()));
        assert!(result.ids[0].contains(&"doc3".to_string()));

        // Clean up: delete the collection
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_delete_with_filtering() {
        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-delete-filtering-unique";
        let collection = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Add test data
        let collection_entries = CollectionEntries {
            ids: vec!["doc1", "doc2", "doc3", "doc4"],
            metadatas: Some(vec![
                HashMap::from([("status".to_string(), "draft".into()), ("version".to_string(), 1.into())]),
                HashMap::from([("status".to_string(), "published".into()), ("version".to_string(), 2.into())]),
                HashMap::from([("status".to_string(), "draft".into()), ("version".to_string(), 1.into())]),
                HashMap::from([("status".to_string(), "archived".into()), ("version".to_string(), 3.into())]),
            ]),
            documents: Some(vec![
                "Draft document about AI",
                "Published research paper",
                "Another draft document",
                "Archived old content"
            ]),
            embeddings: None,
        };

        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        // Test delete with metadata filter
        let metadata_filter = eq("status", "draft");
        collection.delete(None, Some(metadata_filter), None).await.unwrap();

        // Verify deletion
        let remaining = collection.get(GetOptions::new()).await.unwrap();
        assert_eq!(remaining.ids.len(), 2);
        assert!(remaining.ids.contains(&"doc2".to_string()));
        assert!(remaining.ids.contains(&"doc4".to_string()));

        // Test delete with document filter
        let document_filter = contains("Archived");
        collection.delete(None, None, Some(document_filter)).await.unwrap();

        // Verify deletion
        let remaining = collection.get(GetOptions::new()).await.unwrap();
        assert_eq!(remaining.ids.len(), 1);
        assert!(remaining.ids.contains(&"doc2".to_string()));

        // Test delete with combined filters
        let collection_entries = CollectionEntries {
            ids: vec!["new1", "new2"],
            metadatas: Some(vec![
                HashMap::from([("type".to_string(), "temp".into())]),
                HashMap::from([("type".to_string(), "temp".into())]),
            ]),
            documents: Some(vec![
                "Temporary test document",
                "Another temp file"
            ]),
            embeddings: None,
        };
        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        let metadata_filter = eq("type", "temp");
        let document_filter = contains("test");
        collection.delete(None, Some(metadata_filter), Some(document_filter)).await.unwrap();

        // Verify only new1 was deleted (matches both filters)
        let remaining = collection.get(GetOptions::new()).await.unwrap();
        assert_eq!(remaining.ids.len(), 2);
        assert!(remaining.ids.contains(&"doc2".to_string()));
        assert!(remaining.ids.contains(&"new2".to_string()));

        // Clean up: delete the collection
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_complex_filter_combinations() {
        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-complex-filters-unique";
        let collection = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Add test data with complex metadata (removed array values since ChromaDB doesn't support them)
        let collection_entries = CollectionEntries {
            ids: vec!["doc1", "doc2", "doc3", "doc4", "doc5"],
            metadatas: Some(vec![
                HashMap::from([("category".to_string(), "AI".into()), ("score".to_string(), 95.into()), ("published".to_string(), true.into()), ("type".to_string(), "ml".into())]),
                HashMap::from([("category".to_string(), "quantum".into()), ("score".to_string(), 91.into()), ("published".to_string(), false.into()), ("type".to_string(), "physics".into())]),
                HashMap::from([("category".to_string(), "AI".into()), ("score".to_string(), 92.into()), ("published".to_string(), true.into()), ("type".to_string(), "cv".into())]),
                HashMap::from([("category".to_string(), "blockchain".into()), ("score".to_string(), 75.into()), ("published".to_string(), true.into()), ("type".to_string(), "crypto".into())]),
                HashMap::from([("category".to_string(), "AI".into()), ("score".to_string(), 89.into()), ("published".to_string(), false.into()), ("type".to_string(), "nlp".into())]),
            ]),
            documents: Some(vec![
                "Advanced natural language processing with transformers",
                "Quantum supremacy in computational complexity",
                "Computer vision using deep neural networks",
                "Cryptocurrency and blockchain technology overview",
                "Multi-modal AI combining vision and language"
            ]),
            embeddings: None,
        };

        collection.add(collection_entries, Some(Box::new(MockEmbeddingProvider))).await.unwrap();

        // Test complex nested filter: (category=AI AND published=true) OR score>90
        let complex_filter = or_metadata(vec![
            and_metadata(vec![
                eq("category", "AI"),
                eq("published", true)
            ]),
            gt("score", 90)
        ]);

        let get_options = GetOptions::new().where_metadata(complex_filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 3);
        assert!(result.ids.contains(&"doc1".to_string()));
        assert!(result.ids.contains(&"doc2".to_string()));
        assert!(result.ids.contains(&"doc3".to_string()));

        // Test range with AND: score >= 85 AND score <= 95 AND published = true
        let range_filter = and_metadata(vec![
            gte("score", 85),
            lte("score", 95),
            eq("published", true)
        ]);
        let get_options = GetOptions::new().where_metadata(range_filter);
        let result = collection.get(get_options).await.unwrap();
        assert_eq!(result.ids.len(), 2); // doc1 and doc3
        assert!(result.ids.contains(&"doc1".to_string()));
        assert!(result.ids.contains(&"doc3".to_string()));

        // Clean up: delete the collection
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_create_collection_with_hnsw_configuration() {
        use crate::config::{CreateCollectionConfiguration, CreateHNSWConfiguration, Space};

        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-hnsw-config-collection";

        // Create HNSW configuration
        let hnsw_config = CreateHNSWConfiguration::new()
            .space(Space::Cosine)
            .ef_construction(200)
            .max_neighbors(16)
            .ef_search(100)
            .sync_threshold(1000)
            .resize_factor(1.2);

        let config = CreateCollectionConfiguration::new().hnsw(hnsw_config);

        // Create collection with HNSW configuration
        let collection = client
            .get_or_create_collection(collection_name, None, Some(config))
            .await
            .unwrap();

        // Verify the configuration was set
        let collection_config = collection.configuration_json();
        assert!(collection_config.contains_key("hnsw"));
        
        if let Some(hnsw_value) = collection_config.get("hnsw") {
            if let Some(hnsw_obj) = hnsw_value.as_object() {
                assert_eq!(hnsw_obj.get("space").unwrap().as_str(), Some("cosine"));
                assert_eq!(hnsw_obj.get("ef_construction").unwrap().as_i64(), Some(200));
                assert_eq!(hnsw_obj.get("max_neighbors").unwrap().as_i64(), Some(16));
                assert_eq!(hnsw_obj.get("ef_search").unwrap().as_i64(), Some(100));
                assert_eq!(hnsw_obj.get("sync_threshold").unwrap().as_i64(), Some(1000));
                assert_eq!(hnsw_obj.get("resize_factor").unwrap().as_f64(), Some(1.2));
            }
        }

        // Test that we can add data to the configured collection
        let collection_entries = CollectionEntries {
            ids: vec!["test1", "test2"],
            metadatas: None,
            documents: Some(vec!["Sample document 1", "Sample document 2"]),
            embeddings: None,
        };

        let result = collection
            .add(collection_entries, Some(Box::new(MockEmbeddingProvider)))
            .await;
        assert!(result.is_ok());

        // Clean up
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_get_or_create_collection_with_configuration() {
        use crate::config::{CreateCollectionConfiguration, CreateHNSWConfiguration, Space};

        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-get-or-create-config";

        // Create HNSW configuration
        let hnsw_config = CreateHNSWConfiguration::new()
            .space(Space::L2)
            .ef_construction(128);

        let config = CreateCollectionConfiguration::new().hnsw(hnsw_config);

        // First call should create the collection with config
        let collection1 = client
            .get_or_create_collection(collection_name, None, Some(config.clone()))
            .await
            .unwrap();

        // Verify the configuration was set
        let collection_config = collection1.configuration_json();
        assert!(collection_config.contains_key("hnsw"));

        // Second call should return the existing collection (config should be ignored)
        let collection2 = client
            .get_or_create_collection(collection_name, None, None)
            .await
            .unwrap();

        // Both collections should have the same ID (same collection)
        assert_eq!(collection1.id(), collection2.id());

        // The existing collection should still have the original configuration
        let collection_config2 = collection2.configuration_json();
        assert!(collection_config2.contains_key("hnsw"));

        // Clean up
        client.delete_collection(collection_name).await.unwrap();
    }

    #[tokio::test]
    async fn test_configuration_serialization_skips_none_values() {
        use crate::config::{CreateCollectionConfiguration, CreateHNSWConfiguration, Space};

        let client = ChromaClient::new(Default::default()).await.unwrap();
        let collection_name = "test-minimal-config";

        // Create minimal HNSW configuration with only space set
        let hnsw_config = CreateHNSWConfiguration::new()
            .space(Space::Cosine);

        let config = CreateCollectionConfiguration::new().hnsw(hnsw_config);

        // Create collection with minimal configuration
        let collection = client
            .get_or_create_collection(collection_name, None, Some(config))
            .await
            .unwrap();

        // Verify only the set values are present in the configuration
        let collection_config = collection.configuration_json();
        assert!(collection_config.contains_key("hnsw"));
        
        if let Some(hnsw_value) = collection_config.get("hnsw") {
            if let Some(hnsw_obj) = hnsw_value.as_object() {
                assert!(hnsw_obj.contains_key("space"));
                assert_eq!(hnsw_obj.get("space").unwrap().as_str(), Some("cosine"));
                assert!(hnsw_obj.contains_key("ef_construction"));
                assert!(hnsw_obj.contains_key("max_neighbors"));
                assert!(hnsw_obj.contains_key("ef_search"));
                assert!(hnsw_obj.contains_key("sync_threshold"));
                assert!(hnsw_obj.contains_key("resize_factor"));
            }
        }

        println!("{:?}", collection.configuration_json());

        // Clean up
        client.delete_collection(collection_name).await.unwrap();
    }
}
