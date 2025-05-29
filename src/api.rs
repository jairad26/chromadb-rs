use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use reqwest::{Client, Method, Response};
use serde_json::Value;

use super::commons::Result;

#[derive(Clone, Debug)]
pub enum ChromaTokenHeader {
    Authorization,
    XChromaToken,
}

#[derive(Clone, Debug)]
pub enum ChromaAuthMethod {
    None,
    TokenAuth {
        header: ChromaTokenHeader,
        token: String,
    },
}

impl Default for ChromaAuthMethod {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Default, Debug)]
pub(super) struct APIClientAsync {
    client_pool: Mutex<VecDeque<Arc<Client>>>,
    api_endpoint: String,
    auth_method: ChromaAuthMethod,
    tenant: String,
    database: String,
    await_connection: Condvar,
    connections_alloc: AtomicUsize,
    connections_total: AtomicUsize,
}

impl APIClientAsync {
    pub fn new(
        endpoint: String,
        auth_method: ChromaAuthMethod,
        tenant: String,
        database: String,
        connections: usize,
    ) -> Self {
        let client_pool = (0..128)
            .map(|_| Arc::new(Client::new()))
            .collect::<VecDeque<_>>();
        let client_pool = Mutex::new(client_pool);
        Self {
            client_pool,
            api_endpoint: format!("{}/api/v2", endpoint),
            auth_method,
            tenant,
            database,
            await_connection: Condvar::new(),
            connections_alloc: AtomicUsize::new(0),
            connections_total: AtomicUsize::new(connections),
        }
    }

    fn database_url(&self, path: &str) -> String {
        assert!(path.starts_with('/'));
        format!(
            "{}/tenants/{}/databases/{}{}",
            self.api_endpoint, self.tenant, self.database, path
        )
    }

    /// GET from a database-scoped path.
    pub async fn get_database(&self, path: &str) -> Result<Response> {
        let url = self.database_url(path);
        self.send_request(Method::GET, &url, None).await
    }

    /// POST to a database-scoped path.
    pub async fn post_database(&self, path: &str, json_body: Option<Value>) -> Result<Response> {
        let url = self.database_url(path);
        self.send_request(Method::POST, &url, json_body).await
    }

    /// PUT to a database-scoped path.
    pub async fn put_database(&self, path: &str, json_body: Option<Value>) -> Result<Response> {
        let url = self.database_url(path);
        self.send_request(Method::PUT, &url, json_body).await
    }

    /// DELETE to a database-scoped path.  This does not delete a database.
    pub async fn delete_database(&self, path: &str) -> Result<Response> {
        let url = self.database_url(path);
        self.send_request(Method::DELETE, &url, None).await
    }

    /// GET from a v1-scoped path.
    pub async fn get(&self, path: &str) -> Result<Response> {
        assert!(path.starts_with('/'));
        let url = format!("{}{}", self.api_endpoint, path);
        self.send_request(Method::GET, &url, None).await
    }

    async fn send_request(
        &self,
        method: Method,
        url: &str,
        json_body: Option<Value>,
    ) -> Result<Response> {
        let client = {
            // SAFETY(rescrv): Mutex poisioning.
            let mut pool = self.client_pool.lock().unwrap();
            loop {
                if let Some(client) = pool.pop_front() {
                    break client;
                }
                let alloc = self.connections_alloc.load(Ordering::Relaxed);
                // If we haven't allocated everything, and we successfully allocated one more,
                // break with a new client.
                if alloc < self.connections_total.load(Ordering::Relaxed)
                    && self
                        .connections_alloc
                        .compare_exchange(alloc, alloc + 1, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                {
                    break Arc::new(Client::new());
                }
                pool = self.await_connection.wait(pool).unwrap();
            }
        };
        let request = client.request(method, url);
        let res = Self::send_request_no_self(request, &self.auth_method, json_body).await;
        {
            // SAFETY(rescrv): Mutex poisioning.
            let mut pool = self.client_pool.lock().unwrap();
            pool.push_front(client);
            self.await_connection.notify_one();
        }
        res
    }

    async fn send_request_no_self(
        mut request: reqwest::RequestBuilder,
        auth_method: &ChromaAuthMethod,
        json_body: Option<Value>,
    ) -> Result<Response> {
        // Add auth headers if needed
        match &auth_method {
            ChromaAuthMethod::None => {}
            ChromaAuthMethod::TokenAuth { token, header } => match header {
                ChromaTokenHeader::Authorization => {
                    request = request.header("Authorization", format!("Bearer {token}"));
                }
                ChromaTokenHeader::XChromaToken => {
                    request = request.header("X-Chroma-Token", token);
                }
            },
        }

        // Add JSON body if present
        if let Some(body) = json_body {
            request = request
                .header("Content-Type", "application/json")
                .json(&body);
        }

        let response = request.send().await?;
        let status = response.status();

        if status.is_success() {
            Ok(response)
        } else {
            let error_text = response.text().await?;
            anyhow::bail!(
                "{} {}: {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown"),
                error_text
            )
        }
    }
}
