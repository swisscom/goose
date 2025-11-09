use super::api_client::{ApiClient, AuthMethod};
use super::errors::ProviderError;
use super::retry::ProviderRetry;
use super::utils::{
    get_model, handle_response_openai_compat, handle_status_openai_compat, RequestLog,
};
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use crate::providers::base::{
    ConfigKey, MessageStream, ModelInfo, Provider, ProviderMetadata, ProviderUsage, Usage,
};
use crate::providers::formats::openai::{
    create_request, get_usage, response_to_message, response_to_streaming_message,
};
use anyhow::Result;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::TryStreamExt;
use rmcp::model::Tool;
use serde_json::{json, Value};
use std::io;
use tokio::pin;
use tokio_stream::StreamExt;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

pub const SWISS_AI_PLATFORM_API_HOST: &str =
    "https://api.swisscom.com/layer/swiss-ai-platform/gpt-oss-120b/v1";
pub const SWISS_AI_PLATFORM_DEFAULT_MODEL: &str = "openai/gpt-oss-120b";
pub const SWISS_AI_PLATFORM_KNOWN_MODELS: &[(&str, usize)] = &[("openai/gpt-oss-120b", 128_000)];

pub const SWISS_AI_PLATFORM_DOC_URL: &str =
    "https://digital.swisscom.com/products/swiss-ai-platform/info";


/// Provider implementation for the Swiss AI Platform LLM.
///
/// This provider enables interaction with AI models hosted on the Swiss AI Platform.
/// The implementation is based on `tetrate.rs`. It handles authentication, request routing,
/// and response processing for the LLM API endpoints.
#[derive(serde::Serialize, Debug)]
pub struct SwissAiPlatformProvider {
    #[serde(skip)]
    api_client: ApiClient,
    name: String,
    model: ModelConfig,
    supports_streaming: bool,
}

impl SwissAiPlatformProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("SWISS_AI_PLATFORM_API_KEY")?;
        // API host for LLM endpoints
        let host: String = config
            .get_param("SWISS_AI_PLATFORM_HOST")
            .unwrap_or_else(|_| SWISS_AI_PLATFORM_API_HOST.to_string());

        let auth = AuthMethod::BearerToken(api_key);
        let api_client = ApiClient::new(host, auth)?;

        Ok(Self {
            api_client,
            name: "swiss-ai-platform".to_string(),
            model,
            supports_streaming: true,
        })
    }

    async fn post(&self, payload: &Value) -> Result<Value, ProviderError> {
        let response = self
            .api_client
            .response_post("chat/completions", payload)
            .await?;

        // For OpenAI-compatible models, parse the response body to JSON
        let response_body = handle_response_openai_compat(response)
            .await
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to parse response: {e}")))?;

        // Swiss AI Platform can return errors in 200 OK responses, so we have to check for errors explicitly
        if let Some(error_obj) = response_body.get("error") {
            // If there's an error object, extract the error message and code
            let error_message = error_obj
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown Swiss AI Platform error");

            let error_code = error_obj.get("code").and_then(|c| c.as_u64()).unwrap_or(0);

            // Check for context length errors in the error message
            if error_code == 400 && error_message.contains("maximum context length") {
                return Err(ProviderError::ContextLengthExceeded(
                    error_message.to_string(),
                ));
            }

            // Return appropriate error based on the error code
            match error_code {
                401 | 403 => return Err(ProviderError::Authentication(error_message.to_string())),
                429 => {
                    return Err(ProviderError::RateLimitExceeded {
                        details: error_message.to_string(),
                        retry_delay: None,
                    })
                }
                500 | 503 => return Err(ProviderError::ServerError(error_message.to_string())),
                _ => return Err(ProviderError::RequestFailed(error_message.to_string())),
            }
        }

        // No error detected, return the response body
        Ok(response_body)
    }
}

#[async_trait]
impl Provider for SwissAiPlatformProvider {
    fn metadata() -> ProviderMetadata {
        let models = SWISS_AI_PLATFORM_KNOWN_MODELS
            .iter()
            .map(|(name, limit)| ModelInfo::new(*name, *limit))
            .collect();
        ProviderMetadata::with_models(
            "swiss-ai-platform",
            "Swiss AI Platform",
            "AI models powered by Swisscom",
            SWISS_AI_PLATFORM_DEFAULT_MODEL,
            models,
            SWISS_AI_PLATFORM_DOC_URL,
            vec![
                ConfigKey::new(
                    "SWISS_AI_PLATFORM_HOST",
                    true,
                    false,
                    Some(SWISS_AI_PLATFORM_API_HOST),
                ),
                ConfigKey::new("SWISS_AI_PLATFORM_API_KEY", true, true, None),
            ],
        )
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, model_config, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete_with_model(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        // Create the base payload using the provided model_config
        let payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &super::utils::ImageFormat::OpenAi,
        )?;

        let mut log = RequestLog::start(&self.model, &payload)?;

        // Make request
        let response = self
            .with_retry(|| async {
                let payload_clone = payload.clone();
                self.post(&payload_clone).await
            })
            .await
            .inspect_err(|e| {
                let _ = log.error(e);
            })?;

        // Parse response
        let message = response_to_message(&response)?;
        let usage = response.get("usage").map(get_usage).unwrap_or_else(|| {
            tracing::debug!("Failed to get usage data");
            Usage::default()
        });
        let model = get_model(&response);
        log.write(&response, Some(&usage))?;
        Ok((message, ProviderUsage::new(model, usage)))
    }

    async fn stream(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let mut payload = create_request(
            &self.model,
            system,
            messages,
            tools,
            &super::utils::ImageFormat::OpenAi,
        )?;

        // Enable streaming
        payload["stream"] = json!(true);
        payload["stream_options"] = json!({
            "include_usage": true,
        });

        let response = self
            .api_client
            .response_post("chat/completions", &payload)
            .await?;

        let response = handle_status_openai_compat(response).await?;
        let stream = response.bytes_stream().map_err(io::Error::other);
        let mut log = RequestLog::start(&self.model, &payload)?;

        Ok(Box::pin(try_stream! {
            let stream_reader = StreamReader::new(stream);
            let framed = FramedRead::new(stream_reader, LinesCodec::new()).map_err(anyhow::Error::from);

            let message_stream = response_to_streaming_message(framed);
            pin!(message_stream);
            while let Some(message) = message_stream.next().await {
                let (message, usage) = message.map_err(|e| ProviderError::RequestFailed(format!("Stream decode error: {}", e)))?;
                log.write(&message, usage.as_ref().map(|f| f.usage).as_ref())?;
                yield (message, usage);
            }
        }))
    }

    /// Fetch supported models from Swiss AI Platform API
    async fn fetch_supported_models(&self) -> Result<Option<Vec<String>>, ProviderError> {
        // Use the existing api_client which already has authentication configured
        let response = match self.api_client.response_get("models").await {
            Ok(response) => response,
            Err(e) => {
                tracing::warn!("Failed to fetch models from Swiss AI Platform API: {}, falling back to manual model entry", e);
                return Ok(None);
            }
        };

        // Handle JSON parsing failures gracefully
        let json: serde_json::Value = match response.json().await {
            Ok(json) => json,
            Err(e) => {
                tracing::warn!("Failed to parse Swiss AI Platform API response as JSON: {}, falling back to manual model entry", e);
                return Ok(None);
            }
        };

        // Check for error in response
        if let Some(err_obj) = json.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            tracing::warn!("Swiss AI Platform API returned an error: {}", msg);
            return Ok(None);
        }

        // The response format from /models is expected to be OpenAI-compatible
        // It should have a "data" field with an array of model objects
        let data = match json.get("data").and_then(|v| v.as_array()) {
            Some(data) => data,
            None => {
                tracing::warn!("Missing data field in Swiss AI Platform API response, falling back to manual model entry");
                return Ok(None);
            }
        };

        let mut models: Vec<String> = data
            .iter()
            .filter_map(|model| {
                // Get the model ID
                let id = model.get("id").and_then(|v| v.as_str())?;
                Some(id.to_string())
            })
            .collect();

        // If no models were found, fall back to manual entry
        if models.is_empty() {
            tracing::warn!("No models found in Swiss AI Platform API response, falling back to manual model entry");
            return Ok(None);
        }

        models.sort();
        Ok(Some(models))
    }

    fn supports_streaming(&self) -> bool {
        self.supports_streaming
    }
}
