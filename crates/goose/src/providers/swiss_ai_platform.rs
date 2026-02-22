use super::api_client::{ApiClient, AuthMethod};
use super::base::{
    ConfigKey, MessageStream, ModelInfo, Provider, ProviderDef, ProviderMetadata, ProviderUsage,
    Usage,
};
use super::errors::ProviderError;
use super::formats::openai::{create_request, get_usage, response_to_message};
use super::openai_compatible::{
    handle_response_openai_compat, handle_status_openai_compat, map_http_error_to_provider_error,
    stream_openai_compat,
};
use super::retry::ProviderRetry;
use super::utils::{get_model, ImageFormat, RequestLog};
use crate::conversation::message::Message;
use crate::model::ModelConfig;
use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use rmcp::model::Tool;
use serde_json::Value;

pub const SWISS_AI_PLATFORM_API_HOST: &str =
    "https://api.swisscom.com/layer/swiss-ai-platform/gpt-oss-120b/v1";
pub const SWISS_AI_PLATFORM_DEFAULT_MODEL: &str = "openai/gpt-oss-120b";
pub const SWISS_AI_PLATFORM_KNOWN_MODELS: &[(&str, usize)] = &[("openai/gpt-oss-120b", 128_000)];
const SWISS_AI_PLATFORM_PROVIDER_NAME: &str = "swiss-ai-platform";

pub const SWISS_AI_PLATFORM_DOC_URL: &str =
    "https://digital.swisscom.com/products/swiss-ai-platform/info";

#[derive(serde::Serialize, Debug)]
pub struct SwissAiPlatformProvider {
    #[serde(skip)]
    api_client: ApiClient,
    model: ModelConfig,
    #[serde(skip)]
    name: String,
}

impl SwissAiPlatformProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("SWISS_AI_PLATFORM_API_KEY")?;
        let host: String = config
            .get_param("SWISS_AI_PLATFORM_HOST")
            .unwrap_or_else(|_| SWISS_AI_PLATFORM_API_HOST.to_string());

        let auth = AuthMethod::BearerToken(api_key);
        let api_client = ApiClient::new(host, auth)?;

        Ok(Self {
            api_client,
            model,
            name: SWISS_AI_PLATFORM_PROVIDER_NAME.to_string(),
        })
    }

    fn error_from_swiss_payload(payload: Value) -> ProviderError {
        let code = payload
            .get("error")
            .and_then(|e| e.get("code"))
            .and_then(|c| c.as_u64())
            .unwrap_or(500) as u16;
        let status = reqwest::StatusCode::from_u16(code)
            .unwrap_or(reqwest::StatusCode::INTERNAL_SERVER_ERROR);
        map_http_error_to_provider_error(status, Some(payload))
    }

    async fn post(
        &self,
        session_id: Option<&str>,
        payload: &Value,
    ) -> Result<Value, ProviderError> {
        let response = self
            .api_client
            .response_post(session_id, "chat/completions", payload)
            .await?;
        let json = handle_response_openai_compat(response).await?;
        if json.get("error").is_some() {
            return Err(Self::error_from_swiss_payload(json));
        }
        Ok(json)
    }
}

impl ProviderDef for SwissAiPlatformProvider {
    type Provider = Self;

    fn metadata() -> ProviderMetadata {
        let models = SWISS_AI_PLATFORM_KNOWN_MODELS
            .iter()
            .map(|(name, limit)| ModelInfo::new(*name, *limit))
            .collect();

        ProviderMetadata::with_models(
            SWISS_AI_PLATFORM_PROVIDER_NAME,
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
                    true,
                ),
                ConfigKey::new("SWISS_AI_PLATFORM_API_KEY", true, true, None, true),
            ],
        )
    }

    fn from_env(
        model: ModelConfig,
        _extensions: Vec<crate::config::ExtensionConfig>,
    ) -> BoxFuture<'static, Result<Self::Provider>> {
        Box::pin(Self::from_env(model))
    }
}

#[async_trait]
impl Provider for SwissAiPlatformProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    async fn complete(
        &self,
        model_config: &ModelConfig,
        session_id: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            false,
        )?;

        let mut log = RequestLog::start(model_config, &payload)?;
        let json = self
            .with_retry(|| async {
                let payload_clone = payload.clone();
                self.post(Some(session_id), &payload_clone).await
            })
            .await
            .inspect_err(|e| {
                let _ = log.error(e);
            })?;

        let message = response_to_message(&json)?;
        let usage = json
            .get("usage")
            .map(get_usage)
            .unwrap_or_else(Usage::default);
        let model = get_model(&json);
        log.write(&json, Some(&usage))?;
        Ok((message, ProviderUsage::new(model, usage)))
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        session_id: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            true,
        )?;
        let mut log = RequestLog::start(model_config, &payload)?;
        let response = self
            .with_retry(|| async {
                let resp = self
                    .api_client
                    .response_post(Some(session_id), "chat/completions", &payload)
                    .await?;
                let resp = handle_status_openai_compat(resp).await?;

                let is_json = resp
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .map(|v| v.to_ascii_lowercase())
                    .is_some_and(|v| v.contains("json"));

                if is_json {
                    let body = handle_response_openai_compat(resp).await?;
                    if body.get("error").is_some() {
                        return Err(Self::error_from_swiss_payload(body));
                    }

                    return Err(ProviderError::ExecutionError(
                        "Expected streaming response but received non-streaming payload"
                            .to_string(),
                    ));
                }

                Ok(resp)
            })
            .await
            .inspect_err(|e| {
                let _ = log.error(e);
            })?;

        stream_openai_compat(response, log)
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        let response = self
            .api_client
            .response_get(None, "models")
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let json = handle_response_openai_compat(response).await?;

        if json.get("error").is_some() {
            return Err(Self::error_from_swiss_payload(json));
        }

        let data = json.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            ProviderError::RequestFailed("Missing 'data' array in models response".to_string())
        })?;

        let mut models: Vec<String> = data
            .iter()
            .filter_map(|model| model.get("id").and_then(|v| v.as_str()).map(str::to_string))
            .collect();
        models.sort();
        Ok(models)
    }
}
