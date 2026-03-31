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
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::future::BoxFuture;
use rmcp::model::Tool;
use serde_json::Value;

pub const SWISS_AI_PLATFORM_API_HOST: &str =
    "https://api.swisscom.com/layer/swiss-ai-platform/gpt-oss-120b/v1";
pub const SWISS_AI_PLATFORM_DEFAULT_MODEL: &str = "openai/gpt-oss-120b";
pub const SWISS_AI_PLATFORM_QWEN_BASE_URL: &str =
    "https://api.swisscom.com/products/swiss-ai-platform/qwen3.5-397b-a17b/v1";
pub const SWISS_AI_PLATFORM_GPT_OSS_BASE_URL: &str =
    "https://api.swisscom.com/layer/swiss-ai-platform/gpt-oss-120b/v1";
pub const SWISS_AI_PLATFORM_KNOWN_MODELS: &[(&str, usize)] = &[
    ("openai/gpt-oss-120b", 128_000),
    ("qwen/qwen3.5-397b-a17b", 128_000),
    ("meta/llama-4-scout-17b-16e-instruct", 128_000),
];
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
        let (host, api_key) = Self::resolve_connection_settings(&model.model_name)?;

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

    fn customize_payload(model_config: &ModelConfig, tools: &[Tool], payload: &mut Value) {
        if !tools.is_empty() {
            payload["tool_choice"] = Value::String("auto".to_string());
        }

        if let Some(params) = &model_config.request_params {
            if let Some(obj) = payload.as_object_mut() {
                for (key, value) in params {
                    obj.insert(key.clone(), value.clone());
                }
            }
        }
    }

    fn resolve_connection_settings(model_name: &str) -> Result<(String, String)> {
        let config = crate::config::Config::global();

        let mut base_url = None;
        let mut api_key = None;
        Self::overlay_model_specific_env_settings(&config, model_name, &mut base_url, &mut api_key);

        let base_url = base_url
            .ok_or_else(|| {
                anyhow!(
                    "Missing Swiss AI Platform base URL for model '{}'. Configure a model-specific base URL for this model.",
                    model_name
                )
            })?;
        let api_key = api_key
            .ok_or_else(|| {
                anyhow!(
                    "Missing Swiss AI Platform access token for model '{}'. Configure a model-specific access token for this model.",
                    model_name
                )
            })?;

        Self::validate_host_matches_model(&base_url, model_name)?;
        Ok((base_url, api_key))
    }

    fn configured_known_models() -> Vec<String> {
        SWISS_AI_PLATFORM_KNOWN_MODELS
            .iter()
            .filter_map(|(model_name, _)| {
                Self::resolve_connection_settings(model_name)
                    .ok()
                    .map(|_| (*model_name).to_string())
            })
            .collect()
    }

    fn overlay_model_specific_env_settings(
        config: &crate::config::Config,
        model_name: &str,
        base_url: &mut Option<String>,
        api_key: &mut Option<String>,
    ) {
        for prefix in Self::model_env_prefixes(model_name) {
            if let Ok(value) = config.get_param::<String>(&format!("{prefix}_BASE_URL")) {
                *base_url = Some(value);
            } else if let Ok(host) = config.get_param::<String>(&format!("{prefix}_HOST")) {
                *base_url = Some(host);
            }

            if let Ok(value) = config.get_secret::<String>(&format!("{prefix}_API_KEY")) {
                *api_key = Some(value);
            } else if let Ok(value) = config.get_secret::<String>(&format!("{prefix}_ACCESS_TOKEN"))
            {
                *api_key = Some(value);
            }
        }
    }

    fn model_key_candidates(model_name: &str) -> Vec<String> {
        let mut candidates = vec![model_name.to_ascii_lowercase()];

        if let Some(short_name) = model_name.rsplit('/').next() {
            let short_name = short_name.to_ascii_lowercase();
            if !candidates.contains(&short_name) {
                candidates.push(short_name);
            }
        }

        candidates
    }

    fn model_env_prefixes(model_name: &str) -> Vec<String> {
        let mut prefixes = Vec::new();

        for candidate in Self::model_key_candidates(model_name) {
            let normalized = candidate
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_uppercase()
                    } else {
                        '_'
                    }
                })
                .collect::<String>();

            let normalized = normalized
                .split('_')
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("_");

            let prefix = format!("SWISS_AI_PLATFORM_{normalized}");
            if !prefixes.contains(&prefix) {
                prefixes.push(prefix);
            }
        }

        prefixes
    }

    fn endpoint_model_slug(base_url: &str) -> Option<String> {
        let url = url::Url::parse(base_url).ok()?;
        let segments: Vec<_> = url
            .path_segments()?
            .filter(|segment| !segment.is_empty())
            .collect();

        let last = *segments.last()?;
        if !last.eq_ignore_ascii_case("v1") || segments.len() < 2 {
            return None;
        }

        let slug = segments[segments.len() - 2];
        if slug.eq_ignore_ascii_case("swiss-ai-platform") {
            return None;
        }

        Some(slug.to_ascii_lowercase())
    }

    fn validate_host_matches_model(base_url: &str, model_name: &str) -> Result<()> {
        let Some(endpoint_slug) = Self::endpoint_model_slug(base_url) else {
            return Ok(());
        };

        let normalized_model = model_name.to_ascii_lowercase();
        if normalized_model.contains(&endpoint_slug) {
            return Ok(());
        }

        Err(anyhow!(
            "Swiss AI Platform endpoint '{}' does not match model '{}'. Configure the endpoint URL and model as a pair, for example `.../{}/v1` with a model name containing `{}`.",
            base_url,
            model_name,
            endpoint_slug,
            endpoint_slug,
        ))
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
                    "SWISS_AI_PLATFORM_GPT_OSS_120B_BASE_URL",
                    true,
                    false,
                    Some(SWISS_AI_PLATFORM_GPT_OSS_BASE_URL),
                    true,
                ),
                ConfigKey::new(
                    "SWISS_AI_PLATFORM_GPT_OSS_120B_ACCESS_TOKEN",
                    false,
                    true,
                    None,
                    true,
                ),
                ConfigKey::new(
                    "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_BASE_URL",
                    true,
                    false,
                    Some(SWISS_AI_PLATFORM_QWEN_BASE_URL),
                    true,
                ),
                ConfigKey::new(
                    "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_ACCESS_TOKEN",
                    false,
                    true,
                    None,
                    true,
                ),
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
        let mut payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            false,
        )?;
        Self::customize_payload(model_config, tools, &mut payload);

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
        let mut payload = create_request(
            model_config,
            system,
            messages,
            tools,
            &ImageFormat::OpenAi,
            true,
        )?;
        Self::customize_payload(model_config, tools, &mut payload);
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

    async fn fetch_recommended_models(&self) -> Result<Vec<String>, ProviderError> {
        let configured_models = Self::configured_known_models();
        if !configured_models.is_empty() {
            return Ok(configured_models);
        }

        self.fetch_supported_models().await
    }
}

#[cfg(test)]
mod tests {
    use super::SwissAiPlatformProvider;
    use env_lock::lock_env;

    #[test]
    fn extracts_model_slug_from_swiss_endpoint() {
        let slug = SwissAiPlatformProvider::endpoint_model_slug(
            "https://api.swisscom.com/products/swiss-ai-platform/qwen3.5-397b-a17b/v1",
        );

        assert_eq!(slug.as_deref(), Some("qwen3.5-397b-a17b"));
    }

    #[test]
    fn accepts_matching_endpoint_and_model() {
        let result = SwissAiPlatformProvider::validate_host_matches_model(
            "https://api.swisscom.com/layer/swiss-ai-platform/llama-4-scout-17b-16e/v1",
            "meta/llama-4-scout-17b-16e-instruct",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn rejects_mismatched_endpoint_and_model() {
        let result = SwissAiPlatformProvider::validate_host_matches_model(
            "https://api.swisscom.com/products/swiss-ai-platform/qwen3.5-397b-a17b/v1",
            "openai/gpt-oss-120b",
        );

        assert!(result.is_err());
    }

    #[test]
    fn resolves_model_specific_env_settings() {
        let _guard = lock_env([
            (
                "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_API_KEY",
                Some("qwen-key"),
            ),
            (
                "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_BASE_URL",
                Some("https://api.swisscom.com/products/swiss-ai-platform/qwen3.5-397b-a17b/v1"),
            ),
        ]);

        let (base_url, api_key) =
            SwissAiPlatformProvider::resolve_connection_settings("qwen/qwen3.5-397b-a17b").unwrap();

        assert_eq!(
            base_url,
            "https://api.swisscom.com/products/swiss-ai-platform/qwen3.5-397b-a17b/v1"
        );
        assert_eq!(api_key, "qwen-key");
    }

    #[test]
    fn resolves_model_specific_access_token_settings() {
        let _guard = lock_env([
            (
                "SWISS_AI_PLATFORM_GPT_OSS_120B_ACCESS_TOKEN",
                Some("gpt-oss-token"),
            ),
            (
                "SWISS_AI_PLATFORM_GPT_OSS_120B_BASE_URL",
                Some("https://api.swisscom.com/layer/swiss-ai-platform/gpt-oss-120b/v1"),
            ),
        ]);

        let (base_url, api_key) =
            SwissAiPlatformProvider::resolve_connection_settings("openai/gpt-oss-120b").unwrap();

        assert_eq!(
            base_url,
            "https://api.swisscom.com/layer/swiss-ai-platform/gpt-oss-120b/v1"
        );
        assert_eq!(api_key, "gpt-oss-token");
    }

    #[test]
    fn customizes_payload_for_tools_and_request_params() {
        let mut request_params = std::collections::HashMap::new();
        request_params.insert("parallel_tool_calls".to_string(), Value::Bool(false));

        let model_config = crate::model::ModelConfig::new_or_fail("openai/gpt-oss-120b")
            .with_request_params(Some(request_params));
        let tool = Tool::new(
            "test_tool",
            "A test tool",
            rmcp::object!({
                "type": "object",
                "properties": {},
            }),
        );
        let mut payload = serde_json::json!({
            "model": "openai/gpt-oss-120b",
            "tools": []
        });

        SwissAiPlatformProvider::customize_payload(&model_config, &[tool], &mut payload);

        assert_eq!(payload["tool_choice"], "auto");
        assert_eq!(payload["parallel_tool_calls"], false);
    }

    #[test]
    fn metadata_orders_primary_keys_for_cli_setup() {
        let metadata = SwissAiPlatformProvider::metadata();
        let primary_keys: Vec<_> = metadata
            .config_keys
            .iter()
            .filter(|key| key.primary)
            .map(|key| key.name.as_str())
            .collect();

        assert_eq!(
            primary_keys,
            vec![
                "SWISS_AI_PLATFORM_GPT_OSS_120B_BASE_URL",
                "SWISS_AI_PLATFORM_GPT_OSS_120B_ACCESS_TOKEN",
                "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_BASE_URL",
                "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_ACCESS_TOKEN",
            ]
        );
    }

    #[test]
    fn metadata_marks_base_urls_as_required_defaults() {
        let metadata = SwissAiPlatformProvider::metadata();

        let gpt_oss_base_url = metadata
            .config_keys
            .iter()
            .find(|key| key.name == "SWISS_AI_PLATFORM_GPT_OSS_120B_BASE_URL")
            .unwrap();
        let qwen_base_url = metadata
            .config_keys
            .iter()
            .find(|key| key.name == "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_BASE_URL")
            .unwrap();

        assert!(gpt_oss_base_url.required);
        assert_eq!(
            gpt_oss_base_url.default.as_deref(),
            Some(SWISS_AI_PLATFORM_GPT_OSS_BASE_URL)
        );
        assert!(qwen_base_url.required);
        assert_eq!(
            qwen_base_url.default.as_deref(),
            Some(SWISS_AI_PLATFORM_QWEN_BASE_URL)
        );
    }

    #[test]
    fn returns_all_configured_known_models() {
        let _guard = lock_env([
            (
                "SWISS_AI_PLATFORM_GPT_OSS_120B_BASE_URL",
                Some("https://api.swisscom.com/layer/swiss-ai-platform/gpt-oss-120b/v1"),
            ),
            (
                "SWISS_AI_PLATFORM_GPT_OSS_120B_ACCESS_TOKEN",
                Some("gpt-oss-token"),
            ),
            (
                "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_BASE_URL",
                Some("https://api.swisscom.com/products/swiss-ai-platform/qwen3.5-397b-a17b/v1"),
            ),
            (
                "SWISS_AI_PLATFORM_QWEN3_5_397B_A17B_ACCESS_TOKEN",
                Some("qwen-token"),
            ),
        ]);

        let configured_models = SwissAiPlatformProvider::configured_known_models();

        assert_eq!(
            configured_models,
            vec![
                "openai/gpt-oss-120b".to_string(),
                "qwen/qwen3.5-397b-a17b".to_string(),
            ]
        );
    }
}
