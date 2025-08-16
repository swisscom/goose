use super::api_client::{ApiClient, AuthMethod};
use super::errors::ProviderError;
use super::retry::ProviderRetry;
use super::utils::{get_model, handle_response_openai_compat};
use crate::conversation::message::Message;
use crate::impl_provider_default;
use crate::model::ModelConfig;
use crate::providers::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use anyhow::Result;
use async_trait::async_trait;
use rmcp::model::Tool;
use serde_json::Value;

// API endpoints and model configuration
pub const SWISS_AI_PLATFORM_DEFAULT_MODEL: &str = "meta/llama-3.3-70b-instruct";
pub const SWISS_AI_PLATFORM_KNOWN_MODELS: &[&str] = &[
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
];
pub const SWISS_AI_PLATFORM_DOC_URL: &str = "https://api.swisscom.com/layer/swiss-ai-platform";

// Base URLs for different models
const LLAMA_3_3_70B_BASE_URL: &str = "https://api.swisscom.com/layer/swiss-ai-platform/llama-3-3-70b/v1/";
const LLAMA_3_3_NEMOTRON_BASE_URL: &str = "https://api.swisscom.com/layer/swiss-ai-platform/llama-3.3-nemotron-super-49b/v1/";

/// URL mapping function for different models
/// Returns the appropriate base URL based on the model name
fn get_base_url_for_model(model_name: &str) -> &'static str {
    match model_name {
        "meta/llama-3.3-70b-instruct" => LLAMA_3_3_70B_BASE_URL,
        "nvidia/llama-3.3-nemotron-super-49b-v1" => LLAMA_3_3_NEMOTRON_BASE_URL,
        _ => LLAMA_3_3_70B_BASE_URL, // Default fallback to Llama 3.3 70B
    }
}

#[derive(Debug, serde::Serialize)]
pub struct SwissAiPlatformProvider {
    #[serde(skip)]
    api_client: ApiClient,
    model: ModelConfig,
}

impl_provider_default!(SwissAiPlatformProvider);

impl SwissAiPlatformProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        
        // Validate that the model is supported first
        if !SWISS_AI_PLATFORM_KNOWN_MODELS.contains(&model.model_name.as_str()) {
            return Err(anyhow::anyhow!(
                "Unknown model '{}'. Available Swiss AI Platform models: {}. \
                Please check the Swiss AI Platform documentation for supported models.",
                model.model_name,
                SWISS_AI_PLATFORM_KNOWN_MODELS.join(", ")
            ));
        }
        
        // Get API key with proper error handling for missing keys
        let api_key: String = config.get_secret("SWISS_AI_PLATFORM_API_KEY")
            .map_err(|_| anyhow::anyhow!(
                "SWISS_AI_PLATFORM_API_KEY environment variable is required. \
                Please set your Swiss AI Platform API key. \
                You can obtain an API key from the Swiss AI Platform portal."
            ))?;

        // Basic API key validation
        let trimmed_key = api_key.trim();
        if trimmed_key.is_empty() {
            return Err(anyhow::anyhow!(
                "SWISS_AI_PLATFORM_API_KEY cannot be empty or contain only whitespace. \
                Please provide a valid API key."
            ));
        }

        if trimmed_key.len() < 10 {
            return Err(anyhow::anyhow!(
                "SWISS_AI_PLATFORM_API_KEY appears to be too short (less than 10 characters). \
                Please verify your API key is correct."
            ));
        }

        // Check for common placeholder values
        let lowercase_key = trimmed_key.to_lowercase();
        if lowercase_key.contains("your_api_key") 
            || lowercase_key.contains("placeholder") 
            || lowercase_key.contains("example") 
            || lowercase_key == "sk-" 
            || lowercase_key == "test" {
            return Err(anyhow::anyhow!(
                "SWISS_AI_PLATFORM_API_KEY appears to be a placeholder value. \
                Please set your actual Swiss AI Platform API key."
            ));
        }
        
        // Determine base URL based on model, allow override with SWISS_AI_PLATFORM_HOST
        let host: String = if let Ok(custom_host) = config.get_param::<String>("SWISS_AI_PLATFORM_HOST") {
            let trimmed_host = custom_host.trim();
            if trimmed_host.is_empty() {
                return Err(anyhow::anyhow!(
                    "SWISS_AI_PLATFORM_HOST cannot be empty. \
                    Please provide a valid URL or remove the environment variable to use the default."
                ));
            }

            // Basic URL validation
            if !trimmed_host.starts_with("http://") && !trimmed_host.starts_with("https://") {
                return Err(anyhow::anyhow!(
                    "SWISS_AI_PLATFORM_HOST must be a valid URL starting with http:// or https://. \
                    Got: '{}'", trimmed_host
                ));
            }

            trimmed_host.to_string()
        } else {
            get_base_url_for_model(&model.model_name).to_string()
        };

        let auth = AuthMethod::BearerToken(trimmed_key.to_string());
        let api_client = ApiClient::new(host, auth)?;

        Ok(Self { api_client, model })
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        let response = self
            .api_client
            .response_post("chat/completions", &payload)
            .await
            .map_err(|e| {
                tracing::error!("Swiss AI Platform API request failed: {}", e);
                // Enhance error message with Swiss AI Platform specific context
                match e.to_string().as_str() {
                    s if s.contains("401") || s.contains("Unauthorized") => {
                        ProviderError::Authentication(
                            "Authentication failed with Swiss AI Platform. \
                            Please verify your SWISS_AI_PLATFORM_API_KEY is correct and active.".into()
                        )
                    },
                    s if s.contains("403") || s.contains("Forbidden") => {
                        ProviderError::Authentication(
                            "Access forbidden by Swiss AI Platform. \
                            Please check your API key permissions and account status.".into()
                        )
                    },
                    s if s.contains("429") || s.contains("rate limit") => {
                        ProviderError::RateLimitExceeded(
                            "Swiss AI Platform rate limit exceeded. Please wait before retrying.".into()
                        )
                    },
                    s if s.contains("timeout") => {
                        ProviderError::RequestFailed(
                            "Request to Swiss AI Platform timed out. Please try again.".into()
                        )
                    },
                    _ => e.into()
                }
            })?;

        handle_response_openai_compat(response).await
    }
}

#[async_trait]
impl Provider for SwissAiPlatformProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "swiss_ai_platform",
            "Swiss AI Platform",
            "Swiss AI Platform with Llama models",
            SWISS_AI_PLATFORM_DEFAULT_MODEL,
            SWISS_AI_PLATFORM_KNOWN_MODELS.to_vec(),
            SWISS_AI_PLATFORM_DOC_URL,
            vec![
                ConfigKey::new("SWISS_AI_PLATFORM_API_KEY", true, true, None),
                ConfigKey::new("SWISS_AI_PLATFORM_HOST", false, false, None),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        // Basic input validation
        if messages.is_empty() {
            return Err(ProviderError::UsageError(
                "Messages array cannot be empty. At least one message is required.".into()
            ));
        }

        let payload = create_request(
            &self.model,
            system,
            messages,
            tools,
            &super::utils::ImageFormat::OpenAi,
        ).map_err(|e| {
            tracing::error!("Failed to create request payload for Swiss AI Platform: {}", e);
            ProviderError::UsageError(format!(
                "Failed to create request for Swiss AI Platform: {}. \
                Please check your input parameters.", e
            ))
        })?;

        let response = self.with_retry(|| self.post(payload.clone())).await
            .map_err(|e| {
                tracing::error!("Swiss AI Platform completion request failed: {}", e);
                e
            })?;

        let message = response_to_message(&response)
            .map_err(|e| {
                tracing::error!("Failed to parse Swiss AI Platform response: {}", e);
                ProviderError::UsageError(format!(
                    "Failed to parse response from Swiss AI Platform: {}. \
                    The API response format may have changed.", e
                ))
            })?;

        let usage = response.get("usage").map(get_usage).unwrap_or_else(|| {
            tracing::debug!("No usage data available from Swiss AI Platform response");
            Usage::default()
        });

        let model = get_model(&response);
        super::utils::emit_debug_trace(&self.model, &payload, &response, &usage);
        
        Ok((message, ProviderUsage::new(model, usage)))
    }

    /// Fetch supported models from Swiss AI Platform; returns Err on failure, Ok(None) if no models found
    async fn fetch_supported_models(&self) -> Result<Option<Vec<String>>, ProviderError> {
        tracing::debug!("Fetching supported models from Swiss AI Platform");

        let response = self
            .api_client
            .request("models")
            .header("Content-Type", "application/json")?
            .response_get()
            .await
            .map_err(|e| {
                tracing::error!("Failed to fetch models from Swiss AI Platform: {}", e);
                match e.to_string().as_str() {
                    s if s.contains("401") || s.contains("Unauthorized") => {
                        ProviderError::Authentication(
                            "Authentication failed when fetching models from Swiss AI Platform. \
                            Please verify your API key.".into()
                        )
                    },
                    s if s.contains("404") => {
                        ProviderError::UsageError(
                            "Models endpoint not found on Swiss AI Platform. \
                            The API may not support model listing.".into()
                        )
                    },
                    _ => ProviderError::RequestFailed(format!(
                        "Failed to fetch models from Swiss AI Platform: {}", e
                    ))
                }
            })?;

        let response = handle_response_openai_compat(response).await
            .map_err(|e| {
                tracing::error!("Failed to parse models response from Swiss AI Platform: {}", e);
                ProviderError::UsageError(format!(
                    "Invalid response format from Swiss AI Platform models endpoint: {}", e
                ))
            })?;

        let data = response
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                tracing::error!("Swiss AI Platform models response missing 'data' field");
                ProviderError::UsageError(
                    "Invalid models response from Swiss AI Platform: missing or invalid 'data' field".into()
                )
            })?;

        if data.is_empty() {
            tracing::warn!("Swiss AI Platform returned empty models list");
            return Ok(Some(vec![]));
        }

        let mut model_names: Vec<String> = data
            .iter()
            .filter_map(|m| {
                m.get("id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.trim().is_empty())
                    .map(String::from)
            })
            .collect();

        if model_names.is_empty() {
            tracing::warn!("No valid model IDs found in Swiss AI Platform response");
            return Ok(Some(vec![]));
        }

        model_names.sort();
        model_names.dedup(); // Remove any duplicates

        tracing::debug!("Successfully fetched {} models from Swiss AI Platform", model_names.len());
        Ok(Some(model_names))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelConfig;

    #[test]
    fn test_url_mapping() {
        assert_eq!(
            get_base_url_for_model("meta/llama-3.3-70b-instruct"),
            LLAMA_3_3_70B_BASE_URL
        );
        assert_eq!(
            get_base_url_for_model("nvidia/llama-3.3-nemotron-super-49b-v1"),
            LLAMA_3_3_NEMOTRON_BASE_URL
        );
        // Test default fallback
        assert_eq!(
            get_base_url_for_model("unknown-model"),
            LLAMA_3_3_70B_BASE_URL
        );
    }

    #[test]
    fn test_known_models_contains_default() {
        assert!(SWISS_AI_PLATFORM_KNOWN_MODELS.contains(&SWISS_AI_PLATFORM_DEFAULT_MODEL));
    }

    #[test]
    fn test_metadata() {
        let metadata = SwissAiPlatformProvider::metadata();
        assert_eq!(metadata.name, "swiss_ai_platform");
        assert_eq!(metadata.display_name, "Swiss AI Platform");
        assert_eq!(metadata.default_model, SWISS_AI_PLATFORM_DEFAULT_MODEL);
        assert_eq!(metadata.known_models.len(), SWISS_AI_PLATFORM_KNOWN_MODELS.len());
        
        // Check that required config keys are present
        let config_keys: Vec<&str> = metadata.config_keys.iter().map(|k| k.name.as_str()).collect();
        assert!(config_keys.contains(&"SWISS_AI_PLATFORM_API_KEY"));
        assert!(config_keys.contains(&"SWISS_AI_PLATFORM_HOST"));
    }

    #[test]
    fn test_model_validation() {
        // Clean up any existing env vars
        std::env::remove_var("SWISS_AI_PLATFORM_API_KEY");
        std::env::remove_var("SWISS_AI_PLATFORM_HOST");
        
        // Test with valid model
        let valid_model = ModelConfig {
            model_name: "meta/llama-3.3-70b-instruct".to_string(),
            context_limit: None,
            temperature: None,
            max_tokens: None,
            toolshim: false,
            toolshim_model: None,
        };
        
        // This will fail due to missing API key, but should not fail due to invalid model
        let result = SwissAiPlatformProvider::from_env(valid_model);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("SWISS_AI_PLATFORM_API_KEY"));
        
        // Test with invalid model
        let invalid_model = ModelConfig {
            model_name: "invalid-model".to_string(),
            context_limit: None,
            temperature: None,
            max_tokens: None,
            toolshim: false,
            toolshim_model: None,
        };
        
        let result = SwissAiPlatformProvider::from_env(invalid_model);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Unknown model"));
        assert!(error_msg.contains("invalid-model"));
    }

    #[test]
    fn test_api_key_validation() {
        // Clean up environment first
        std::env::remove_var("SWISS_AI_PLATFORM_API_KEY");
        std::env::remove_var("SWISS_AI_PLATFORM_HOST");

        // Test empty API key
        std::env::set_var("SWISS_AI_PLATFORM_API_KEY", "");
        let model = ModelConfig {
            model_name: "meta/llama-3.3-70b-instruct".to_string(),
            context_limit: None,
            temperature: None,
            max_tokens: None,
            toolshim: false,
            toolshim_model: None,
        };
        let result = SwissAiPlatformProvider::from_env(model);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        // Should fail due to empty API key OR missing API key
        assert!(error_msg.contains("cannot be empty") || error_msg.contains("required"));

        // Test too short API key
        std::env::set_var("SWISS_AI_PLATFORM_API_KEY", "short");
        let model = ModelConfig {
            model_name: "meta/llama-3.3-70b-instruct".to_string(),
            context_limit: None,
            temperature: None,
            max_tokens: None,
            toolshim: false,
            toolshim_model: None,
        };
        let result = SwissAiPlatformProvider::from_env(model);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too short"));

        // Test placeholder API key
        std::env::set_var("SWISS_AI_PLATFORM_API_KEY", "your_api_key_here");
        let model = ModelConfig {
            model_name: "meta/llama-3.3-70b-instruct".to_string(),
            context_limit: None,
            temperature: None,
            max_tokens: None,
            toolshim: false,
            toolshim_model: None,
        };
        let result = SwissAiPlatformProvider::from_env(model);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("placeholder"));

        // Clean up
        std::env::remove_var("SWISS_AI_PLATFORM_API_KEY");
    }



    #[test]
    fn test_base_url_constants() {
        // Ensure the base URLs are properly defined
        assert!(!LLAMA_3_3_70B_BASE_URL.is_empty());
        assert!(!LLAMA_3_3_NEMOTRON_BASE_URL.is_empty());
        assert!(LLAMA_3_3_70B_BASE_URL.starts_with("https://"));
        assert!(LLAMA_3_3_NEMOTRON_BASE_URL.starts_with("https://"));
    }

    #[test]
    fn test_constants_consistency() {
        // Test that default model is in known models
        assert!(SWISS_AI_PLATFORM_KNOWN_MODELS.contains(&SWISS_AI_PLATFORM_DEFAULT_MODEL));

        // Test that all constants are properly defined
        assert!(!SWISS_AI_PLATFORM_DEFAULT_MODEL.is_empty());
        assert!(!SWISS_AI_PLATFORM_DOC_URL.is_empty());
        assert!(SWISS_AI_PLATFORM_DOC_URL.starts_with("https://"));
        assert!(SWISS_AI_PLATFORM_KNOWN_MODELS.len() > 0);

        // Test that known models are unique
        let mut unique_models = SWISS_AI_PLATFORM_KNOWN_MODELS.to_vec();
        unique_models.sort();
        unique_models.dedup();
        assert_eq!(unique_models.len(), SWISS_AI_PLATFORM_KNOWN_MODELS.len());
    }
}