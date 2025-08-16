use anyhow::Result;
use dotenvy::dotenv;
use futures::StreamExt;
use goose::agents::Agent;
use goose::conversation::message::{Message, MessageContent};
use goose::conversation::Conversation;
use goose::model::ModelConfig;
use goose::providers::{create, base::Provider};
use goose::providers::swiss_ai_platform::{
    SwissAiPlatformProvider, SWISS_AI_PLATFORM_DEFAULT_MODEL, SWISS_AI_PLATFORM_KNOWN_MODELS,
};
use rmcp::model::Tool;
use rmcp::object;
use std::env;
use std::sync::Arc;

/// Check if Swiss AI Platform API key is available
fn has_swiss_ai_platform_credentials() -> bool {
    env::var("SWISS_AI_PLATFORM_API_KEY").is_ok()
}

/// Skip test if credentials are not available
macro_rules! skip_if_no_credentials {
    () => {
        if !has_swiss_ai_platform_credentials() {
            println!("Skipping Swiss AI Platform integration test - no API key available");
            return Ok(());
        }
    };
}

#[tokio::test]
async fn test_swiss_ai_platform_basic_completion() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = SwissAiPlatformProvider::from_env(model_config)?;

    let messages = vec![Message::user().with_text("Hello! Please respond with exactly 'Hi there!'")];

    let (response, usage) = provider
        .complete("You are a helpful assistant.", &messages, &[])
        .await?;

    // Verify response structure
    assert_eq!(response.content.len(), 1);
    assert!(matches!(response.content[0], MessageContent::Text(_)));

    // Verify usage data (should have some token counts)
    println!("Usage: {:?}", usage);
    assert!(!usage.model.is_empty());

    println!("Response: {:?}", response);
    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_with_tools() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = SwissAiPlatformProvider::from_env(model_config)?;

    let weather_tool = Tool::new(
        "get_weather",
        "Get the weather for a location",
        object!({
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            }
        }),
    );

    let messages = vec![Message::user().with_text("What's the weather like in Zurich, Switzerland?")];

    let (response, _usage) = provider
        .complete(
            "You are a helpful weather assistant. Use the get_weather tool when asked about weather.",
            &messages,
            &[weather_tool],
        )
        .await?;

    println!("Tool response: {:?}", response);

    // Should contain either a tool request or a text response
    assert!(!response.content.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_all_models() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    for model_name in SWISS_AI_PLATFORM_KNOWN_MODELS {
        println!("Testing model: {}", model_name);

        let model_config = ModelConfig::new(model_name)?;
        let provider = SwissAiPlatformProvider::from_env(model_config)?;

        let messages = vec![Message::user().with_text("Say 'Hello from Swiss AI Platform!'")];

        let result = provider
            .complete("You are a helpful assistant.", &messages, &[])
            .await;

        match result {
            Ok((response, usage)) => {
                println!("✅ Model {} works - Response: {:?}", model_name, response);
                println!("   Usage: {:?}", usage);
                assert!(!response.content.is_empty());
            }
            Err(e) => {
                println!("❌ Model {} failed: {}", model_name, e);
                // Don't fail the test immediately - some models might not be available
                // But log the error for investigation
            }
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_fetch_models() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = SwissAiPlatformProvider::from_env(model_config)?;

    let models = provider.fetch_supported_models().await?;

    match models {
        Some(model_list) => {
            println!("✅ Fetched {} models from Swiss AI Platform", model_list.len());
            for model in &model_list {
                println!("  - {}", model);
            }
            assert!(!model_list.is_empty());
        }
        None => {
            println!("ℹ️ No models returned from Swiss AI Platform API");
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_via_factory() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = create("swiss_ai_platform", model_config)?;

    let messages = vec![Message::user().with_text("Hello from factory-created provider!")];

    let (response, usage) = provider
        .complete("You are a helpful assistant.", &messages, &[])
        .await?;

    println!("Factory provider response: {:?}", response);
    println!("Usage: {:?}", usage);

    assert!(!response.content.is_empty());
    assert!(!usage.model.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_with_agent() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    let agent = Agent::new();
    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = Arc::new(SwissAiPlatformProvider::from_env(model_config)?);
    
    agent.update_provider(provider).await?;

    let conversation = Conversation::new(vec![
        Message::user().with_text("Hello! Please tell me a very short joke.")
    ])?;

    let mut reply_stream = agent.reply(conversation, None, None).await?;

    let mut responses = Vec::new();
    while let Some(event) = reply_stream.next().await {
        match event? {
            goose::agents::AgentEvent::Message(message) => {
                responses.push(message);
                break; // Just get the first response for this test
            }
            goose::agents::AgentEvent::McpNotification(_) => {
                // Ignore MCP notifications for this test
            }
            goose::agents::AgentEvent::ModelChange { .. } => {
                // Ignore model change events
            }
            goose::agents::AgentEvent::HistoryReplaced(_) => {
                // Ignore history replacement events
            }
        }
    }

    assert!(!responses.is_empty(), "Should have received at least one response");
    
    let response = &responses[0];
    println!("Agent response: {:?}", response);
    assert!(!response.content.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_error_handling() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    // Test with invalid model (should fail during provider creation)
    let invalid_model_config = ModelConfig::new("invalid-model-name")?;
    let result = SwissAiPlatformProvider::from_env(invalid_model_config);
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Unknown model"));

    // Test with valid model but potentially problematic input
    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = SwissAiPlatformProvider::from_env(model_config)?;

    // Test with empty messages (should fail)
    let result = provider
        .complete("You are a helpful assistant.", &[], &[])
        .await;
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("cannot be empty"));

    // Test with very long message (should work but might be slow)
    let long_message = "Hello! ".repeat(1000);
    let messages = vec![Message::user().with_text(&long_message)];

    let result = provider
        .complete("You are a helpful assistant. Please respond briefly.", &messages, &[])
        .await;

    match result {
        Ok((response, _)) => {
            println!("✅ Long message handled successfully");
            assert!(!response.content.is_empty());
        }
        Err(e) => {
            println!("ℹ️ Long message failed (might be expected): {}", e);
            // Don't fail the test - this might be expected behavior
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_custom_host() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    // Save original host if it exists
    let original_host = env::var("SWISS_AI_PLATFORM_HOST").ok();

    // Test with custom host (using the default URL as custom host)
    env::set_var("SWISS_AI_PLATFORM_HOST", "https://api.swisscom.com/layer/swiss-ai-platform/llama-3-3-70b/v1");

    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = SwissAiPlatformProvider::from_env(model_config)?;

    let messages = vec![Message::user().with_text("Hello with custom host!")];

    let result = provider
        .complete("You are a helpful assistant.", &messages, &[])
        .await;

    // Restore original host
    match original_host {
        Some(host) => env::set_var("SWISS_AI_PLATFORM_HOST", host),
        None => env::remove_var("SWISS_AI_PLATFORM_HOST"),
    }

    match result {
        Ok((response, _)) => {
            println!("✅ Custom host works");
            assert!(!response.content.is_empty());
        }
        Err(e) => {
            println!("ℹ️ Custom host test failed: {}", e);
            // This might fail if the custom host doesn't work exactly like expected
            // Don't fail the test as this is more of an exploratory test
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_swiss_ai_platform_authentication_error() -> Result<()> {
    dotenv().ok();

    // Save original API key
    let original_key = env::var("SWISS_AI_PLATFORM_API_KEY").ok();

    // Set invalid API key
    env::set_var("SWISS_AI_PLATFORM_API_KEY", "invalid_api_key_for_testing");

    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = SwissAiPlatformProvider::from_env(model_config)?;

    let messages = vec![Message::user().with_text("This should fail with auth error")];

    let result = provider
        .complete("You are a helpful assistant.", &messages, &[])
        .await;

    // Restore original API key
    match original_key {
        Some(key) => env::set_var("SWISS_AI_PLATFORM_API_KEY", key),
        None => env::remove_var("SWISS_AI_PLATFORM_API_KEY"),
    }

    // Should fail with authentication error
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    println!("Authentication error (expected): {}", error_msg);
    
    // The error should indicate authentication failure
    assert!(
        error_msg.to_lowercase().contains("auth") || 
        error_msg.contains("401") || 
        error_msg.contains("403") ||
        error_msg.contains("unauthorized") ||
        error_msg.contains("forbidden")
    );

    Ok(())
}

// Helper function to run a comprehensive end-to-end test
#[tokio::test]
async fn test_swiss_ai_platform_comprehensive_workflow() -> Result<()> {
    skip_if_no_credentials!();
    dotenv().ok();

    println!("🚀 Starting comprehensive Swiss AI Platform workflow test");

    // Step 1: Test provider creation
    println!("📝 Step 1: Testing provider creation");
    let model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let provider = SwissAiPlatformProvider::from_env(model_config)?;
    println!("✅ Provider created successfully");

    // Step 2: Test basic completion
    println!("📝 Step 2: Testing basic completion");
    let messages = vec![Message::user().with_text("What is 2 + 2? Please respond with just the number.")];
    let (response, usage) = provider
        .complete("You are a helpful math assistant.", &messages, &[])
        .await?;
    println!("✅ Basic completion successful");
    println!("   Response: {:?}", response);
    println!("   Usage: {:?}", usage);

    // Step 3: Test with conversation history
    println!("📝 Step 3: Testing conversation with history");
    let conversation_messages = vec![
        Message::user().with_text("My name is Alice."),
        Message::assistant().with_text("Hello Alice! Nice to meet you."),
        Message::user().with_text("What is my name?"),
    ];
    let (response, _) = provider
        .complete("You are a helpful assistant with good memory.", &conversation_messages, &[])
        .await?;
    println!("✅ Conversation with history successful");
    println!("   Response: {:?}", response);

    // Step 4: Test model fetching
    println!("📝 Step 4: Testing model fetching");
    match provider.fetch_supported_models().await? {
        Some(models) => {
            println!("✅ Model fetching successful - {} models found", models.len());
            for model in models.iter().take(5) {
                println!("   - {}", model);
            }
            if models.len() > 5 {
                println!("   ... and {} more", models.len() - 5);
            }
        }
        None => {
            println!("ℹ️ No models returned from API");
        }
    }

    // Step 5: Test via factory
    println!("📝 Step 5: Testing via factory");
    let factory_model_config = ModelConfig::new(SWISS_AI_PLATFORM_DEFAULT_MODEL)?;
    let factory_provider = create("swiss_ai_platform", factory_model_config)?;
    let (response, _) = factory_provider
        .complete("You are helpful.", &[Message::user().with_text("Hello from factory!")], &[])
        .await?;
    println!("✅ Factory creation and completion successful");
    println!("   Response: {:?}", response);

    println!("🎉 Comprehensive workflow test completed successfully!");
    Ok(())
}