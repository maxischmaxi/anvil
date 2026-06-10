use crate::llm::ChatMessage;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenStats {
    pub sent_since_prompt: usize,
    pub received_since_command: usize,
    pub context: usize,
    pub context_limit: Option<usize>,
}

pub fn estimate_text(text: &str) -> usize {
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    ((chars + 3) / 4).max(words).max(usize::from(!text.is_empty()))
}

pub fn estimate_messages(system: Option<&str>, messages: &[ChatMessage]) -> usize {
    let mut tokens = system.map_or(0, estimate_text);
    for message in messages {
        tokens += 4;
        match message {
            ChatMessage::User(text) => tokens += estimate_text(text),
            ChatMessage::Assistant { text, tool_calls } => {
                tokens += estimate_text(text);
                for call in tool_calls {
                    tokens += 8 + estimate_text(&call.name) + estimate_text(&call.arguments.to_string());
                }
            }
            ChatMessage::ToolResults(results) => {
                for result in results {
                    tokens += 8 + estimate_text(&result.id) + estimate_text(&result.content);
                }
            }
        }
    }
    tokens
}

pub fn context_limit(model: &str) -> Option<usize> {
    let normalized = model.to_ascii_lowercase();
    let model = normalized
        .strip_prefix("openai/")
        .or_else(|| normalized.strip_prefix("anthropic/"))
        .or_else(|| normalized.strip_prefix("google/"))
        .unwrap_or(&normalized);

    match model {
        "gpt-5.5" | "gpt-5.5-mini" | "gpt-5.2" | "gpt-5.1-codex" | "gpt-5.1-codex-mini" => Some(400_000),
        "gpt-4.1" | "gpt-4.1-mini" | "gpt-4.1-nano" => Some(1_047_576),
        "gpt-4o" | "gpt-4o-mini" | "o1" | "o1-mini" | "o3" | "o3-mini" | "o4-mini" => Some(128_000),
        "claude-sonnet-4-6" | "claude-opus-4-1" | "claude-sonnet-4" | "claude-opus-4" => Some(200_000),
        "gemini-2.5-pro" | "gemini-2.5-flash" | "gemini-1.5-pro" | "gemini-1.5-flash" => Some(1_048_576),
        _ if model.contains("gemini") => Some(1_048_576),
        _ if model.contains("claude") => Some(200_000),
        _ if model.contains("gpt-4.1") => Some(1_047_576),
        _ if model.contains("gpt") || model.starts_with('o') => Some(128_000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knows_openrouter_prefixed_models() {
        assert_eq!(context_limit("openai/gpt-5.5"), Some(400_000));
        assert_eq!(context_limit("anthropic/claude-sonnet-4-6"), Some(200_000));
    }
}
