use super::CompactionCheckpoint;
use super::CompactionCheckpointState;
use super::concatenate_summaries;
use super::delta_compaction_instructions;
use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use pretty_assertions::assert_eq;
use std::sync::Arc;
use tokio_util::task::AbortOnDropHandle;

fn message(role: &str, text: &str) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    })
}

fn idle_handle() -> AbortOnDropHandle<()> {
    AbortOnDropHandle::new(tokio::spawn(std::future::pending()))
}

#[tokio::test]
async fn one_checkpoint_attempt_per_window() {
    let mut state = CompactionCheckpointState::default();
    assert!(state.reserve(1));
    state.begin(idle_handle());
    assert!(!state.reserve(1));
    state.finish(None);
    assert!(
        !state.reserve(1),
        "a failed attempt is not retried in the same window"
    );
    assert!(state.reserve(2));
    state.reset();
    assert!(state.reserve(2), "reset allows a fresh attempt");
}

#[tokio::test]
async fn ready_checkpoint_requires_matching_window_and_prefix() {
    let prefix = vec![message("user", "one"), message("assistant", "two")];
    let mut state = CompactionCheckpointState::default();
    assert!(state.reserve(3));
    state.begin(idle_handle());
    state.finish(Some(CompactionCheckpoint {
        window_number: 3,
        prefix: Arc::new(prefix.clone()),
        summary: "summary".to_string(),
    }));

    let mut history = prefix.clone();
    history.push(message("user", "three"));
    assert_eq!(
        state
            .ready_for(3, &history)
            .map(|checkpoint| checkpoint.summary.as_str()),
        Some("summary")
    );
    assert!(state.ready_for(4, &history).is_none(), "other window");
    assert!(
        state.ready_for(3, &prefix[..1]).is_none(),
        "history shorter than prefix"
    );
    let mut rewritten = history.clone();
    rewritten[0] = message("user", "edited");
    assert!(state.ready_for(3, &rewritten).is_none(), "rewritten prefix");
}

#[test]
fn concatenation_skips_empty_halves() {
    assert_eq!(concatenate_summaries("a", "b"), "a\n\nb");
    assert_eq!(concatenate_summaries("a\n", "  "), "a");
    assert_eq!(concatenate_summaries("", "b"), "b");
}

#[test]
fn delta_instructions_embed_checkpoint_and_user_instructions() {
    let text = delta_compaction_instructions("PRIOR", "DO THIS");
    assert!(text.contains("<prior_checkpoint>\nPRIOR\n</prior_checkpoint>"));
    assert!(text.ends_with("DO THIS"));
}
