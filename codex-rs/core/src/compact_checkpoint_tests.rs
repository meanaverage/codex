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
async fn stages_run_once_each_in_ascending_order() {
    let mut state = CompactionCheckpointState::default();
    assert_eq!(state.reserve(1, 0), Some(Vec::new()));
    state.begin(idle_handle());
    assert_eq!(state.reserve(1, 1), None, "one capture at a time");
    state.finish(None);
    assert_eq!(state.reserve(1, 0), None, "a failed stage is not retried");
    assert!(state.reserve(1, 1).is_some(), "the next stage still runs");
    state.finish(None);
    assert_eq!(state.reserve(2, 0), Some(Vec::new()), "a new window starts over");
}

#[tokio::test]
async fn later_stage_builds_on_completed_stages() {
    let first = vec![message("user", "one"), message("assistant", "two")];
    let mut state = CompactionCheckpointState::default();
    state.reserve(3, 0);
    state.begin(idle_handle());
    state.finish(Some(CompactionCheckpoint {
        prefix: Arc::new(first),
        summary: "first".to_string(),
    }));
    let prior = state.reserve(3, 1).expect("stage 1 reserved");
    assert_eq!(prior.len(), 1);
    assert_eq!(prior[0].summary, "first");
}

#[tokio::test]
async fn ready_stages_require_matching_window_and_prefixes() {
    let first = vec![message("user", "one"), message("assistant", "two")];
    let mut second = first.clone();
    second.push(message("user", "three"));
    let mut state = CompactionCheckpointState::default();
    state.reserve(3, 0);
    state.finish(Some(CompactionCheckpoint {
        prefix: Arc::new(first.clone()),
        summary: "first".to_string(),
    }));
    state.reserve(3, 1);
    state.finish(Some(CompactionCheckpoint {
        prefix: Arc::new(second.clone()),
        summary: "second".to_string(),
    }));

    let mut history = second.clone();
    history.push(message("assistant", "four"));
    let summaries = |stages: Vec<CompactionCheckpoint>| {
        stages
            .into_iter()
            .map(|stage| stage.summary)
            .collect::<Vec<_>>()
    };
    assert_eq!(summaries(state.ready_for(3, &history)), vec!["first", "second"]);
    assert!(state.ready_for(4, &history).is_empty(), "other window");
    assert_eq!(
        summaries(state.ready_for(3, &first)),
        vec!["first"],
        "history shorter than the second prefix keeps only the first stage"
    );
    let mut rewritten = history.clone();
    rewritten[0] = message("user", "edited");
    assert!(state.ready_for(3, &rewritten).is_empty(), "rewritten prefix");
}

#[test]
fn concatenation_skips_empty_parts() {
    assert_eq!(concatenate_summaries(["a", "b"]), "a\n\nb");
    assert_eq!(concatenate_summaries(["a\n", "  ", "c"]), "a\n\nc");
    assert_eq!(concatenate_summaries(["", "b"]), "b");
}

#[test]
fn delta_instructions_embed_checkpoint_and_user_instructions() {
    let text = delta_compaction_instructions("PRIOR", "DO THIS");
    assert!(text.contains("<prior_checkpoint>\nPRIOR\n</prior_checkpoint>"));
    assert!(text.ends_with("DO THIS"));
}
