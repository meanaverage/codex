//! Background compaction checkpoints.
//!
//! When `compact_checkpoint_threshold_percent` is set, a session that crosses that share of
//! its context window captures a checkpoint: a snapshot of history so far is summarized on the
//! compaction provider (`compact_model_provider` / `compact_model`) without touching the live
//! context. When real compaction later fires, only the items recorded after the checkpoint are
//! summarized (with the checkpoint summary supplied as read-only context), and the two summaries
//! are concatenated into the handover. One checkpoint is attempted per auto-compact window; a
//! failed or stale checkpoint simply falls back to ordinary full compaction.

use std::sync::Arc;

use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::WarningEvent;
use codex_protocol::user_input::UserInput;
use futures::prelude::*;
use tokio_util::task::AbortOnDropHandle;
use tracing::info;
use tracing::warn;

use crate::Prompt;
use crate::client::ModelClient;
use crate::client::ModelClientSession;
use crate::client_common::ResponseEvent;
use crate::compact::SUMMARIZATION_PROMPT;
use crate::context_manager::ContextManager;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::session::Session;
use crate::session::turn::get_last_assistant_message_from_turn;
use crate::session::turn_context::TurnContext;
use crate::util::backoff;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_features::Feature;
use codex_login::auth::AgentIdentityAuthPolicy;
use codex_model_provider::create_model_provider;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::openai_models::ModelInfo;
use codex_rollout_trace::InferenceTraceContext;

/// A completed checkpoint: the summarized history prefix and its summary.
#[derive(Debug, Clone)]
pub(crate) struct CompactionCheckpoint {
    /// Auto-compact window the checkpoint was captured in.
    pub(crate) window_number: u64,
    /// Exact history items the summary covers, oldest first.
    pub(crate) prefix: Arc<Vec<ResponseItemEnvelope>>,
    /// Raw model output for the prefix, without `SUMMARY_PREFIX`.
    pub(crate) summary: String,
}

/// Per-session checkpoint bookkeeping. Reset whenever a compaction succeeds.
#[derive(Default)]
pub(crate) struct CompactionCheckpointState {
    /// Window in which a checkpoint was last attempted; one attempt per window.
    attempted_window: Option<u64>,
    /// In-flight capture; dropping the handle aborts it.
    running: Option<AbortOnDropHandle<()>>,
    ready: Option<CompactionCheckpoint>,
}

impl CompactionCheckpointState {
    /// Marks this window as attempted and returns whether a capture may start.
    pub(crate) fn try_begin(&mut self, window_number: u64, handle: AbortOnDropHandle<()>) -> bool {
        if self.attempted_window == Some(window_number) {
            return false;
        }
        self.attempted_window = Some(window_number);
        self.ready = None;
        self.running = Some(handle);
        true
    }

    pub(crate) fn finish(&mut self, checkpoint: Option<CompactionCheckpoint>) {
        self.running = None;
        self.ready = checkpoint;
    }

    /// Returns the checkpoint if it still describes a prefix of `history`.
    pub(crate) fn ready_for(
        &self,
        window_number: u64,
        history: &[ResponseItemEnvelope],
    ) -> Option<&CompactionCheckpoint> {
        let checkpoint = self.ready.as_ref()?;
        if checkpoint.window_number != window_number
            || history.len() < checkpoint.prefix.len()
            || history[..checkpoint.prefix.len()] != checkpoint.prefix[..]
        {
            return None;
        }
        Some(checkpoint)
    }

    pub(crate) fn has_ready(&self) -> bool {
        self.ready.is_some()
    }

    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Request-side context for compaction: the session context with the compaction provider and
/// model swapped in when configured, plus a client bound to that provider.
pub(crate) struct CompactionRequest {
    pub(crate) turn_context: Arc<TurnContext>,
    pub(crate) client_session: ModelClientSession,
}

impl Session {
    /// Builds the request context and client for compaction. Without `compact_model_provider`
    /// or `compact_model` this is the session's own context and client.
    pub(crate) async fn compaction_request(
        &self,
        turn_context: &Arc<TurnContext>,
    ) -> CompactionRequest {
        let config = &turn_context.config;
        if config.compact_model_provider.is_none() && config.compact_model.is_none() {
            return CompactionRequest {
                turn_context: Arc::clone(turn_context),
                client_session: self.services.model_client.new_session(),
            };
        }

        // `with_model` is the only owned-copy constructor for a turn context; reusing the
        // session model keeps its settings while giving us a context we can rebind.
        let model = config
            .compact_model
            .clone()
            .unwrap_or_else(|| turn_context.model_info().slug.clone());
        let mut request_context = turn_context
            .with_model(model, &self.services.models_manager)
            .await;
        let Some(provider_info) = config.compact_model_provider.clone() else {
            return CompactionRequest {
                turn_context: Arc::new(request_context),
                client_session: self.services.model_client.new_session(),
            };
        };

        let auth_manager = Some(Arc::clone(&self.services.auth_manager));
        request_context.provider =
            create_model_provider(provider_info.clone(), auth_manager.clone());
        let client = ModelClient::new(
            auth_manager,
            if config.features.enabled(Feature::UseAgentIdentity) {
                AgentIdentityAuthPolicy::ChatGptAuth
            } else {
                AgentIdentityAuthPolicy::JwtOnly
            },
            self.thread_id(),
            provider_info,
            turn_context.session_source.clone(),
            turn_context.originator.clone(),
            config.model_verbosity,
            config.features.enabled(Feature::ContentItemKinds),
            self.services
                .model_client
                .reasoning_effort_override_enabled(request_context.model_info()),
            config.features.enabled(Feature::EnableRequestCompression),
            config.features.enabled(Feature::RuntimeMetrics),
            Self::build_model_client_beta_features_header(config.as_ref()),
            config
                .features
                .enabled(Feature::ConcurrentReasoningSummaries),
            self.services.attestation_provider.clone(),
            config.http_client_factory(),
            config.workspace_routing_context(),
        );
        CompactionRequest {
            turn_context: Arc::new(request_context),
            client_session: client.new_session(),
        }
    }

    /// Starts a background checkpoint capture for the current window if none was attempted yet.
    pub(crate) async fn maybe_start_compaction_checkpoint(
        self: &Arc<Self>,
        turn_context: &Arc<TurnContext>,
    ) {
        if turn_context.config.compact_checkpoint_threshold_percent == 0 {
            return;
        }
        let (_, window_number, _) = self.current_window().await;
        let prefix = self.clone_history().await.into_shared_annotated_items();
        if prefix.is_empty() {
            return;
        }
        let sess = Arc::clone(self);
        let ctx = Arc::clone(turn_context);
        let items = Arc::clone(&prefix);
        let handle = AbortOnDropHandle::new(tokio::spawn(async move {
            let result = capture_checkpoint(&sess, &ctx, window_number, items).await;
            let checkpoint = match result {
                Ok(checkpoint) => {
                    info!(
                        window_number,
                        items = checkpoint.prefix.len(),
                        summary_chars = checkpoint.summary.len(),
                        "compaction checkpoint captured"
                    );
                    Some(checkpoint)
                }
                Err(err) => {
                    warn!(error = %err, "compaction checkpoint failed; next compaction will summarize the full history");
                    sess.send_event(
                        ctx.as_ref(),
                        EventMsg::Warning(WarningEvent {
                            message: format!(
                                "Compaction checkpoint failed ({err}); the next compaction will summarize the full history."
                            ),
                        }),
                    )
                    .await;
                    None
                }
            };
            sess.finish_compaction_checkpoint(checkpoint).await;
        }));
        let started = self
            .begin_compaction_checkpoint(window_number, handle)
            .await;
        if started {
            info!(
                window_number,
                items = prefix.len(),
                "compaction checkpoint started"
            );
        }
    }
}

/// The instructions used for every local compaction request.
pub(crate) fn compaction_instructions(turn_context: &TurnContext) -> String {
    turn_context
        .config
        .compact_prompt
        .as_deref()
        .unwrap_or(SUMMARIZATION_PROMPT)
        .to_string()
}

/// Instructions for summarizing only the portion of history recorded after a checkpoint.
pub(crate) fn delta_compaction_instructions(
    checkpoint_summary: &str,
    instructions: &str,
) -> String {
    format!(
        "A checkpoint summary of the earlier part of this conversation was already produced by a \
previous compaction pass and is included below for reference only. It will be placed verbatim \
before your output in the handover, so do not repeat or restate it. Apply the instructions that \
follow it to the conversation items above this message, which are the portion recorded after \
that checkpoint, and describe only what is new.\n\n\
<prior_checkpoint>\n{checkpoint_summary}\n</prior_checkpoint>\n\n{instructions}"
    )
}

/// Joins the checkpoint summary and the delta summary into one handover summary.
pub(crate) fn concatenate_summaries(checkpoint_summary: &str, delta_summary: &str) -> String {
    let checkpoint_summary = checkpoint_summary.trim();
    let delta_summary = delta_summary.trim();
    if delta_summary.is_empty() {
        checkpoint_summary.to_string()
    } else if checkpoint_summary.is_empty() {
        delta_summary.to_string()
    } else {
        format!("{checkpoint_summary}\n\n{delta_summary}")
    }
}

/// Builds a history containing only `items` plus the compaction instructions.
pub(crate) fn history_for_items(
    items: &[ResponseItemEnvelope],
    instructions: String,
    model_info: &ModelInfo,
) -> ContextManager {
    let mut history = ContextManager::new();
    history.record_annotated_items(items, model_info.truncation_policy.into());
    let input: ResponseInputItem = vec![UserInput::Text {
        text: instructions,
        text_elements: Vec::new(),
    }]
    .into();
    history.record_items(&[input.into()], model_info.truncation_policy.into());
    history
}

async fn capture_checkpoint(
    sess: &Arc<Session>,
    turn_context: &Arc<TurnContext>,
    window_number: u64,
    prefix: Arc<Vec<ResponseItemEnvelope>>,
) -> CodexResult<CompactionCheckpoint> {
    let CompactionRequest {
        turn_context: request_context,
        mut client_session,
    } = sess.compaction_request(turn_context).await;
    let instructions = compaction_instructions(turn_context);
    let history = history_for_items(&prefix, instructions, request_context.model_info());
    let turn_input = history.for_prompt(&request_context.model_info().input_modalities);
    let prompt = Prompt {
        input: turn_input,
        base_instructions: sess.get_prompt_base_instructions().await,
        ..Default::default()
    };
    let responses_metadata = sess
        .compaction_responses_metadata(
            request_context.as_ref(),
            CompactionTurnMetadata::new(
                CompactionTrigger::Auto,
                CompactionReason::ContextLimit,
                CompactionImplementation::Responses,
                CompactionPhase::PostTurn,
            ),
        )
        .await;

    let max_retries = request_context.provider.info().stream_max_retries();
    let mut retries = 0;
    let summary = loop {
        match stream_summary(
            sess,
            request_context.as_ref(),
            &mut client_session,
            &responses_metadata,
            &prompt,
        )
        .await
        {
            Ok(summary) => break summary,
            Err(err) if retries < max_retries => {
                retries += 1;
                warn!(error = %err, retries, "compaction checkpoint request failed; retrying");
                tokio::time::sleep(backoff(retries)).await;
            }
            Err(err) => return Err(err),
        }
    };
    if summary.trim().is_empty() {
        return Err(CodexErr::Stream(
            "compaction checkpoint completed without an assistant summary".to_string(),
        ));
    }
    Ok(CompactionCheckpoint {
        window_number,
        prefix,
        summary,
    })
}

/// Streams one summarization request and returns the assistant's final message. Nothing is
/// recorded into session history or token accounting; the checkpoint is invisible to the live
/// context until compaction consumes it.
async fn stream_summary(
    sess: &Session,
    request_context: &TurnContext,
    client_session: &mut ModelClientSession,
    responses_metadata: &crate::responses_metadata::CodexResponsesMetadata,
    prompt: &Prompt,
) -> CodexResult<String> {
    let mut stream = client_session
        .stream(
            prompt,
            request_context.model_info(),
            &request_context.session_telemetry,
            sess.reasoning_effort_for_compaction(request_context).await,
            request_context.reasoning_summary(),
            request_context
                .service_tier_for_compaction(request_context.config.service_tier.clone()),
            responses_metadata,
            &InferenceTraceContext::disabled(),
        )
        .await?;
    let mut output: Vec<ResponseItem> = Vec::new();
    loop {
        let Some(event) = stream.next().await else {
            return Err(CodexErr::Stream(
                "stream closed before response.completed".into(),
            ));
        };
        match event {
            Ok(ResponseEvent::OutputItemDone(item)) => output.push(item),
            Ok(ResponseEvent::Completed { .. }) => {
                return Ok(get_last_assistant_message_from_turn(output.iter()).unwrap_or_default());
            }
            Ok(_) => continue,
            Err(err) => return Err(err),
        }
    }
}

#[cfg(test)]
#[path = "compact_checkpoint_tests.rs"]
mod tests;
