//! Background compaction checkpoints.
//!
//! Each entry of `compact_checkpoint_percents` is a stage. When a session crosses that share of
//! its context window, the history recorded since the previous stage is summarized on the
//! compaction provider (`compact_model_provider` / `compact_model`) without touching the live
//! context, with the earlier stage summaries supplied as read-only context. When real compaction
//! fires, only the items after the last stage are summarized and every stage summary is
//! concatenated into the handover. Each stage is attempted at most once per auto-compact window;
//! a failed stage is skipped and its span is covered by the next stage or the final compaction,
//! and a stale stage (history rewritten underneath it) falls back to full compaction.

use std::sync::Arc;

use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::WarningEvent;
use codex_protocol::user_input::UserInput;
use futures::prelude::*;
use tokio_util::task::AbortOnDropHandle;
use tracing::debug;
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
use codex_model_provider::RemoteCompactionSupport;
use codex_model_provider::create_model_provider;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::openai_models::ModelInfo;
use codex_rollout_trace::InferenceTraceContext;

/// Upper bound for one checkpoint capture, including retries.
const CHECKPOINT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20 * 60);

/// A completed checkpoint stage: the history prefix it covers and the summary of the part of
/// that prefix recorded after the previous stage.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompactionCheckpoint {
    /// Exact history items covered so far, oldest first (includes earlier stages' items).
    pub(crate) prefix: Arc<Vec<ResponseItemEnvelope>>,
    /// Raw model output for this stage's span, without `SUMMARY_PREFIX`.
    pub(crate) summary: String,
}

/// Per-session checkpoint bookkeeping for the active auto-compact window.
#[derive(Default)]
pub(crate) struct CompactionCheckpointState {
    window_number: Option<u64>,
    /// Highest stage attempted in this window; each stage runs at most once.
    attempted_stage: Option<usize>,
    /// In-flight capture; dropping the handle aborts it.
    running: Option<AbortOnDropHandle<()>>,
    /// Completed stages, ascending; each prefix extends the previous one.
    completed: Vec<CompactionCheckpoint>,
}

impl CompactionCheckpointState {
    /// Reserves `stage` for this window and returns the completed stages it builds on, or
    /// `None` when the stage was already attempted or another capture is in flight.
    pub(crate) fn reserve(
        &mut self,
        window_number: u64,
        stage: usize,
    ) -> Option<Vec<CompactionCheckpoint>> {
        if self.window_number != Some(window_number) {
            self.reset();
            self.window_number = Some(window_number);
        }
        if self.running.is_some() || self.attempted_stage.is_some_and(|last| last >= stage) {
            return None;
        }
        self.attempted_stage = Some(stage);
        Some(self.completed.clone())
    }

    /// Stores the in-flight capture so dropping the state aborts it.
    pub(crate) fn begin(&mut self, handle: AbortOnDropHandle<()>) {
        self.running = Some(handle);
    }

    pub(crate) fn finish(&mut self, checkpoint: Option<CompactionCheckpoint>) {
        self.running = None;
        if let Some(checkpoint) = checkpoint {
            self.completed.push(checkpoint);
        }
    }

    /// Returns the completed stages that still describe a prefix of `history`, ascending.
    pub(crate) fn ready_for(
        &self,
        window_number: u64,
        history: &[ResponseItemEnvelope],
    ) -> Vec<CompactionCheckpoint> {
        if self.window_number != Some(window_number) {
            return Vec::new();
        }
        self.completed
            .iter()
            .take_while(|checkpoint| {
                history.len() >= checkpoint.prefix.len()
                    && history[..checkpoint.prefix.len()] == checkpoint.prefix[..]
            })
            .cloned()
            .collect()
    }

    pub(crate) fn has_completed(&self) -> bool {
        !self.completed.is_empty()
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

        let auth_manager = Some(Arc::clone(&self.services.auth_manager));
        let provider = config
            .compact_model_provider
            .clone()
            .map(|provider_info| create_model_provider(provider_info, auth_manager.clone()))
            .unwrap_or_else(|| turn_context.provider.clone());
        let mut request_context = turn_context.with_provider(provider);
        // Only a different compaction model needs the (catalog-refreshing) model re-resolution.
        if let Some(model) = config.compact_model.clone()
            && model != turn_context.model_info().slug
        {
            debug!(model, "resolving compaction model");
            let provider = request_context.provider.clone();
            request_context = request_context
                .with_model(model, &self.services.models_manager)
                .await;
            request_context.provider = provider;
        }
        let Some(provider_info) = config.compact_model_provider.clone() else {
            return CompactionRequest {
                turn_context: Arc::new(request_context),
                client_session: self.services.model_client.new_session(),
            };
        };

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

    /// Starts a background capture for checkpoint `stage` unless it already ran this window.
    pub(crate) async fn maybe_start_compaction_checkpoint(
        self: &Arc<Self>,
        turn_context: &Arc<TurnContext>,
        stage: usize,
    ) {
        // Remote (backend-side) compaction never consumes checkpoints, so only capture them
        // when compaction runs locally or on a dedicated provider.
        let config = &turn_context.config;
        if config.compact_model_provider.is_none()
            && !matches!(
                turn_context.provider.capabilities().remote_compaction,
                RemoteCompactionSupport::Unsupported
            )
        {
            return;
        }
        let (_, window_number, _) = self.current_window().await;
        let Some(prior) = self
            .reserve_compaction_checkpoint(window_number, stage)
            .await
        else {
            return;
        };
        let prefix = self.clone_history().await.into_shared_annotated_items();
        let covered = prior.last().map_or(0, |last| last.prefix.len());
        if prefix.len() <= covered {
            return;
        }
        let sess = Arc::clone(self);
        let ctx = Arc::clone(turn_context);
        let items = Arc::clone(&prefix);
        let handle = AbortOnDropHandle::new(tokio::spawn(async move {
            let result = match tokio::time::timeout(
                CHECKPOINT_TIMEOUT,
                capture_checkpoint(&sess, &ctx, stage, prior, items),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(CodexErr::Stream(format!(
                    "compaction checkpoint timed out after {}s",
                    CHECKPOINT_TIMEOUT.as_secs()
                ))),
            };
            let checkpoint = match result {
                Ok(checkpoint) => {
                    info!(
                        window_number,
                        stage,
                        items = checkpoint.prefix.len(),
                        summary_chars = checkpoint.summary.len(),
                        "compaction checkpoint captured"
                    );
                    Some(checkpoint)
                }
                Err(err) => {
                    warn!(error = %err, stage, "compaction checkpoint failed; its span will be summarized by the next stage or compaction");
                    sess.send_event(
                        ctx.as_ref(),
                        EventMsg::Warning(WarningEvent {
                            message: format!(
                                "Compaction checkpoint {} failed ({err}); its span will be summarized later.",
                                stage + 1
                            ),
                        }),
                    )
                    .await;
                    None
                }
            };
            sess.finish_compaction_checkpoint(checkpoint).await;
        }));
        self.begin_compaction_checkpoint(handle).await;
        info!(
            window_number,
            stage,
            items = prefix.len() - covered,
            "compaction checkpoint started"
        );
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

/// Instructions for summarizing only the portion of history recorded after the checkpoints
/// whose combined summary is `prior_summary`.
pub(crate) fn delta_compaction_instructions(prior_summary: &str, instructions: &str) -> String {
    format!(
        "Checkpoint summaries of the earlier part of this conversation were already produced by \
previous compaction passes and are included below for reference only. They will be placed \
verbatim before your output in the handover, so do not repeat or restate them. Apply the \
instructions that follow them to the conversation items above this message, which are the \
portion recorded after the last checkpoint, and describe only what is new.\n\n\
<prior_checkpoint>\n{prior_summary}\n</prior_checkpoint>\n\n{instructions}"
    )
}

/// Joins stage summaries (and the final delta summary) into one handover summary.
pub(crate) fn concatenate_summaries<'a>(summaries: impl IntoIterator<Item = &'a str>) -> String {
    summaries
        .into_iter()
        .map(str::trim)
        .filter(|summary| !summary.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The combined summary of completed stages, in order.
pub(crate) fn checkpoint_summary(checkpoints: &[CompactionCheckpoint]) -> String {
    concatenate_summaries(
        checkpoints
            .iter()
            .map(|checkpoint| checkpoint.summary.as_str()),
    )
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
    stage: usize,
    prior: Vec<CompactionCheckpoint>,
    prefix: Arc<Vec<ResponseItemEnvelope>>,
) -> CodexResult<CompactionCheckpoint> {
    debug!(stage, "compaction checkpoint: building request context");
    let CompactionRequest {
        turn_context: request_context,
        mut client_session,
    } = sess.compaction_request(turn_context).await;
    debug!(
        provider = request_context.provider.info().name,
        model = request_context.model_info().slug,
        "compaction checkpoint: request context ready"
    );
    let mut instructions = compaction_instructions(turn_context);
    let covered = prior.last().map_or(0, |last| last.prefix.len());
    if !prior.is_empty() {
        instructions = delta_compaction_instructions(&checkpoint_summary(&prior), &instructions);
    }
    let history = history_for_items(
        &prefix[covered..],
        instructions,
        request_context.model_info(),
    );
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
    debug!(
        input_items = prompt.input.len(),
        "compaction checkpoint: sending request"
    );
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
    Ok(CompactionCheckpoint { prefix, summary })
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
