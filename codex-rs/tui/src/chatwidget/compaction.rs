//! Live compaction status. Its wall clock is separate from the turn's running time,
//! and only a matching live completion contributes a duration to the transcript.

use super::*;

pub(super) const COMPACTION_HEADER: &str = "Compacting context";
pub(super) const COMPACTION_DETAILS: &str = "Making room to continue.";

#[derive(Debug)]
pub(super) struct ActiveCompaction {
    pub(super) id: String,
    pub(super) started_at: Instant,
}

impl ChatWidget {
    pub(super) fn on_context_compaction_started(&mut self, id: String, elapsed: Duration) {
        if self
            .status_state
            .compaction
            .as_ref()
            .is_some_and(|active| active.id == id)
        {
            return;
        }
        self.flush_answer_stream_with_separator();
        let now = Instant::now();
        let started_at = now.checked_sub(elapsed).unwrap_or(now);
        self.status_state.compaction = Some(ActiveCompaction { id, started_at });
        self.bottom_pane.set_status_timer_origin(Some(started_at));
        self.bottom_pane.ensure_status_indicator();
        self.set_status_header(COMPACTION_HEADER.to_string());
    }

    pub(super) fn clear_context_compaction(&mut self) {
        if self.status_state.compaction.take().is_some() {
            self.bottom_pane
                .set_status_timer_origin(/*started_at*/ None);
            self.set_status_header("Working".to_string());
        }
    }

    pub(super) fn on_context_compaction_completed(&mut self, id: &str, from_replay: bool) {
        let mut message = "Context compacted".to_string();
        if let Some(active) = self.status_state.compaction.as_ref()
            && active.id == id
        {
            if !from_replay {
                let elapsed = crate::status_indicator_widget::fmt_elapsed_compact(
                    active.started_at.elapsed().as_secs(),
                );
                message = format!("Context compacted · {elapsed}");
            }
            self.clear_context_compaction();
        }
        self.add_info_message(message, /*hint*/ None);
    }
}

/// Describes a provider for `/squisher` output: display name plus base URL when known.
fn describe_provider(provider: &codex_model_provider_info::ModelProviderInfo) -> String {
    match provider.base_url.as_deref() {
        Some(base_url) => format!("{} ({base_url})", provider.name),
        None => provider.name.clone(),
    }
}

impl ChatWidget {
    /// Provider keys a user may pick for compaction, excluding the session provider itself
    /// (selecting that is spelled `off`).
    fn compaction_provider_choices(&self) -> Vec<String> {
        let mut keys = self
            .config
            .model_providers
            .keys()
            .filter(|key| **key != self.config.model_provider_id)
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();
        keys
    }

    fn compaction_provider_summary(&self) -> String {
        match &self.config.compact_model_provider {
            Some(provider) => format!("Compaction runs on {}.", describe_provider(provider)),
            None => format!(
                "Compaction runs on the session provider, {}.",
                describe_provider(&self.config.model_provider)
            ),
        }
    }

    /// `/squisher` with no argument: open a picker over the configured providers.
    pub(super) fn show_compaction_provider(&mut self) {
        let session_id = self.config.model_provider_id.clone();
        let current_id = self
            .config
            .compact_model_provider
            .as_ref()
            .and_then(|current| {
                self.config
                    .model_providers
                    .iter()
                    .find(|(_, info)| {
                        info.base_url == current.base_url && info.name == current.name
                    })
                    .map(|(key, _)| key.clone())
            });

        let mut items = Vec::new();
        let session_provider = self.config.model_provider.clone();
        items.push(SelectionItem {
            name: format!("Session provider ({session_id})"),
            description: Some(format!(
                "Compact on the same server as the conversation: {}",
                describe_provider(&session_provider)
            )),
            is_current: current_id.is_none(),
            actions: vec![Box::new(|tx: &AppEventSender| {
                tx.send(AppEvent::PersistCompactionProviderSelection { provider_id: None });
            })],
            dismiss_on_select: true,
            search_value: Some(format!("off session {session_id}")),
            ..Default::default()
        });
        for key in self.compaction_provider_choices() {
            let Some(info) = self.config.model_providers.get(&key) else {
                continue;
            };
            let provider_id = key.clone();
            items.push(SelectionItem {
                name: key.clone(),
                description: Some(describe_provider(info)),
                is_current: current_id.as_deref() == Some(key.as_str()),
                actions: vec![Box::new(move |tx: &AppEventSender| {
                    tx.send(AppEvent::PersistCompactionProviderSelection {
                        provider_id: Some(provider_id.clone()),
                    });
                })],
                dismiss_on_select: true,
                search_value: Some(format!(
                    "{key} {}",
                    info.base_url.clone().unwrap_or_default()
                )),
                ..Default::default()
            });
        }
        let initial_selected_idx = items.iter().position(|item| item.is_current);
        self.show_selection_view(SelectionViewParams {
            title: Some("Where should compaction run?".to_string()),
            subtitle: Some(self.compaction_provider_summary()),
            footer_hint: Some(standard_popup_hint_line()),
            initial_selected_idx,
            is_searchable: items.len() > 4,
            search_placeholder: Some("Type to filter providers".to_string()),
            items,
            ..SelectionViewParams::picker()
        });
    }

    /// `/squisher <provider|off>`: persist the compaction provider to the active config
    /// file and hot-reload it; the change applies to the next checkpoint or compaction.
    pub(super) fn select_compaction_provider(&mut self, arg: &str) {
        let arg = arg.trim();
        let selection = match arg.to_ascii_lowercase().as_str() {
            "off" | "none" | "session" => None,
            _ => {
                let choices = self.compaction_provider_choices();
                // Accept an exact key or a unique suffix such as `dgx3`.
                let matched = if self.config.model_providers.contains_key(arg) {
                    Some(arg.to_string())
                } else {
                    let candidates = choices
                        .iter()
                        .filter(|key| key.ends_with(arg) || key.contains(arg))
                        .cloned()
                        .collect::<Vec<_>>();
                    match candidates.as_slice() {
                        [only] => Some(only.clone()),
                        [] => None,
                        _ => {
                            self.add_error_message(format!(
                                "`{arg}` matches several providers: {}",
                                candidates.join(", ")
                            ));
                            return;
                        }
                    }
                };
                let Some(key) = matched else {
                    self.add_error_message(format!(
                        "Unknown model provider `{arg}`. Available: {}",
                        if choices.is_empty() {
                            "(none)".to_string()
                        } else {
                            choices.join(", ")
                        }
                    ));
                    return;
                };
                if key == self.config.model_provider_id {
                    None
                } else {
                    Some(key)
                }
            }
        };
        self.app_event_tx
            .send(AppEvent::PersistCompactionProviderSelection {
                provider_id: selection,
            });
    }

    /// Applies a persisted compaction provider to the widget's config copy and reports it.
    pub(crate) fn on_compaction_provider_saved(&mut self, provider_id: Option<String>) {
        self.config.compact_model_provider = provider_id
            .as_deref()
            .and_then(|key| self.config.model_providers.get(key).cloned());
        self.add_info_message(
            self.compaction_provider_summary(),
            Some("Applies to the next compaction checkpoint or compaction.".to_string()),
        );
    }
}
