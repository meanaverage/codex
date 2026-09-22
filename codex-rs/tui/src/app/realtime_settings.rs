//! Resolve and persist voice preferences through the owning server before updating the UI.

use super::*;
use codex_app_server_protocol::ClientRequest;
use codex_app_server_protocol::RequestId;
use codex_app_server_protocol::ThreadRealtimeListVoicesParams;
use codex_app_server_protocol::ThreadRealtimeListVoicesResponse;
use codex_protocol::protocol::RealtimeVoice;
use codex_protocol::protocol::RealtimeVoicesList;
use uuid::Uuid;

/// Unknown voices belong to newer servers; omit the override instead of blocking audio.
/// An unset preference explicitly selects the V1/V3 server default, not a stale thread preference.
fn configured_voice(
    config: &codex_app_server_protocol::Config,
    default_voice: RealtimeVoice,
) -> Result<Option<RealtimeVoice>> {
    let value = config
        .additional
        .get("realtime")
        .and_then(|realtime| realtime.get("voice"));
    match value {
        None | Some(serde_json::Value::Null) => Ok(Some(default_voice)),
        Some(value @ serde_json::Value::String(_)) => {
            Ok(serde_json::from_value(value.clone()).ok())
        }
        Some(value) => Ok(Some(serde_json::from_value(value.clone())?)),
    }
}

impl App {
    pub(super) async fn realtime_voices(
        &self,
        app_server: &AppServerSession,
    ) -> RealtimeVoicesList {
        match app_server
            .request_handle()
            .request_typed::<ThreadRealtimeListVoicesResponse>(
                ClientRequest::ThreadRealtimeListVoices {
                    request_id: RequestId::String(format!("tui-voice-list-{}", Uuid::new_v4())),
                    params: ThreadRealtimeListVoicesParams {},
                },
            )
            .await
        {
            Ok(response) => response.voices,
            Err(_) => RealtimeVoicesList::builtin(),
        }
    }

    pub(super) async fn effective_realtime_voice(
        &self,
        app_server: &AppServerSession,
        voices: &RealtimeVoicesList,
    ) -> Result<Option<RealtimeVoice>> {
        // Session setup installs the active conversation cwd for both server modes.
        let config = crate::config_update::read_effective_config_if_supported(
            app_server.request_handle(),
            &self.chat_widget.config_ref().cwd,
        )
        .await?;
        config
            .as_ref()
            .map(|config| configured_voice(config, voices.default_v1))
            .transpose()
            .map(Option::flatten)
    }

    pub(super) async fn open_realtime_settings(&mut self, app_server: &AppServerSession) {
        let voices = self.realtime_voices(app_server).await;
        match self.effective_realtime_voice(app_server, &voices).await {
            Ok(voice) => self.chat_widget.open_realtime_settings(voice, voices),
            Err(error) => self
                .chat_widget
                .add_error_message(format!("Failed to read voice settings: {error}")),
        }
    }

    /// `/squisher`: write the compaction provider and cadence into the active config file
    /// (the profile file under `--profile`) in one batch and hot-reload running threads.
    pub(super) async fn persist_compaction_settings(
        &mut self,
        app_server: &AppServerSession,
        provider_id: Option<String>,
        checkpoint_percents: Option<Vec<u8>>,
        trigger_percent: Option<u8>,
    ) {
        let mut edits = vec![match provider_id.as_deref() {
            Some(key) => crate::config_update::replace_config_value(
                "compact_model_provider",
                serde_json::json!(key),
            ),
            None => crate::config_update::clear_config_value("compact_model_provider"),
        }];
        if let Some(percents) = &checkpoint_percents {
            edits.push(crate::config_update::replace_config_value(
                "compact_checkpoint_percents",
                serde_json::json!(percents),
            ));
        }
        if let Some(percent) = trigger_percent {
            edits.push(crate::config_update::replace_config_value(
                "compact_trigger_percent",
                serde_json::json!(percent),
            ));
        }
        match crate::config_update::write_config_batch(app_server.request_handle(), edits).await {
            Ok(response) => {
                if response.status == codex_app_server_protocol::WriteStatus::OkOverridden {
                    self.chat_widget.add_error_message(format!(
                        "Compaction settings were saved but not applied: {}",
                        super::config_persistence::overridden_write_message(&response),
                    ));
                    return;
                }
                self.config.compact_model_provider = provider_id
                    .as_deref()
                    .and_then(|key| self.config.model_providers.get(key).cloned());
                if let Some(percents) = &checkpoint_percents {
                    self.config.compact_checkpoint_percents = percents.clone();
                }
                if let Some(percent) = trigger_percent {
                    self.config.compact_trigger_percent = percent;
                }
                self.chat_widget.on_compaction_settings_saved(
                    provider_id,
                    checkpoint_percents,
                    trigger_percent,
                );
            }
            Err(error) => self
                .chat_widget
                .add_error_message(format!("Failed to save compaction settings: {error}")),
        }
    }

    pub(super) async fn persist_realtime_voice(
        &mut self,
        app_server: &AppServerSession,
        voice: RealtimeVoice,
    ) {
        match crate::config_update::write_config_batch(
            app_server.request_handle(),
            vec![crate::config_update::replace_config_value(
                "realtime.voice",
                serde_json::json!(voice),
            )],
        )
        .await
        {
            Ok(response) => {
                let voices = self.realtime_voices(app_server).await;
                let effective = crate::config_update::read_effective_config_if_supported(
                    app_server.request_handle(), &self.chat_widget.config_ref().cwd,
                ).await.and_then(|config| match config.as_ref() {
                    Some(config) => configured_voice(config, voices.default_v1),
                    None if response.status == codex_app_server_protocol::WriteStatus::OkOverridden => Err(color_eyre::eyre::eyre!("the saved voice is overridden, but this server cannot read effective settings")),
                    None => Ok(Some(voice)),
                });
                match effective {
                    Ok(effective_voice) => {
                        self.config.realtime.voice = effective_voice;
                        self.chat_widget.set_realtime_voice(effective_voice);
                        if effective_voice == Some(voice) {
                            self.chat_widget.on_realtime_voice_saved(voice);
                        } else {
                            self.chat_widget.add_error_message(format!(
                                "Voice preference was saved but not applied: {}",
                                super::config_persistence::overridden_write_message(&response),
                            ));
                        }
                    }
                    Err(error) => self.chat_widget.add_error_message(format!("Voice preference was saved, but effective settings could not be read: {error}")),
                }
            }
            Err(error) => self
                .chat_widget
                .add_error_message(format!("Failed to save voice: {error}")),
        }
    }
}
