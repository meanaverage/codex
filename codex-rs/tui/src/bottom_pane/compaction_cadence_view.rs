//! Slider view for `/squisher`: choose how many compaction checkpoints run and at which
//! context-window percentages, plus the final compaction trigger. Nothing is persisted until
//! Enter on the sliders; Esc or Ctrl-C leaves the configuration untouched.

use crossterm::event::KeyCode;
use crossterm::event::KeyEvent;
use crossterm::event::KeyModifiers;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Widget;

use super::CancellationEvent;
use super::bottom_pane_view::BottomPaneView;
use super::bottom_pane_view::ViewCompletion;
use super::selection_popup_common::menu_surface_padding_height;
use super::selection_popup_common::render_menu_surface;
use crate::app_event::AppEvent;
use crate::app_event_sender::AppEventSender;
use crate::key_hint;
use crate::keymap::KeymapContext;
use crate::keymap::KeymapContextSet;
use crate::render::renderable::Renderable;

/// Most checkpoint stages the view offers; more than this rarely helps a handover.
const MAX_CHECKPOINTS: usize = 4;
const SMALL_STEP: u8 = 1;
const LARGE_STEP: u8 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// Pick how many checkpoint stages to run.
    Divisions,
    /// Set each stage's percentage and the compaction trigger.
    Sliders,
}

/// Cadence values: ascending checkpoint percentages followed by the compaction trigger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompactionCadence {
    pub(crate) checkpoint_percents: Vec<u8>,
    pub(crate) trigger_percent: u8,
}

impl CompactionCadence {
    /// Rebuilds the stage list for `divisions` stages, spacing new stages evenly below the
    /// trigger while keeping already-configured values when the count is unchanged.
    fn with_divisions(&self, divisions: usize) -> Self {
        if divisions == self.checkpoint_percents.len() {
            return self.clone();
        }
        let trigger = u32::from(self.trigger_percent);
        let checkpoint_percents = (1..=divisions as u32)
            .map(|index| (trigger * index / (divisions as u32 + 1)).max(index) as u8)
            .collect();
        Self {
            checkpoint_percents,
            trigger_percent: self.trigger_percent,
        }
    }

    fn values(&self) -> Vec<u8> {
        let mut values = self.checkpoint_percents.clone();
        values.push(self.trigger_percent);
        values
    }
}

pub(crate) struct CompactionCadenceView {
    provider_id: Option<String>,
    provider_label: String,
    step: Step,
    divisions: usize,
    cadence: CompactionCadence,
    /// Focused handle in the sliders step; the last index is the compaction trigger.
    focus: usize,
    app_event_tx: AppEventSender,
    completion: Option<ViewCompletion>,
}

impl CompactionCadenceView {
    pub(crate) fn new(
        provider_id: Option<String>,
        provider_label: String,
        current: CompactionCadence,
        app_event_tx: AppEventSender,
    ) -> Self {
        let divisions = current.checkpoint_percents.len().min(MAX_CHECKPOINTS);
        let cadence = current.with_divisions(divisions);
        Self {
            provider_id,
            provider_label,
            step: Step::Divisions,
            divisions,
            focus: divisions,
            cadence,
            app_event_tx,
            completion: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn cadence(&self) -> &CompactionCadence {
        &self.cadence
    }

    fn set_divisions(&mut self, divisions: usize) {
        let divisions = divisions.min(MAX_CHECKPOINTS);
        self.cadence = self.cadence.with_divisions(divisions);
        self.divisions = divisions;
        self.focus = self.focus.min(divisions);
    }

    /// Moves the focused handle by `delta`, clamped so values stay strictly ascending and the
    /// trigger stays within 1–100.
    fn nudge(&mut self, delta: i16) {
        let mut values = self.cadence.values();
        let index = self.focus.min(values.len() - 1);
        let lower = if index == 0 {
            1
        } else {
            values[index - 1].saturating_add(1)
        };
        let upper = if index + 1 < values.len() {
            values[index + 1].saturating_sub(1)
        } else {
            100
        };
        let next = (i16::from(values[index]) + delta).clamp(i16::from(lower), i16::from(upper));
        values[index] = next as u8;
        self.cadence.trigger_percent = values.pop().unwrap_or(self.cadence.trigger_percent);
        self.cadence.checkpoint_percents = values;
    }

    fn accept(&mut self) {
        self.app_event_tx.send(AppEvent::PersistCompactionSettings {
            provider_id: self.provider_id.clone(),
            checkpoint_percents: Some(self.cadence.checkpoint_percents.clone()),
            trigger_percent: Some(self.cadence.trigger_percent),
        });
        self.completion = Some(ViewCompletion::Accepted);
    }

    fn handle_divisions_key(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('-') => {
                self.set_divisions(self.divisions.saturating_sub(1));
            }
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('+') | KeyCode::Char('=') => {
                self.set_divisions(self.divisions + 1);
            }
            KeyCode::Char(digit @ '0'..='4') => {
                self.set_divisions(digit as usize - '0' as usize);
            }
            KeyCode::Enter | KeyCode::Tab | KeyCode::Down => {
                self.step = Step::Sliders;
                self.focus = 0;
            }
            KeyCode::Esc => self.completion = Some(ViewCompletion::Cancelled),
            _ => {}
        }
    }

    fn handle_sliders_key(&mut self, key_event: KeyEvent) {
        let large = key_event.modifiers.contains(KeyModifiers::SHIFT);
        let step = i16::from(if large { LARGE_STEP } else { SMALL_STEP });
        let handles = self.divisions + 1;
        match key_event.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.focus == 0 {
                    self.step = Step::Divisions;
                } else {
                    self.focus -= 1;
                }
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => {
                self.focus = (self.focus + 1).min(handles - 1);
            }
            KeyCode::Left | KeyCode::Char('h') => self.nudge(-step),
            KeyCode::Right | KeyCode::Char('l') => self.nudge(step),
            KeyCode::PageDown => self.nudge(-i16::from(LARGE_STEP)),
            KeyCode::PageUp => self.nudge(i16::from(LARGE_STEP)),
            KeyCode::Home => self.nudge(-100),
            KeyCode::End => self.nudge(100),
            KeyCode::Backspace => self.step = Step::Divisions,
            KeyCode::Enter => self.accept(),
            KeyCode::Esc => self.completion = Some(ViewCompletion::Cancelled),
            _ => {}
        }
    }

    fn slider_line(&self, label: &str, value: u8, width: u16, focused: bool) -> Line<'static> {
        let label = format!("{label:<13}");
        let value_text = format!(" {value:>3}%");
        let bar_width = usize::from(width)
            .saturating_sub(label.len() + value_text.len() + 4)
            .clamp(10, 60);
        let filled = bar_width * usize::from(value) / 100;
        let mut spans: Vec<Span<'static>> = Vec::new();
        spans.push(if focused { "› ".into() } else { "  ".into() });
        spans.push(if focused { label.bold() } else { label.dim() });
        spans.push("▕".dim());
        spans.push(if focused {
            "█".repeat(filled).cyan()
        } else {
            "█".repeat(filled).dim()
        });
        spans.push("░".repeat(bar_width - filled).dim());
        spans.push("▏".dim());
        spans.push(if focused {
            value_text.bold()
        } else {
            value_text.dim()
        });
        Line::from(spans)
    }

    fn lines(&self, width: u16) -> Vec<Line<'static>> {
        let mut lines = vec![
            Line::from(vec![
                "Compaction cadence".bold(),
                " — ".dim(),
                self.provider_label.clone().into(),
            ]),
            Line::from(""),
        ];
        let divisions_focused = self.step == Step::Divisions;
        let count = format!(" {} ", self.divisions);
        lines.push(Line::from(vec![
            if divisions_focused { "› " } else { "  " }.into(),
            if divisions_focused {
                "Checkpoints  ".bold()
            } else {
                "Checkpoints  ".dim()
            },
            "◂".dim(),
            if divisions_focused {
                count.bold()
            } else {
                count.dim()
            },
            "▸".dim(),
            format!(
                "   (0–{MAX_CHECKPOINTS}; each stage summarizes what came after the previous one)"
            )
            .dim(),
        ]));
        lines.push(Line::from(""));
        let values = self.cadence.values();
        for (index, value) in values.iter().enumerate() {
            let focused = self.step == Step::Sliders && index == self.focus;
            let label = if index < self.divisions {
                format!("checkpoint {}", index + 1)
            } else {
                "compaction".to_string()
            };
            lines.push(self.slider_line(&label, *value, width, focused));
        }
        lines.push(Line::from(""));
        let hint = match self.step {
            Step::Divisions => Line::from(vec![
                key_hint::plain(KeyCode::Left).into(),
                key_hint::plain(KeyCode::Right).into(),
                " count  ".dim(),
                key_hint::plain(KeyCode::Enter).into(),
                " set percentages  ".dim(),
                key_hint::plain(KeyCode::Esc).into(),
                " cancel".dim(),
            ]),
            Step::Sliders => Line::from(vec![
                key_hint::plain(KeyCode::Up).into(),
                key_hint::plain(KeyCode::Down).into(),
                " select  ".dim(),
                key_hint::plain(KeyCode::Left).into(),
                key_hint::plain(KeyCode::Right).into(),
                " ±1  ".dim(),
                key_hint::shift(KeyCode::Left).into(),
                " ".into(),
                key_hint::shift(KeyCode::Right).into(),
                " ±5  ".dim(),
                key_hint::plain(KeyCode::Enter).into(),
                " save  ".dim(),
                key_hint::plain(KeyCode::Esc).into(),
                " cancel".dim(),
            ]),
        };
        lines.push(hint);
        lines
    }
}

impl Renderable for CompactionCadenceView {
    fn desired_height(&self, width: u16) -> u16 {
        self.lines(width).len() as u16 + menu_surface_padding_height()
    }

    fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        let inner = render_menu_surface(area, buf);
        Paragraph::new(self.lines(inner.width)).render(inner, buf);
    }
}

impl BottomPaneView for CompactionCadenceView {
    fn keymap_contexts(&self) -> KeymapContextSet {
        KeymapContextSet::new(KeymapContext::List)
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match self.step {
            Step::Divisions => self.handle_divisions_key(key_event),
            Step::Sliders => self.handle_sliders_key(key_event),
        }
    }

    fn on_ctrl_c(&mut self) -> CancellationEvent {
        self.completion = Some(ViewCompletion::Cancelled);
        CancellationEvent::Handled
    }

    fn is_complete(&self) -> bool {
        self.completion.is_some()
    }

    fn completion(&self) -> Option<ViewCompletion> {
        self.completion
    }
}

#[cfg(test)]
#[path = "compaction_cadence_view_tests.rs"]
mod tests;
