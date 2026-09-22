use super::*;
use crate::app_event::AppEvent;
use pretty_assertions::assert_eq;
use tokio::sync::mpsc::UnboundedReceiver;

fn make_view(
    percents: Vec<u8>,
    trigger: u8,
) -> (CompactionCadenceView, UnboundedReceiver<AppEvent>) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let view = CompactionCadenceView::new(
        Some("sparkops-qwen-dgx3".to_string()),
        "SparkOps Qwen3.8 on DGX3+DGX4 (http://10.10.10.103:8078/v1)".to_string(),
        CompactionCadence {
            checkpoint_percents: percents,
            trigger_percent: trigger,
        },
        AppEventSender::new(tx),
    );
    (view, rx)
}

fn rows(view: &CompactionCadenceView, width: u16) -> String {
    let height = view.desired_height(width);
    let area = Rect::new(/*x*/ 0, /*y*/ 0, width, height);
    let mut buf = Buffer::empty(area);
    view.render(area, &mut buf);
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn renders_divisions_then_sliders() {
    let (mut view, _rx) = make_view(vec![50], 80);
    insta::assert_snapshot!("cadence_divisions", rows(&view, 80));
    view.handle_key_event(KeyEvent::from(KeyCode::Right));
    view.handle_key_event(KeyEvent::from(KeyCode::Enter));
    insta::assert_snapshot!("cadence_sliders", rows(&view, 80));
}

#[test]
fn changing_divisions_spaces_new_stages_evenly() {
    let (mut view, _rx) = make_view(vec![50], 80);
    view.handle_key_event(KeyEvent::from(KeyCode::Right));
    assert_eq!(
        view.cadence(),
        &CompactionCadence {
            checkpoint_percents: vec![26, 53],
            trigger_percent: 80
        }
    );
    view.handle_key_event(KeyEvent::from(KeyCode::Char('0')));
    assert_eq!(view.cadence().checkpoint_percents, Vec::<u8>::new());
}

#[test]
fn sliders_stay_strictly_ascending() {
    let (mut view, _rx) = make_view(vec![50, 65], 80);
    view.handle_key_event(KeyEvent::from(KeyCode::Enter));
    // Push checkpoint 1 up past checkpoint 2: it stops one below.
    for _ in 0..40 {
        view.handle_key_event(KeyEvent::from(KeyCode::Right));
    }
    assert_eq!(view.cadence().checkpoint_percents, vec![64, 65]);
    // The trigger cannot exceed 100 and cannot drop below the last checkpoint.
    view.handle_key_event(KeyEvent::from(KeyCode::Down));
    view.handle_key_event(KeyEvent::from(KeyCode::Down));
    view.handle_key_event(KeyEvent::from(KeyCode::End));
    assert_eq!(view.cadence().trigger_percent, 100);
    view.handle_key_event(KeyEvent::from(KeyCode::Home));
    assert_eq!(view.cadence().trigger_percent, 66);
    view.handle_key_event(KeyEvent::new(KeyCode::Right, KeyModifiers::SHIFT));
    assert_eq!(view.cadence().trigger_percent, 71);
}

#[test]
fn enter_persists_and_esc_cancels_without_writing() {
    let (mut view, mut rx) = make_view(vec![50], 80);
    view.handle_key_event(KeyEvent::from(KeyCode::Esc));
    assert_eq!(view.completion(), Some(ViewCompletion::Cancelled));
    assert!(rx.try_recv().is_err());

    let (mut view, mut rx) = make_view(vec![50], 80);
    view.handle_key_event(KeyEvent::from(KeyCode::Enter));
    view.handle_key_event(KeyEvent::from(KeyCode::Down));
    view.handle_key_event(KeyEvent::new(KeyCode::Right, KeyModifiers::SHIFT));
    view.handle_key_event(KeyEvent::from(KeyCode::Enter));
    assert_eq!(view.completion(), Some(ViewCompletion::Accepted));
    match rx.try_recv() {
        Ok(AppEvent::PersistCompactionSettings {
            provider_id,
            checkpoint_percents,
            trigger_percent,
        }) => {
            assert_eq!(provider_id.as_deref(), Some("sparkops-qwen-dgx3"));
            assert_eq!(checkpoint_percents, Some(vec![50]));
            assert_eq!(trigger_percent, Some(85));
        }
        other => panic!("unexpected event: {other:?}"),
    }
}
