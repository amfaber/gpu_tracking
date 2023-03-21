// use egui;
use eframe;
use gpu_tracking_app;
use tracing_subscriber;
fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::ERROR)
        .with_thread_ids(true).init();
    let options = eframe::NativeOptions {
        drag_and_drop_support: true,
        // maximized: true,

        initial_window_size: Some([1200., 1000.].into()),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "gpu_tracking",
        options,
        Box::new(|cc| Box::new(gpu_tracking_app::custom3d_wgpu::AppWrapper::test(cc).unwrap())),
    ).unwrap();
}
