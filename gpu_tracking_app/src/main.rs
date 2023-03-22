use std::path::PathBuf;
use eframe;
use gpu_tracking_app;
use tracing_subscriber;
use clap::Parser;

#[derive(Parser, Debug)]
struct Args{
    paths: Vec<String>,
    #[arg(short, long)]
    verbosity: Option<u32>,

    #[arg(short, long)]
    test: Option<bool>,
}

fn main() {
    let Args{
        paths,
        verbosity,
        test,
    } = Args::parse();
    let verbosity = verbosity.unwrap_or(0);
    let test = test.unwrap_or(false);

    if verbosity > 0{
        tracing_subscriber::fmt()
            .with_max_level(tracing_subscriber::filter::LevelFilter::ERROR)
            .with_thread_ids(true).init();
    }

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
        Box::new(move |cc| {
            if !test{
                Box::new(gpu_tracking_app::custom3d_wgpu::AppWrapper::new(cc, paths).unwrap())
            } else {
                Box::new(gpu_tracking_app::custom3d_wgpu::AppWrapper::test(cc).unwrap())
            }
        }),
    ).unwrap();
}
