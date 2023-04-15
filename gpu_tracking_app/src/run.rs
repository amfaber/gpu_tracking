use clap::Parser;
use eframe;
use tracing_subscriber;
use com::runtime::{ApartmentType, init_runtime};


#[derive(Parser, Debug)]
struct Args {
    paths: Vec<String>,
    #[arg(short, long)]
    verbosity: Option<u32>,

    #[arg(short, long)]
    test: Option<bool>,
}

#[derive(Parser, Debug)]
struct IgnoreFirstArgs {
	exename: String,
	
    paths: Vec<String>,
    #[arg(short, long)]
    verbosity: Option<u32>,

    #[arg(short, long)]
    test: Option<bool>,
}

impl Into<Args> for IgnoreFirstArgs{
    fn into(self) -> Args {
        let Self{
			paths,
			verbosity,
			test,
			..
		} = self;
		Args{
			paths,
			verbosity,
			test,
		}
    }
}

pub fn run_ignore(){
	run(IgnoreFirstArgs::parse().into())
}

pub fn run_all(){
	run(Args::parse())
}

fn run(args: Args) {
    let Args {
        paths,
        verbosity,
        test,
    } = args;
    let verbosity = verbosity.unwrap_or(0);
    let test = test.unwrap_or(false);
    
    if verbosity > 0 {
        tracing_subscriber::fmt()
            .with_max_level(tracing_subscriber::filter::LevelFilter::ERROR)
            .with_thread_ids(true)
            .init();
    }

    let options = eframe::NativeOptions {
        drag_and_drop_support: true,
        initial_window_size: Some([1200., 800.].into()),
        renderer: eframe::Renderer::Wgpu,

        ..Default::default()
    };
    eframe::run_native(
        "gpu_tracking",
        options,
        Box::new(move |cc| {
            if !test {
                Box::new(crate::app::AppWrapper::new(cc, paths).unwrap())
            } else {
                Box::new(crate::app::AppWrapper::test(cc).unwrap())
            }
        }),
    )
    .unwrap();
}
