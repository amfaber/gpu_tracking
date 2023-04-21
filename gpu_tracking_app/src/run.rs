use std::path::PathBuf;

use clap::Parser;
use eframe;
use tracing_subscriber;


#[derive(Parser, Debug)]
struct Args {
    paths: Vec<String>,
    #[arg(short, long)]
    verbosity: Option<u32>,

    #[arg(short, long)]
    test: Option<bool>,

    #[arg(short, long)]
    help: bool,
}

#[derive(Parser, Debug)]
struct IgnoreFirstArgs {
	exename: String,
	
    paths: Vec<String>,
    #[arg(short, long)]
    verbosity: Option<u32>,

    #[arg(short, long)]
    test: Option<bool>,
    
    #[arg(short, long)]
    help: bool,
}

impl Into<Args> for IgnoreFirstArgs{
    fn into(self) -> Args {
        let Self{
			paths,
			verbosity,
			test,
			help,
            exename: _exename,
		} = self;
		Args{
			paths,
			verbosity,
			test,
            help,
		}
    }
}

pub fn run_python(doc_dir: PathBuf){
	run(IgnoreFirstArgs::parse().into(), Some(doc_dir))
}

pub fn run_all(){
	run(Args::parse(), None)
}

fn run(args: Args, doc_dir: Option<PathBuf>) {
    let Args {
        paths,
        verbosity,
        test,
        help,
    } = args;

    if help{
        // dbg!(std::env::current_dir());
        dbg!(std::env::current_exe());
        return
    }
    
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
                Box::new(crate::app::AppWrapper::new(cc, paths, doc_dir).unwrap())
            } else {
                Box::new(crate::app::AppWrapper::test(cc).unwrap())
            }
        }),
    )
    .unwrap();
}
