use proc_macro2::Ident;
use proc_macro::TokenStream;
use quote::{quote, format_ident};
use InputType::*;
use Style::*;
use strum::{EnumIter, IntoEnumIterator};

type TokenStream2 = proc_macro2::TokenStream;

#[derive(Clone, Copy, EnumIter, Debug, PartialEq)]
enum InputType{
	File,
	Array,
}

#[derive(Clone, Copy, EnumIter, Debug, PartialEq)]
enum Style{
	Trackpy,
	LoG,
	Characterize,
}

fn make_func(inp: InputType, style: Style) -> TokenStream2{
	let name = name(inp, style);
	let ret = return_type();
	let bod = body(inp, style);
	let all_args = args(inp, style);
	let out = quote!(
		#[pyfunction]
		fn #name<'py>(#all_args) -> #ret{
			#bod
		}
	);
	out
}

fn args(inp: InputType, style: Style) -> TokenStream2{
	let preargs = match inp{
		File => file_args(),
		Array => array_args(),
	};

	let midargs = match style{
		Trackpy => tp_args(),
		LoG => log_args(),
		Characterize => char_args(),
	};
	let postargs = match inp{
		File => file_post_args(),
		Array => array_post_args(),
	};
	
	quote!(
		#preargs
		#midargs
		#postargs
	)
}
fn body(inp: InputType, style: Style) -> TokenStream2{
	let file_prelude = match inp{
		Array => quote!(
	        let array = pyarr.as_array();
		),
		File => TokenStream2::new(),
	};
	
	let style_prelude = match style{
		Trackpy => parse_tp(),
		LoG => parse_log(),
		Characterize => {
			parse_char()
		}
	};


	let func_name = match inp{
		File => quote!(execute_file),
		Array => quote!(execute_ndarray),
	};
	let argdiff = match inp{
		File => quote!(filename, channel),
		Array => quote!(array),
	};

	let execution = quote!(
	    let res = if tqdm{
	        std::thread::scope(|scope|{
	            let mut worker = ScopedProgressFuture::new(scope, |job, progress, interrupt|{
	                let (#argdiff, params, characterize_points) = job;
	                #func_name(#argdiff, params, 0, characterize_points, Some(interrupt), Some(progress))
	            });
	            worker.submit_same((&#argdiff, params, characterize_points));
	            Python::with_gil(|py|{
	                let tqdm = PyModule::import(py, "tqdm")?;
	                let tqdm_func = tqdm.getattr("tqdm")?;
	                let mut pbar: Option<&PyAny> = None;
		            let mut ctx_pbar: Option<&PyAny> = None;
	                let res = loop{
		                std::thread::sleep(std::time::Duration::from_millis(50));
	                    match py.check_signals(){
	                        Ok(()) => (),
	                        Err(e) => {
	                            worker.interrupt();
	                            return Err(e)
	                        }
	                    }
	                    match worker.poll().map_err(|err| err.pyerr())?{
	                        PollResult::Done(res) => break res.map_err(|err| err.pyerr())?,
	                        PollResult::Pending((cur, total)) => {
	                            if let Some(ictx_pbar) = ctx_pbar{
	                                ictx_pbar.setattr("n", cur)?;
	                                ictx_pbar.setattr("total", total)?;
	                                ictx_pbar.call_method0("refresh")?;
	                            } else {
	                                let kwargs = [("total", total)].into_py_dict(py);
	                                let inner_pbar = tqdm_func.call((), Some(kwargs))?;
	                                ctx_pbar = Some(inner_pbar.call_method0("__enter__")?);
	                                pbar = Some(inner_pbar);
	                            }
	                        },
	                        PollResult::NoJobRunning => return Err(Error::ThreadError.pyerr()),
	                    }
	                };
	                if let Some(ctx_pbar) = ctx_pbar{
						let last = worker.read_progress().map_err(|err| err.pyerr())?;
						ctx_pbar.setattr("n", last.0)?;
	                    ctx_pbar.call_method("__exit__", (None::<i32>, None::<i32>, None::<i32>), None);
	                }
	                Ok(res)
	            })
	        })?
	    } else {
	        let res = #func_name(&#argdiff, params, 0, None, None, None).map_err(|err| err.pyerr())?;
	        res
	    };
	    let res = (res.0.into_pyarray(py), res.1.into_py(py));
	    Ok(res)
	);
	
	quote!(
		#file_prelude
		#style_prelude
		#execution
	)
}

fn parse_char() -> TokenStream2{
	let parse = parse_tp();
	quote!(
		#parse
        params.characterize = true;
        let characterize_points = Some((points_to_characterize.as_array(), points_has_frames, points_has_r));
        if points_has_r{
            params.include_r_in_output = true;
        }
        if let ParamStyle::Trackpy{ ref mut filter_close, .. } = params.style{
            *filter_close = false;
        }
	)
}

fn name(inp: InputType, style: Style) -> Ident{
	let mut out = String::new();
	out.push_str(match style{
		Trackpy => "batch",
		LoG => "log",
		Characterize => "characterize",
	});
	out.push_str(match inp{
		File => "_file",
		Array => "",
	});
	out.push_str("_rust");
	format_ident!("{}", out)
}


fn tp_args() -> TokenStream2{
	let common_args = common_args();
	quote!(
        diameter: u32,
        maxsize: Option<my_dtype>,
        separation: Option<u32>,
        threshold: Option<my_dtype>,
        invert: Option<bool>,
        percentile: Option<my_dtype>,
        topn: Option<u32>,
        preprocess: Option<bool>,
        filter_close: Option<bool>,
		#common_args
	)
}

fn log_args() -> TokenStream2{
	let common_args = common_args();
	quote!(
        min_radius: my_dtype,
        max_radius: my_dtype,
        n_radii: Option<usize>,
        log_spacing: Option<bool>,
        overlap_threshold: Option<my_dtype>,
		#common_args
	)
}

fn char_args() -> TokenStream2{
	let tp_args = tp_args();
	quote!(
        points_to_characterize: PyReadonlyArray2<my_dtype>,
        points_has_frames: bool,
        points_has_r: bool,
		#tp_args
	)
}

fn common_args() -> TokenStream2{
	quote!(
        noise_size: Option<my_dtype>,
        smoothing_size: Option<u32>,
        minmass: Option<my_dtype>,
        max_iterations: Option<u32>,
        characterize: Option<bool>,
        search_range: Option<my_dtype>,
        memory: Option<usize>,
        doughnut_correction: Option<bool>,
        bg_radius: Option<my_dtype>,
        gap_radius: Option<my_dtype>,
        snr: Option<my_dtype>,
        minmass_snr: Option<my_dtype>,
        truncate_preprocessed: Option<bool>,
        correct_illumination: Option<bool>,
        illumination_sigma: Option<my_dtype>,
        adaptive_background: Option<usize>,
        shift_threshold: Option<my_dtype>,
        linker_reset_points: Option<Vec<usize>>,
        keys: Option<Vec<usize>>,
        illumination_correction_per_frame: Option<bool>,
		tqdm: Option<bool>,
	)
}

fn parse_tp() -> TokenStream2{
	let common = parse_common();
	quote!(
        let maxsize = maxsize.unwrap_or(f32::INFINITY);
        let separation = separation.unwrap_or(diameter + 1);
        let threshold = threshold.unwrap_or(1./255.);
        let invert = invert.unwrap_or(false);
        let percentile = percentile.unwrap_or(64.);
        let topn = topn.unwrap_or(u32::MAX);
        let preprocess = preprocess.unwrap_or(true);
        let filter_close = filter_close.unwrap_or(true);
		let style = ParamStyle::Trackpy{
            diameter,
            maxsize,
            separation,
            threshold,
            invert,
            percentile,
            topn,
            preprocess,
            filter_close,
        };
		let include_r_in_output = false;
		#common
		
	)
}

fn parse_log() -> TokenStream2{
	let common = parse_common();
	quote!(
        let n_radii = n_radii.unwrap_or(10);
        let log_spacing = log_spacing.unwrap_or(false);
        let overlap_threshold = overlap_threshold.unwrap_or(0.);
		let style = ParamStyle::Log{
	        min_radius,
	        max_radius,
	        n_radii,
	        log_spacing,
	        overlap_threshold,
	    };
		let include_r_in_output = true;
		#common
	)
}

fn parse_common() -> TokenStream2{
	quote!(
        let minmass = minmass.unwrap_or(0.);
        let max_iterations = max_iterations.unwrap_or(10);
        let characterize = characterize.unwrap_or(false);
        let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));
        let truncate_preprocessed = truncate_preprocessed.unwrap_or(true);
        let shift_threshold = shift_threshold.unwrap_or(0.6);
        let noise_size = noise_size.unwrap_or(1.0);
		
        let doughnut_correction = doughnut_correction.unwrap_or(false);
        let illumination_correction_per_frame = illumination_correction_per_frame.unwrap_or(false);

        let illumination_sigma = match illumination_sigma{
            Some(val) => Some(val),
            None => {
                if correct_illumination.unwrap_or(false){
                    Some(30.)
                } else {
                    None
                }
            }
        };
		let tqdm = tqdm.unwrap_or(true);

		let mut characterize_points = None::<(ArrayView2<my_dtype>, bool, bool)>;
		
		let mut params = TrackingParams{
			style,
            minmass,
            max_iterations,
            characterize,
            search_range,
            memory,
            doughnut_correction,
            bg_radius,
            gap_radius,
            snr,
            minmass_snr,
            truncate_preprocessed,
            illumination_sigma,
            adaptive_background,
            include_r_in_output,
            shift_threshold,
            linker_reset_points,
            keys,
            noise_size,
            smoothing_size,
            illumination_correction_per_frame,
		};
	)
}

fn file_args() -> TokenStream2{
	quote!(
        py: Python<'py>,
        filename: String,
	)
}

fn file_post_args() -> TokenStream2{
	quote!(
		channel: Option<usize>,
	)
}

fn array_args() -> TokenStream2{
	quote!(
        py: Python<'py>,
        pyarr: PyReadonlyArray3<my_dtype>,
	)
}

fn array_post_args() -> TokenStream2{
	TokenStream2::new()
}

fn return_type() -> TokenStream2{
	quote!(
	    PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)>	
	)
}


#[proc_macro]
pub fn gen_python_functions(_item: TokenStream) -> TokenStream{
	let mut out = TokenStream2::new();
	for s in Style::iter(){
		for inp in InputType::iter(){
			let this_func = make_func(inp, s);
			// if s == Style::Characterize{
			// 	println!("{}\n\n\n\n\n\n\n\n", this_func);
			// }
			out.extend(this_func);
		}
	}
	out.into()
}
