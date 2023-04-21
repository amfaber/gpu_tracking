use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{format_ident, quote};
use strum::{EnumIter, IntoEnumIterator};
use std::io::Write;
use Style::*;

struct Argument{
    func: TokenStream2,
    init: TokenStream2,
    docs: String,
    sorter: i32,
}

type TokenStream2 = proc_macro2::TokenStream;

#[derive(Clone, Copy, EnumIter, Debug, PartialEq)]
enum Style {
    Trackpy,
    LoG,
    Characterize,
}

fn make_func(style: Style) -> (TokenStream2, syn::Ident, Vec<Argument>) {
    let name = name(style);
    let ret = return_type();
    let (all_args, init) = args(style);
    let bod = body(&all_args, init);
    let arg_stream: TokenStream2 = all_args.iter().map(|arg| arg.func.clone()).collect();
    let out = quote!(
        #[pyfunction]
        fn #name<'py>(#arg_stream) -> #ret{
            #bod
        }
    );
    (out, name, all_args)
}

fn args(style: Style) -> (Vec<Argument>, TokenStream2) {

    let (mut args, init) = match style {
        Trackpy => tp_args(),
        LoG => log_args(),
        Characterize => char_args(),
    };

    args.sort_by(|a, b| a.sorter.cmp(&b.sorter));
    (args, init)

    // quote!(
    //     py: Python<'py>,
    //     file_or_array: &PyAny,
    //     #midargs
    //     channel: Option<usize>,
    // )
}
fn body(args: &Vec<Argument>, init: TokenStream2) -> TokenStream2 {
    // let style_prelude = match style {
    //     Trackpy => parse_tp(),
    //     LoG => parse_log(),
    //     Characterize => parse_char(),
    // };
    let mut full_init: TokenStream2 = args.iter().map(|arg| arg.init.clone()).collect();

    full_init.extend(init);
    // full_init = quote!(
    //     #full_init
    //     #init
    // );

    let execution = quote!(
        let device_queue = new_device_queue(&new_adapter());
        let foa_owned = OwnedFileOrArray::new(py, file_or_array)?;
        let foa_ref = foa_owned.borrow();
        let res = if tqdm{
            std::thread::scope(|scope|{
                let mut worker = ScopedProgressFuture::new(scope, move |job, progress, interrupt|{
                    let (sent_file_or_array, channel, params, characterize_points): (FileOrArray<String>, Option<usize>, TrackingParams, Option<(ArrayView2<my_dtype>, bool, bool)>) = job;
                    CommandBuilder::new()
                        .set_file_or_array(sent_file_or_array)
                        .set_interruption(Some(interrupt), Some(progress))
                        .set_rest(params, 0, characterize_points, &device_queue)
                        .execute()
                });
                worker.submit_same((
                    foa_ref,
                    channel,
                    params,
                    characterize_points,
                ));
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
                            PollResult::Done(res) => break res,
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
                    Ok(res.map_err(|e| e.pyerr())?)
                })
            })?
        } else {
            let res = CommandBuilder::new()
                .set_file_or_array(foa_ref)
                .set_rest(params, 0, characterize_points, &device_queue)
                .execute()
                .map_err(|e| e.pyerr())?;
            res
        };
        let (np_arr, types) = (res.0.into_pyarray(py), res.1);
        let pandas = PANDAS.as_ref(py);
        let column_names: Vec<_> = types.iter().cloned().map(|(name, ty)| name).collect();
        let column_types: HashMap<&str, &str> = types.iter().cloned().collect();
        let kwargs = HashMap::from([
            ("columns", column_names.into_py(py)),
        ]);
        let df_func = pandas.getattr("DataFrame")?;
        let df = df_func.call((np_arr,), Some(kwargs.into_py_dict(py)))?;
        let df = df.call_method1("astype", (column_types, ));
        
        df
    );

    quote!(
        #full_init
        #execution
    )
}

// fn parse_char() -> TokenStream2 {
//     let parse = parse_tp();
//     quote!(
//         #parse
//         params.characterize = true;
//         let characterize_points = Some((points_to_characterize.as_array(), points_has_frames, points_has_r));
//         if points_has_r{
//             params.include_r_in_output = true;
//         }
//         if let ParamStyle::Trackpy{ ref mut filter_close, .. } = params.style{
//             *filter_close = false;
//         }
//     )
// }

fn name(style: Style) -> Ident {
    let mut out = String::new();
    out.push_str(match style {
        Trackpy => "batch_rust",
        LoG => "LoG_rust",
        Characterize => "characterize_points_rust",
    });
    format_ident!("{}", out)
}

fn fmt(s: &str) -> String{
    s.replace("\n", "").replace("\t", "")
}

fn common_args() -> (Vec<Argument>, TokenStream2) {
    let args = vec![
        Argument{
            func: quote!(py: Python<'py>,),
            init: quote!(),
            docs: fmt(""),
            sorter: -100,
        },
        Argument{
            func: quote!(file_or_array: &PyAny,),
            init: quote!(),
            docs: fmt(
                "The input video. Either a file-path to where the video can be found (tiff, vsi or ets format), or a numpy array with shape
                (T, Y, X), T corresponding to the time dimension, and Y and X corresponding to y and x in the output dataframe"
            ),
            sorter: -50,
        },
        Argument{
            func: quote!(channel: Option<usize>,),
            init: quote!(),
            docs: fmt("In case a .vsi / .ets video is supplied, this channel will be used from the video. Defaults to 0"),
            sorter: 50,
        },
        Argument{
            func: quote!(noise_size: Option<my_dtype>,),
            init: quote!(let noise_size = noise_size.unwrap_or(1.0);),
            docs: fmt(
                "The sigma of a gaussian smoothing that is applied to the video during preprocessing. Defaults to 1.0.
                Was introduced by Crocker-Grier to counteract some digitalization noise of CCD cameras"
            ),
            sorter: 0,
        },
        Argument{
            func: quote!(smoothing_size: Option<u32>,),
            init: quote!(),
            docs: fmt(
                r#"The side-length of the uniform convolution that subtracts local background during preprocessing. Defaults to diameter.
                Setting this too low and close to the actual size of the signal runs the risk of supressing the signal, as most of the "local background"
                will be the signal itself. As such, it can be beneficial to set this higher than the diameter in some cases."# 
            ),
            sorter: 0,
        },
        Argument{
            func: quote!(minmass: Option<my_dtype>,),
            init: quote!(let minmass = minmass.unwrap_or(0.);),
            docs: fmt(r#"
                The minimum integrated intensity a particle has to have to be considered a true particle, and not a spurious
                detection. Defaults to 0. This setting provides the same functionality as "minmass_snr", only the threshold
                is set as an absolute number rather than being relative to the video's noise level. "minmass_snr" should be
                preferred in most cases."#
            ),
            sorter: 0,
        },
        Argument{
            func: quote!(max_iterations: Option<u32>,),
            init: quote!(let max_iterations = max_iterations.unwrap_or(10);),
            docs: fmt(r#"
                The maximum number of steps that the localization refinement algorithm is allowed to take. Defaults to 10.
                Increasing this number is unlikely to increase localization accuracy, as it just allows each detection to
                move further away from the local maximum that seeded the algorithm.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(characterize: Option<bool>,),
            init: quote!(let characterize = characterize.unwrap_or(true);),
            docs: fmt(r#"
                Whether to include the columns "Rg", "raw", "signal" and "ecc" in the output dataframe. Defaults to True.
                "Rg" is the radius of gyration of the detection. "raw" is the integrated intensity of the particle in the
                unprocessed (and not background subtracted) image. "signal" is the peak (i.e. maximum pixel value) of the
                signal in the preprocessed image. "ecc" is the particle's eccentricity. These can be helpful measures for
                further processing beyond gpu_tracking.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(search_range: Option<my_dtype>,),
            init: quote!(),
            docs: fmt(r#"
                The search range in pixel space used for linking the particles through time. Defaults to None, meaning
                that linking won't be performed. The linking algorithm is identical to Trackpy's linking, and does
                optimal (as opposed to greedy) nearest neighbor linking.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(memory: Option<usize>,),
            init: quote!(),
            docs: fmt(r#"
                The number of frames that a particle is allowed to disappear before reappearing, for the purposes of linking.
                memory = 5 means that a particle can be gone for 5 frames, reappearing in the 6th and still be linked to the
                track it had built previously. If it had reappeared in the 7th, it would instead have been considered a new
                particle. Defaults to 0, and has no effect if search_range is not set.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(doughnut_correction: Option<bool>,),
            init: quote!(let doughnut_correction = doughnut_correction.unwrap_or(true);),
            docs: fmt(r#"
                Whether to include the columns "raw_mass", "raw_bg_median" and "raw_mass_corrected" in the output dataframe. Like "raw"
                from characterize, "raw_mass" is the integrated intensity of the particle in the raw input image. "raw_bg_median" is the
                median of the background around the particle in a hollow "doughnut" of outer radius "bg_radius", inner radius "diameter / 2
                + gap_radius". "raw_mass_corrected" is "raw_mass" - "raw_bg_median" * #pixels_in_particle. "raw_mass_corrected" is generally
                the most accurate measure we have of particle intensities.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(bg_radius: Option<my_dtype>,),
            init: quote!(),
            docs: fmt(r#"
                The radius to use for "doughnut_correction". Defaults to "diameter", i.e. twice the particles radius.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(gap_radius: Option<my_dtype>,),
            init: quote!(let gap_radius = bg_radius.map(|_| gap_radius.unwrap_or(0.));),
            docs: fmt(r#"
                An optional amount of gap between the particle and background measurement for "doughnut_correction". Defaults to 0.0.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(snr: Option<my_dtype>,),
            init: quote!(),
            docs: fmt(r#"
                Primary filter for what to consider a particle and what is simply suprious detections. Defaults to 0. "snr" measures the
                noise level of the video by taking the standard deviation of each frame individually and taking this to be the global
                noise level. The peak of the particle in the proprocessed image must then be above [noise_level] * [snr] to be considered
                a particle. Videos with a very non-uniform background cause trouble for this setting, as the global noise level will be
                artificially inflated, necessitating setting a lower "snr". This setting is a useful heuristic for setting comparable
                thresholds across quite different videos, but should not be interpreted as a strict filter for only getting particles above
                the set snr level.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(minmass_snr: Option<my_dtype>,),
            init: quote!(),
            docs: fmt(r#"
                Serves the same role as "snr", except where "snr" filters on the particle's peak signal, "snr_minmass" filters on the particle's
                integrated intensity, potentially squashing random high "lone peaks". Defaults to 0.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(truncate_preprocessed: Option<bool>,),
            init: quote!(let truncate_preprocessed = truncate_preprocessed.unwrap_or(true);),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(correct_illumination: Option<bool>,),
            init: quote!(),
            docs: fmt(r#"
                Whether to correct the illumination profile of the video before tracking. Defaults to False. This is done by smoothing the video with sigma=30 pixels
                and then dividing the raw video by the very smoothed video, leveling out any local differences in background. This can be helpful
                in the case of uneven illumination profiles, but also in other cases of uneven backgrounds.
                "#),
            sorter: 0,
        },
        
        Argument{
            func: quote!(illumination_sigma: Option<my_dtype>,),
            init: quote!(
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
            ),
            docs: fmt(r#"
                Same as "correct_illumination", except a sigma can be provided. Defaults to 30 if correction_illumination is True, and None otherwise, meaning that
                correction will not be done. If both are provided, "illumination_sigma" takes precedence.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(illumination_correction_per_frame: Option<bool>,),
            init: quote!(let illumination_correction_per_frame = illumination_correction_per_frame.unwrap_or(false);),
            docs: fmt(r#"
                When doing illumination correction, this setting controls whether to do the correction on a per-frame basis, or if the
                entire video should be loaded, averaged across frames, and then the resulting average frame is the only one that is smoothed
                and used to do the correction. Defaults to False, meaning that the video is loaded, averaged and smoothed before starting the actual
                detection algorithm.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(adaptive_background: Option<usize>,),
            init: quote!(),
            docs: fmt(r#"
                If "snr" or "minmass_snr" are provided, this setting allows the measurement of the global background noise level to be updated adaptively.
                Once a number of particles have been detected as being above the set "snr" and "minmass_snr", the pixels of these particles are removed
                from the raw image, and the global noise level is recalculated. Particles are then tested again if they are now below the thresholds of
                "snr" and "minmass_snr" with the updated noise level, and included if they now pass the check. This process can be repeated iteratively,
                and "adaptive_background" sets the number of times it is repeated. Defaults to None, meaning that the process is not run at all.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(shift_threshold: Option<my_dtype>,),
            init: quote!(let shift_threshold = shift_threshold.unwrap_or(0.6);),
            docs: fmt(r#"
                The threshold for stopping the localization refinement algorithm. Defaults to 0.6, and should never be below 0.5. Generally
                not recommended to change.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(linker_reset_points: Option<Vec<usize>>,),
            init: quote!(),
            docs: fmt(r#"
                A list of points at which the linking should be reset. Defaults to no reset points. Useful if a single video has points at
                which the recording was paused and then later resumed. Supplying these points to this option ensures that particles that are
                actually temporally very far from eachother aren't linked together.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(frames: Option<Vec<usize>>,),
            init: quote!(),
            docs: fmt(r#"
                A sequence that specifies what frames from the video to track. For example, to only load and track the first 50 frames of a video,
                frames = range(50) can be supplied. Be aware that all the frames in the sequence are assumed to be next to eachother in time, i.e.
                specifying frames = [0, 1, 2, 50, 100] will attempt to link the detections in frame 2 to those in frame 50, and those in frame 50 to
                those in frame 100. This can be further customized with "linker_reset_points". Defaults to tracking all supplied frames.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(tqdm: Option<bool>,),
            init: quote!(let tqdm = tqdm.unwrap_or(true);),
            docs: fmt(r#"Whether to use tqdm to report progress. Defaults to True"#),
            sorter: 0,
        },
    ];


    let additional_init = quote!(
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
            keys: frames,
            noise_size,
            smoothing_size,
            illumination_correction_per_frame,
        };
    );

    (args, additional_init)
    // quote!(
    //     noise_size: Option<my_dtype>,
    //     smoothing_size: Option<u32>,
    //     minmass: Option<my_dtype>,
    //     max_iterations: Option<u32>,
    //     characterize: Option<bool>,
    //     search_range: Option<my_dtype>,
    //     memory: Option<usize>,
    //     doughnut_correction: Option<bool>,
    //     bg_radius: Option<my_dtype>,
    //     gap_radius: Option<my_dtype>,
    //     snr: Option<my_dtype>,
    //     minmass_snr: Option<my_dtype>,
    //     truncate_preprocessed: Option<bool>,
    //     correct_illumination: Option<bool>,
    //     illumination_sigma: Option<my_dtype>,
    //     adaptive_background: Option<usize>,
    //     shift_threshold: Option<my_dtype>,
    //     linker_reset_points: Option<Vec<usize>>,
    //     keys: Option<Vec<usize>>,
    //     illumination_correction_per_frame: Option<bool>,
    //     tqdm: Option<bool>,
    // )
}

fn tp_args() -> (Vec<Argument>, TokenStream2) {
    let (common_args, common_init) = common_args();
    let mut tp_args = vec![
        Argument{
            func: quote!(diameter: u32,),
            init: quote!(),
            docs: fmt(r#"
                The diameter of the particles that are searched for. This is the only required parameter (except for in
                characterize_points), and so many other parameters default to values based on the diameter. On its own, it is the diameter
                of the circle within which all intensity integration calculations are done.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(maxsize: Option<my_dtype>,),
            init: quote!(let maxsize = maxsize.unwrap_or(f32::INFINITY);),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(separation: Option<u32>,),
            init: quote!(let separation = separation.unwrap_or(diameter + 1);),
            docs: fmt("The minimum separation between particles in pixels. Defaults to diameter + 1. This is used for the maximum filter
            in the Crocker-Grier algorithm, and for subsequently filtering detections that are too close to eachother."),
            sorter: 0,
        },
        Argument{
            func: quote!(threshold: Option<my_dtype>,),
            init: quote!(let threshold = threshold.unwrap_or(1./255.);),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(invert: Option<bool>,),
            init: quote!(let invert = invert.unwrap_or(false);),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(percentile: Option<my_dtype>,),
            init: quote!(let percentile = percentile.unwrap_or(64.);),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(topn: Option<u32>,),
            init: quote!(let topn = topn.unwrap_or(u32::MAX);),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(preprocess: Option<bool>,),
            init: quote!(let preprocess = preprocess.unwrap_or(true);),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(filter_close: Option<bool>,),
            init: quote!(let filter_close = filter_close.unwrap_or(true);),
            docs: fmt(r#"Whether to filter detections that are too close to eachother with "separation""#),
            sorter: 0,
        },
    ];
    let init = quote!(
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
        #common_init
    );
    
    
    tp_args.extend(common_args.into_iter());
    (tp_args, init)
}

fn log_args() -> (Vec<Argument>, TokenStream2) {
    let (common_args, common_init) = common_args();
    let mut log_args = vec![
        Argument{
            func: quote!(min_radius: my_dtype,),
            init: quote!(),
            docs: fmt(r#"The minimum radius of the radius scan of laplacian of the gaussian"#),
            sorter: 0,
        },
        Argument{
            func: quote!(max_radius: my_dtype,),
            init: quote!(),
            docs: fmt(r#"
                The maximum radius of the radius scan of laplacian of the gaussian. All the parameters that are set based on defaults of
                the "diameter" parameter in Trackpy style tracking instead use 2x max_radius when tracking with LoG. This can have severe
                consequences if the maximum radius is set very large without also adjusting parameters that are set based on this.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(n_radii: Option<usize>,),
            init: quote!(let n_radii = n_radii.unwrap_or(10);),
            docs: fmt("
                The number of radii in the radius scan. Defaults to 10. Execution time scales linearly with this setting,
                as expensive convolutions need to be calculated for each radius in the radius scan.
                "),
            sorter: 0,
        },
        Argument{
            func: quote!(log_spacing: Option<bool>,),
            init: quote!(let log_spacing = log_spacing.unwrap_or(false);),
            docs: fmt(r#"Whether to use logarithmic spacing in the radius scan. Defaults to false, using linear spacing."#),
            sorter: 0,
        },
        Argument{
            func: quote!(overlap_threshold: Option<my_dtype>,),
            init: quote!(let overlap_threshold = overlap_threshold.unwrap_or(0.);),
            docs: fmt(r#"
                The maximum allowed overlap before overlapping detections are culled. Defaults to 0, meaning no overlap is allowed
                Setting to e.g. 0.3 allows detections to overlap by 30%, while setting to 1 disables all culling. This will generally
                lead to many detections of the same particle at different radii in the radius scan, and is inadvicable
                "#),
            sorter: 0,
        },
    ];

    log_args.extend(common_args.into_iter());
    let log_init = quote!(
        let style = ParamStyle::Log{
            min_radius,
            max_radius,
            n_radii,
            log_spacing,
            overlap_threshold,
        };
        let include_r_in_output = true;
        #common_init
    );
    // log_init.extend(common_init);
    (log_args, log_init)
    // quote!(
    //     min_radius: my_dtype,
    //     max_radius: my_dtype,
    //     n_radii: Option<usize>,
    //     log_spacing: Option<bool>,
    //     overlap_threshold: Option<my_dtype>,
    //     #common_args
    // )
}

fn char_args() -> (Vec<Argument>, TokenStream2) {
    let (tp_args, tp_init) = tp_args();
    let mut char_args = vec![
        Argument{
            func: quote!(points_to_characterize: PyReadonlyArray2<my_dtype>,),
            init: quote!(),
            docs: fmt(r#"
                The points to do characterization at. Should be a pandas dataframe with atleast the columns "y" and "x".
                A characterization exactly like that which would be done if the points had been found as particles by gpu_tracking
                will be done at the points. If "frame" is a column in the dataframe, the characterization will be done just at the specified
                frames. If a "frame" column is absent, it is assumed that the points should be characterized at all frames in the supplied
                video. If an "r" column is supplied (like the one returned by LoG), this is taken to be the size of the supplied particles.
                If an "r" column is not supplied, a diameter should be.
                "#),
            sorter: 0,
        },
        Argument{
            func: quote!(points_has_frames: bool,),
            init: quote!(),
            docs: fmt(""),
            sorter: 0,
        },
        Argument{
            func: quote!(points_has_r: bool,),
            init: quote!(),
            docs: fmt(""),
            sorter: 0,
        },
    ];
    char_args.extend(tp_args.into_iter());
    let char_init = quote!(
        #tp_init
        params.characterize = true;
        let characterize_points = Some((points_to_characterize.as_array(), points_has_frames, points_has_r));
        if points_has_r{
            params.include_r_in_output = true;
        }
        if let ParamStyle::Trackpy{ ref mut filter_close, .. } = params.style{
            *filter_close = false;
        }
    );
    (char_args, char_init)
}



fn parse_tp() -> TokenStream2 {
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

fn parse_log() -> TokenStream2 {
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

fn parse_common() -> TokenStream2 {
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
            keys: frames,
            noise_size,
            smoothing_size,
            illumination_correction_per_frame,
        };
    )
}

// fn file_args() -> TokenStream2 {
//     quote!(py: Python<'py>, filename: String,)
// }

// fn file_post_args() -> TokenStream2 {
//     quote!(channel: Option<usize>,)
// }

// fn array_args() -> TokenStream2 {
//     quote!(py: Python<'py>, pyarr: PyReadonlyArray3<my_dtype>,)
// }

// fn array_post_args() -> TokenStream2 {
//     TokenStream2::new()
// }

fn return_type() -> TokenStream2 {
    // quote!(PyResult<(&'py PyArray2<my_dtype>, Py<PyAny>)>)
    quote!(PyResult<&'py PyAny>)
}

#[proc_macro]
pub fn gen_python_functions(_item: TokenStream) -> TokenStream {
    use rust_format::{Formatter, RustFmt};
    let mut out = TokenStream2::new();
    let mut wrappers = std::fs::read_to_string("python/gpu_tracking/wrappers.py").expect("Couldn't read python/gpu_tracking/wrappers.py");
    let mut file = std::fs::File::create("python/gpu_tracking/generated.py").expect("Couldn't open python/gpu_tracking/generated.py");
    
    for s in Style::iter() {
        let mut this_doc = "\n".to_string();
        let (this_func, func_name, all_args) = make_func(s);
        let func_name = func_name.to_string();
        for arg in all_args.into_iter(){
            let Some(proc_macro2::TokenTree::Ident(arg_name)) = arg.func.into_iter().next() else { panic!( "First token isn't an ident" )};
            let docs = arg.docs;
            if docs.len() == 0{
                continue
            }
            this_doc.push_str(&format!("\t\t{arg_name}: {docs}\n"));
        }
        wrappers = wrappers.replace(&format!("__{func_name}_arg_docs__"), &this_doc);
        out.extend(this_func);
    }
    
    let expansion = RustFmt::default().format_str(out.to_string()).unwrap_or(out.to_string());
    // eprintln!("{}", &expansion);

    let source = std::fs::read_to_string("gpu_tracking_python/src/python_bindings.rs").expect("cant read source");
    let mut expansion_file = std::fs::File::create("gpu_tracking_python/src/expansion.rs").expect("Couldn't open python/gpu_tracking/generated.py");
    write!(expansion_file, "{}", source.replace("gen_python_functions!();", &expansion)).expect("can't write to expansion");
    
    write!(file, "{}", wrappers).expect("Couldn't write to python/gpu_tracking/generated.py");
    
    out.into()
}

