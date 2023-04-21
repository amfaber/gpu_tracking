from gpu_tracking import *
def batch(
    file_or_array,
    diameter,
    **kwargs
    ):
    """
    Trackpy-like tracking, executed on a GPU. Intensity integration, linking and other properties
    can be calculated while the GPU is running for further efficiency. It is recommended to visually inspect tracking with both trackpy-like
    tracking and LoG tracking to choose the best approach and parameters for the experiment.

    The most important parameters are "diameter", "snr" and "search_range". These are discribed below.

    Returns:
        pandas.DataFrame: A pandas dataframe containing the results of tracking. Which columns are present depend on the parameters used for tracking, but the "frame", "y" and "x" always specify where in the video the detection is found.

    Examples:
        The following example tracks "path/to/video.vsi", looking for particles of diameter 9 pixels and runs linking with a cutoff of 10 pixels.
    
        .. code-block:: python
    
            import gpu_tracking as gt
            gt.batch("path/to/video.vsi", 9, search_range = 10)
    
    Args:
		file_or_array: The input video. Either a file-path to where the video can be found (tiff, vsi or ets format), or a numpy array with shape                (T, Y, X), T corresponding to the time dimension, and Y and X corresponding to y and x in the output dataframe
		diameter:                 The diameter of the particles that are searched for. This is the only required parameter (except for in                characterize_points), and so many other parameters default to values based on the diameter. On its own, it is the diameter                of the circle within which all intensity integration calculations are done.                
		separation: The minimum separation between particles in pixels. Defaults to diameter + 1. This is used for the maximum filter            in the Crocker-Grier algorithm, and for subsequently filtering detections that are too close to eachother.
		filter_close: Whether to filter detections that are too close to eachother with "separation"
		noise_size: The sigma of a gaussian smoothing that is applied to the video during preprocessing. Defaults to 1.0.                Was introduced by Crocker-Grier to counteract some digitalization noise of CCD cameras
		smoothing_size: The side-length of the uniform convolution that subtracts local background during preprocessing. Defaults to diameter.                Setting this too low and close to the actual size of the signal runs the risk of supressing the signal, as most of the "local background"                will be the signal itself. As such, it can be beneficial to set this higher than the diameter in some cases.
		minmass:                 The minimum integrated intensity a particle has to have to be considered a true particle, and not a spurious                detection. Defaults to 0. This setting provides the same functionality as "minmass_snr", only the threshold                is set as an absolute number rather than being relative to the video's noise level. "minmass_snr" should be                preferred in most cases.
		max_iterations:                 The maximum number of steps that the localization refinement algorithm is allowed to take. Defaults to 10.                Increasing this number is unlikely to increase localization accuracy, as it just allows each detection to                move further away from the local maximum that seeded the algorithm.                
		characterize:                 Whether to include the columns "Rg", "raw", "signal" and "ecc" in the output dataframe. Defaults to True.                "Rg" is the radius of gyration of the detection. "raw" is the integrated intensity of the particle in the                unprocessed (and not background subtracted) image. "signal" is the peak (i.e. maximum pixel value) of the                signal in the preprocessed image. "ecc" is the particle's eccentricity. These can be helpful measures for                further processing beyond gpu_tracking.                
		search_range:                 The search range in pixel space used for linking the particles through time. Defaults to None, meaning                that linking won't be performed. The linking algorithm is identical to Trackpy's linking, and does                optimal (as opposed to greedy) nearest neighbor linking.                
		memory:                 The number of frames that a particle is allowed to disappear before reappearing, for the purposes of linking.                memory = 5 means that a particle can be gone for 5 frames, reappearing in the 6th and still be linked to the                track it had built previously. If it had reappeared in the 7th, it would instead have been considered a new                particle. Defaults to 0, and has no effect if search_range is not set.                
		doughnut_correction:                 Whether to include the columns "raw_mass", "raw_bg_median" and "raw_mass_corrected" in the output dataframe. Like "raw"                from characterize, "raw_mass" is the integrated intensity of the particle in the raw input image. "raw_bg_median" is the                median of the background around the particle in a hollow "doughnut" of outer radius "bg_radius", inner radius "diameter / 2                + gap_radius". "raw_mass_corrected" is "raw_mass" - "raw_bg_median" * #pixels_in_particle. "raw_mass_corrected" is generally                the most accurate measure we have of particle intensities.                
		bg_radius:                 The radius to use for "doughnut_correction". Defaults to "diameter", i.e. twice the particles radius.                
		gap_radius:                 An optional amount of gap between the particle and background measurement for "doughnut_correction". Defaults to 0.0.                
		snr:                 Primary filter for what to consider a particle and what is simply suprious detections. Defaults to 0. "snr" measures the                noise level of the video by taking the standard deviation of each frame individually and taking this to be the global                noise level. The peak of the particle in the proprocessed image must then be above [noise_level] * [snr] to be considered                a particle. Videos with a very non-uniform background cause trouble for this setting, as the global noise level will be                artificially inflated, necessitating setting a lower "snr". This setting is a useful heuristic for setting comparable                thresholds across quite different videos, but should not be interpreted as a strict filter for only getting particles above                the set snr level.                
		minmass_snr:                 Serves the same role as "snr", except where "snr" filters on the particle's peak signal, "snr_minmass" filters on the particle's                integrated intensity, potentially squashing random high "lone peaks". Defaults to 0.                
		correct_illumination:                 Whether to correct the illumination profile of the video before tracking. Defaults to False. This is done by smoothing the video with sigma=30 pixels                and then dividing the raw video by the very smoothed video, leveling out any local differences in background. This can be helpful                in the case of uneven illumination profiles, but also in other cases of uneven backgrounds.                
		illumination_sigma:                 Same as "correct_illumination", except a sigma can be provided. Defaults to 30 if correction_illumination is True, and None otherwise, meaning that                correction will not be done. If both are provided, "illumination_sigma" takes precedence.                
		illumination_correction_per_frame:                 When doing illumination correction, this setting controls whether to do the correction on a per-frame basis, or if the                entire video should be loaded, averaged across frames, and then the resulting average frame is the only one that is smoothed                and used to do the correction. Defaults to False, meaning that the video is loaded, averaged and smoothed before starting the actual                detection algorithm.                
		adaptive_background:                 If "snr" or "minmass_snr" are provided, this setting allows the measurement of the global background noise level to be updated adaptively.                Once a number of particles have been detected as being above the set "snr" and "minmass_snr", the pixels of these particles are removed                from the raw image, and the global noise level is recalculated. Particles are then tested again if they are now below the thresholds of                "snr" and "minmass_snr" with the updated noise level, and included if they now pass the check. This process can be repeated iteratively,                and "adaptive_background" sets the number of times it is repeated. Defaults to None, meaning that the process is not run at all.                
		shift_threshold:                 The threshold for stopping the localization refinement algorithm. Defaults to 0.6, and should never be below 0.5. Generally                not recommended to change.                
		linker_reset_points:                 A list of points at which the linking should be reset. Defaults to no reset points. Useful if a single video has points at                which the recording was paused and then later resumed. Supplying these points to this option ensures that particles that are                actually temporally very far from eachother aren't linked together.                
		frames:                 A sequence that specifies what frames from the video to track. For example, to only load and track the first 50 frames of a video,                frames = range(50) can be supplied. Be aware that all the frames in the sequence are assumed to be next to eachother in time, i.e.                specifying frames = [0, 1, 2, 50, 100] will attempt to link the detections in frame 2 to those in frame 50, and those in frame 50 to                those in frame 100. This can be further customized with "linker_reset_points". Defaults to tracking all supplied frames.                
		tqdm: Whether to use tqdm to report progress. Defaults to True
		channel: In case a .vsi / .ets video is supplied, this channel will be used from the video. Defaults to 0

    """
    return batch_rust(
        file_or_array,
        diameter,
        **kwargs
    )

def characterize_points(
    file_or_array,
    points_to_characterize,
    diameter = None,
    # has_frames = None,
    # has_r = None,
    **kwargs,
):
    """
    Runs gpu_tracking particle characterization of the provided points, as if gpu_tracking had itself found these points as particle locations.
    

    Returns:
        pandas.DataFrame: A pandas dataframe containing the results of characterization.
    
    Examples:
        Characterize the points [340, 200] and [100, 50] in frames 0 and 40 of the video located at "path/to/video.vsi"
    
        .. code-block:: python
    
            import gpu_tracking as gt
            import pandas as pd
            points = pd.DataFrame()
            points.loc[0, [["y", "x", "frame"]]] = [340, 200, 0]
            points.loc[1, [["y", "x", "frame"]]] = [100, 50, 40]
            gt.characterize_points("path/to/video.vsi", points)
    

        Use the particles found by tracking the average of one channel to integrate intensity at those points throughout all frames in another channel.
    
        .. code-block:: python
    
            import gpu_tracking as gt
            mean_frame = gt.mean_from_file("path/to/channel1.vsi")
            channel1_detections = gt.batch(mean_frame, 9)
            channel1_detections.drop(columns = "frame", inplace = True)
            channel2_characterizations = gt.characterize_points("path/to/channel2.vsi", channel1_detections)

    
    Args:
		file_or_array: The input video. Either a file-path to where the video can be found (tiff, vsi or ets format), or a numpy array with shape                (T, Y, X), T corresponding to the time dimension, and Y and X corresponding to y and x in the output dataframe
		points_to_characterize:                 The points to do characterization at. Should be a pandas dataframe with atleast the columns "y" and "x".                A characterization exactly like that which would be done if the points had been found as particles by gpu_tracking                will be done at the points. If "frame" is a column in the dataframe, the characterization will be done just at the specified                frames. If a "frame" column is absent, it is assumed that the points should be characterized at all frames in the supplied                video. If an "r" column is supplied (like the one returned by LoG), this is taken to be the size of the supplied particles.                If an "r" column is not supplied, a diameter should be.                
		diameter:                 The diameter of the particles that are searched for. This is the only required parameter (except for in                characterize_points), and so many other parameters default to values based on the diameter. On its own, it is the diameter                of the circle within which all intensity integration calculations are done.                
		separation: The minimum separation between particles in pixels. Defaults to diameter + 1. This is used for the maximum filter            in the Crocker-Grier algorithm, and for subsequently filtering detections that are too close to eachother.
		filter_close: Whether to filter detections that are too close to eachother with "separation"
		noise_size: The sigma of a gaussian smoothing that is applied to the video during preprocessing. Defaults to 1.0.                Was introduced by Crocker-Grier to counteract some digitalization noise of CCD cameras
		smoothing_size: The side-length of the uniform convolution that subtracts local background during preprocessing. Defaults to diameter.                Setting this too low and close to the actual size of the signal runs the risk of supressing the signal, as most of the "local background"                will be the signal itself. As such, it can be beneficial to set this higher than the diameter in some cases.
		minmass:                 The minimum integrated intensity a particle has to have to be considered a true particle, and not a spurious                detection. Defaults to 0. This setting provides the same functionality as "minmass_snr", only the threshold                is set as an absolute number rather than being relative to the video's noise level. "minmass_snr" should be                preferred in most cases.
		max_iterations:                 The maximum number of steps that the localization refinement algorithm is allowed to take. Defaults to 10.                Increasing this number is unlikely to increase localization accuracy, as it just allows each detection to                move further away from the local maximum that seeded the algorithm.                
		characterize:                 Whether to include the columns "Rg", "raw", "signal" and "ecc" in the output dataframe. Defaults to True.                "Rg" is the radius of gyration of the detection. "raw" is the integrated intensity of the particle in the                unprocessed (and not background subtracted) image. "signal" is the peak (i.e. maximum pixel value) of the                signal in the preprocessed image. "ecc" is the particle's eccentricity. These can be helpful measures for                further processing beyond gpu_tracking.                
		search_range:                 The search range in pixel space used for linking the particles through time. Defaults to None, meaning                that linking won't be performed. The linking algorithm is identical to Trackpy's linking, and does                optimal (as opposed to greedy) nearest neighbor linking.                
		memory:                 The number of frames that a particle is allowed to disappear before reappearing, for the purposes of linking.                memory = 5 means that a particle can be gone for 5 frames, reappearing in the 6th and still be linked to the                track it had built previously. If it had reappeared in the 7th, it would instead have been considered a new                particle. Defaults to 0, and has no effect if search_range is not set.                
		doughnut_correction:                 Whether to include the columns "raw_mass", "raw_bg_median" and "raw_mass_corrected" in the output dataframe. Like "raw"                from characterize, "raw_mass" is the integrated intensity of the particle in the raw input image. "raw_bg_median" is the                median of the background around the particle in a hollow "doughnut" of outer radius "bg_radius", inner radius "diameter / 2                + gap_radius". "raw_mass_corrected" is "raw_mass" - "raw_bg_median" * #pixels_in_particle. "raw_mass_corrected" is generally                the most accurate measure we have of particle intensities.                
		bg_radius:                 The radius to use for "doughnut_correction". Defaults to "diameter", i.e. twice the particles radius.                
		gap_radius:                 An optional amount of gap between the particle and background measurement for "doughnut_correction". Defaults to 0.0.                
		snr:                 Primary filter for what to consider a particle and what is simply suprious detections. Defaults to 0. "snr" measures the                noise level of the video by taking the standard deviation of each frame individually and taking this to be the global                noise level. The peak of the particle in the proprocessed image must then be above [noise_level] * [snr] to be considered                a particle. Videos with a very non-uniform background cause trouble for this setting, as the global noise level will be                artificially inflated, necessitating setting a lower "snr". This setting is a useful heuristic for setting comparable                thresholds across quite different videos, but should not be interpreted as a strict filter for only getting particles above                the set snr level.                
		minmass_snr:                 Serves the same role as "snr", except where "snr" filters on the particle's peak signal, "snr_minmass" filters on the particle's                integrated intensity, potentially squashing random high "lone peaks". Defaults to 0.                
		correct_illumination:                 Whether to correct the illumination profile of the video before tracking. Defaults to False. This is done by smoothing the video with sigma=30 pixels                and then dividing the raw video by the very smoothed video, leveling out any local differences in background. This can be helpful                in the case of uneven illumination profiles, but also in other cases of uneven backgrounds.                
		illumination_sigma:                 Same as "correct_illumination", except a sigma can be provided. Defaults to 30 if correction_illumination is True, and None otherwise, meaning that                correction will not be done. If both are provided, "illumination_sigma" takes precedence.                
		illumination_correction_per_frame:                 When doing illumination correction, this setting controls whether to do the correction on a per-frame basis, or if the                entire video should be loaded, averaged across frames, and then the resulting average frame is the only one that is smoothed                and used to do the correction. Defaults to False, meaning that the video is loaded, averaged and smoothed before starting the actual                detection algorithm.                
		adaptive_background:                 If "snr" or "minmass_snr" are provided, this setting allows the measurement of the global background noise level to be updated adaptively.                Once a number of particles have been detected as being above the set "snr" and "minmass_snr", the pixels of these particles are removed                from the raw image, and the global noise level is recalculated. Particles are then tested again if they are now below the thresholds of                "snr" and "minmass_snr" with the updated noise level, and included if they now pass the check. This process can be repeated iteratively,                and "adaptive_background" sets the number of times it is repeated. Defaults to None, meaning that the process is not run at all.                
		shift_threshold:                 The threshold for stopping the localization refinement algorithm. Defaults to 0.6, and should never be below 0.5. Generally                not recommended to change.                
		linker_reset_points:                 A list of points at which the linking should be reset. Defaults to no reset points. Useful if a single video has points at                which the recording was paused and then later resumed. Supplying these points to this option ensures that particles that are                actually temporally very far from eachother aren't linked together.                
		frames:                 A sequence that specifies what frames from the video to track. For example, to only load and track the first 50 frames of a video,                frames = range(50) can be supplied. Be aware that all the frames in the sequence are assumed to be next to eachother in time, i.e.                specifying frames = [0, 1, 2, 50, 100] will attempt to link the detections in frame 2 to those in frame 50, and those in frame 50 to                those in frame 100. This can be further customized with "linker_reset_points". Defaults to tracking all supplied frames.                
		tqdm: Whether to use tqdm to report progress. Defaults to True
		channel: In case a .vsi / .ets video is supplied, this channel will be used from the video. Defaults to 0

    """
    has_frames = None,
    has_r = None,
    if isinstance(points_to_characterize, pd.DataFrame):
        cols = ["y", "x"]
        if "frame" in points_to_characterize.columns:
            cols = ["frame"] + cols
            if has_frames is None:
                has_frames = True
        if "r" in points_to_characterize.columns:
            cols = cols + ["r"]
            if has_r is None:
                has_r = True

        if has_frames is None:
            has_frames = False
        if has_r is None:
            has_r = False
        points_arr = points_to_characterize[cols].to_numpy().astype("float32")
    else:
        points_arr = points_to_characterize

    if diameter is None:
        if has_r:
            diameter = 2*int(points_arr[:, -1].max() + 0.5) + 1
        else:
            raise ValueError("Diameter needs to be specified if the supplied points don't have associated radiuses")
            
    return characterize_rust(
        file_or_array,
        points_arr,
        has_frames,
        has_r,
        diameter,
        **kwargs
    )

    
def LoG(file_or_array, min_radius, max_radius, **kwargs):
    """
    Performs tracking with Laplacian of the Gaussian blob detection rather than local maximum,
    as Trackpy does it. Laplacian of the Gaussian is a more expensive algorithm, taking longer
    to run (though this can be reduced by reducing n_radii), but offers automatic detection of
    the size of the found particles, allowing tracking of particles of heterogenous sizes.
    It is also better able to pick up particles that are very close to eachother, though it is
    still a difficult task. It is recommended to visually inspect tracking with both trackpy-like
    tracking and LoG tracking to choose the best approach and parameters for the experiment.

    The most important parameters are "min_radius", "max_radius", "snr" and "search_range".
    These are discribed below.

    Returns:
        pandas.DataFrame: A pandas dataframe containing the results of tracking. Which columns are present
        depend on the parameters used for tracking, but the "frame", "y", "x" and "r" always
        specify where in the video the detection is found and the size of the particle
        according to LoG.

    Examples:
        The following example tracks "path/to/video.vsi", looking for particles with radii
        between 2.2 pixels and 3.5 pixels and runs linking with a cutoff of 10 pixels.

        .. code-block:: python
    
            import gpu_tracking as gt
            gt.LoG("path/to/video.vsi", 2.2, 3.5, search_range = 10)
    
    Args:
		file_or_array: The input video. Either a file-path to where the video can be found (tiff, vsi or ets format), or a numpy array with shape                (T, Y, X), T corresponding to the time dimension, and Y and X corresponding to y and x in the output dataframe
		min_radius: The minimum radius of the radius scan of laplacian of the gaussian
		max_radius:                 The maximum radius of the radius scan of laplacian of the gaussian. All the parameters that are set based on defaults of                the "diameter" parameter in Trackpy style tracking instead use 2x max_radius when tracking with LoG. This can have severe                consequences if the maximum radius is set very large without also adjusting parameters that are set based on this.                
		n_radii:                 The number of radii in the radius scan. Defaults to 10. Execution time scales linearly with this setting,                as expensive convolutions need to be calculated for each radius in the radius scan.                
		log_spacing: Whether to use logarithmic spacing in the radius scan. Defaults to false, using linear spacing.
		overlap_threshold:                 The maximum allowed overlap before overlapping detections are culled. Defaults to 0, meaning no overlap is allowed                Setting to e.g. 0.3 allows detections to overlap by 30%, while setting to 1 disables all culling. This will generally                lead to many detections of the same particle at different radii in the radius scan, and is inadvicable                
		noise_size: The sigma of a gaussian smoothing that is applied to the video during preprocessing. Defaults to 1.0.                Was introduced by Crocker-Grier to counteract some digitalization noise of CCD cameras
		smoothing_size: The side-length of the uniform convolution that subtracts local background during preprocessing. Defaults to diameter.                Setting this too low and close to the actual size of the signal runs the risk of supressing the signal, as most of the "local background"                will be the signal itself. As such, it can be beneficial to set this higher than the diameter in some cases.
		minmass:                 The minimum integrated intensity a particle has to have to be considered a true particle, and not a spurious                detection. Defaults to 0. This setting provides the same functionality as "minmass_snr", only the threshold                is set as an absolute number rather than being relative to the video's noise level. "minmass_snr" should be                preferred in most cases.
		max_iterations:                 The maximum number of steps that the localization refinement algorithm is allowed to take. Defaults to 10.                Increasing this number is unlikely to increase localization accuracy, as it just allows each detection to                move further away from the local maximum that seeded the algorithm.                
		characterize:                 Whether to include the columns "Rg", "raw", "signal" and "ecc" in the output dataframe. Defaults to True.                "Rg" is the radius of gyration of the detection. "raw" is the integrated intensity of the particle in the                unprocessed (and not background subtracted) image. "signal" is the peak (i.e. maximum pixel value) of the                signal in the preprocessed image. "ecc" is the particle's eccentricity. These can be helpful measures for                further processing beyond gpu_tracking.                
		search_range:                 The search range in pixel space used for linking the particles through time. Defaults to None, meaning                that linking won't be performed. The linking algorithm is identical to Trackpy's linking, and does                optimal (as opposed to greedy) nearest neighbor linking.                
		memory:                 The number of frames that a particle is allowed to disappear before reappearing, for the purposes of linking.                memory = 5 means that a particle can be gone for 5 frames, reappearing in the 6th and still be linked to the                track it had built previously. If it had reappeared in the 7th, it would instead have been considered a new                particle. Defaults to 0, and has no effect if search_range is not set.                
		doughnut_correction:                 Whether to include the columns "raw_mass", "raw_bg_median" and "raw_mass_corrected" in the output dataframe. Like "raw"                from characterize, "raw_mass" is the integrated intensity of the particle in the raw input image. "raw_bg_median" is the                median of the background around the particle in a hollow "doughnut" of outer radius "bg_radius", inner radius "diameter / 2                + gap_radius". "raw_mass_corrected" is "raw_mass" - "raw_bg_median" * #pixels_in_particle. "raw_mass_corrected" is generally                the most accurate measure we have of particle intensities.                
		bg_radius:                 The radius to use for "doughnut_correction". Defaults to "diameter", i.e. twice the particles radius.                
		gap_radius:                 An optional amount of gap between the particle and background measurement for "doughnut_correction". Defaults to 0.0.                
		snr:                 Primary filter for what to consider a particle and what is simply suprious detections. Defaults to 0. "snr" measures the                noise level of the video by taking the standard deviation of each frame individually and taking this to be the global                noise level. The peak of the particle in the proprocessed image must then be above [noise_level] * [snr] to be considered                a particle. Videos with a very non-uniform background cause trouble for this setting, as the global noise level will be                artificially inflated, necessitating setting a lower "snr". This setting is a useful heuristic for setting comparable                thresholds across quite different videos, but should not be interpreted as a strict filter for only getting particles above                the set snr level.                
		minmass_snr:                 Serves the same role as "snr", except where "snr" filters on the particle's peak signal, "snr_minmass" filters on the particle's                integrated intensity, potentially squashing random high "lone peaks". Defaults to 0.                
		correct_illumination:                 Whether to correct the illumination profile of the video before tracking. Defaults to False. This is done by smoothing the video with sigma=30 pixels                and then dividing the raw video by the very smoothed video, leveling out any local differences in background. This can be helpful                in the case of uneven illumination profiles, but also in other cases of uneven backgrounds.                
		illumination_sigma:                 Same as "correct_illumination", except a sigma can be provided. Defaults to 30 if correction_illumination is True, and None otherwise, meaning that                correction will not be done. If both are provided, "illumination_sigma" takes precedence.                
		illumination_correction_per_frame:                 When doing illumination correction, this setting controls whether to do the correction on a per-frame basis, or if the                entire video should be loaded, averaged across frames, and then the resulting average frame is the only one that is smoothed                and used to do the correction. Defaults to False, meaning that the video is loaded, averaged and smoothed before starting the actual                detection algorithm.                
		adaptive_background:                 If "snr" or "minmass_snr" are provided, this setting allows the measurement of the global background noise level to be updated adaptively.                Once a number of particles have been detected as being above the set "snr" and "minmass_snr", the pixels of these particles are removed                from the raw image, and the global noise level is recalculated. Particles are then tested again if they are now below the thresholds of                "snr" and "minmass_snr" with the updated noise level, and included if they now pass the check. This process can be repeated iteratively,                and "adaptive_background" sets the number of times it is repeated. Defaults to None, meaning that the process is not run at all.                
		shift_threshold:                 The threshold for stopping the localization refinement algorithm. Defaults to 0.6, and should never be below 0.5. Generally                not recommended to change.                
		linker_reset_points:                 A list of points at which the linking should be reset. Defaults to no reset points. Useful if a single video has points at                which the recording was paused and then later resumed. Supplying these points to this option ensures that particles that are                actually temporally very far from eachother aren't linked together.                
		frames:                 A sequence that specifies what frames from the video to track. For example, to only load and track the first 50 frames of a video,                frames = range(50) can be supplied. Be aware that all the frames in the sequence are assumed to be next to eachother in time, i.e.                specifying frames = [0, 1, 2, 50, 100] will attempt to link the detections in frame 2 to those in frame 50, and those in frame 50 to                those in frame 100. This can be further customized with "linker_reset_points". Defaults to tracking all supplied frames.                
		tqdm: Whether to use tqdm to report progress. Defaults to True
		channel: In case a .vsi / .ets video is supplied, this channel will be used from the video. Defaults to 0

    """
    return LoG_rust(
        file_or_array,
        min_r,
        max_r,
        **kwargs
    )
