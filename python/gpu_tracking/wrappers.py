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
    
    Args:__batch_rust_arg_docs__
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

    
    Args:__characterize_points_rust_arg_docs__
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
    
    Args:__LoG_rust_arg_docs__
    """
    return LoG_rust(
        file_or_array,
        min_r,
        max_r,
        **kwargs
    )
