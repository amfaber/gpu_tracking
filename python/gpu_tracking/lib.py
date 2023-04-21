from gpu_tracking import *
import pandas as pd
import numpy as np
import uuid
from .generated import *

# __all__ = ["batch", "characterize_points", "link", "connect", "LoG"]
# print("getting imported")
# print(__all__)

# def batch(
#     video_or_path,
#     diameter,
#     **kwargs
#     ):
#     """Test: {}"""
#     return batch_rust(
#         video_or_path,
#         diameter,
#         **kwargs
#     )

# def characterize_points(
#     video_or_path,
#     points_to_characterize,
#     diameter = None,
#     has_frames = None,
#     has_r = None,
#     **kwargs,
# ):
#     if isinstance(points_to_characterize, pd.DataFrame):
#         cols = ["y", "x"]
#         if "frame" in points_to_characterize.columns:
#             cols = ["frame"] + cols
#             if has_frames is None:
#                 has_frames = True
#         if "r" in points_to_characterize.columns:
#             cols = cols + ["r"]
#             if has_r is None:
#                 has_r = True

#         if has_frames is None:
#             has_frames = False
#         if has_r is None:
#             has_r = False
#         points_arr = points_to_characterize[cols].to_numpy().astype("float32")
#     else:
#         points_arr = points_to_characterize

#     if diameter is None:
#         if has_r:
#             diameter = 2*int(points_arr[:, -1].max() + 0.5) + 1
#         else:
#             raise ValueError("Diameter needs to be specified if the supplied points don't have associated radiuses")
            
#     return characterize_rust(
#         video_or_path,
#         points_arr,
#         has_frames,
#         has_r,
#         diameter,
#         **kwargs
#     )

    
# def LoG(video_or_path, min_r, max_r, **kwargs):
#     return LoG_rust(
#         video_or_path,
#         min_r,
#         max_r,
#         **kwargs
#     )

def link(to_link, search_range, memory):
    """
    Performs optimal (as opposed to greedy) nearest neighbor linking on the provided dataframe.
    
    Returns:
        pandas.DataFrame: The input dataframe with an additional column "particle", which denotes which
        particles are identical throughout frames.

    Args:
        to_link: A pandas dataframe with atleast the columns "frame", "y" and "x".
        
        search_range: The upper cutoff distance for nearest neighbor linking.
        
        memory: The number of frames that a particle is allowed to disappear before reappearing, for the purposes of linking.\
        memory = 5 means that a particle can be gone for 5 frames, reappearing in the 6th and still be linked to the\
        track it had built previously. If it had reappeared in the 7th, it would instead have been considered a new\
        particle. Defaults to 0, and has no effect if search_range is not set.
    
    """
    if isinstance(to_link, pd.DataFrame):
        to_link_np = to_link[["frame", "y", "x"]].to_numpy()
    else:
        to_link_np = to_link
    to_link_np = to_link_np.as_type("float32")
    result = link_rust(to_link_np, search_range, memory)

    if isinstance(to_link, pd.DataFrame):
        output = to_link.copy()
        output["particle"] = result
    else:
        output = result

    return output

def connect(to_link1, to_link2, search_range, merge = True):
    """
    Does linking on a frame-by-frame basis between two dataframes from different detection runs.
    This has multiple uses, such colocalizing associating detections in two separate channels,
    or evaluating a tracking algorithm by "connect"ing the predicted positions to ground truth
    detections or detections from another algorithm.

    Returns:
        pandas.DataFrame: A merged dataframe containing all the detections from the two\
        input dataframes with an additional column "connect particle", that associates detections
        from the two input dataframes. Detections that are present in, for example, dataframe 1
        but not dataframe 2 will show as having NaN for all the "_y" suffixed columns, whereas the
        inverse shows as NaNs in the "_x" suffixed columns

    Args:
        to_link1: First input dataframe
        to_link2: Second input dataframe
        search_range: The search range for linking. See :ref:`link <link>`
        merge: Whether to return a merged dataframe. Defaults to True. If False, instead returns\
        the input dataframes with "connect particle" as an additional column in both.
    
    """
    if isinstance(to_link1, pd.DataFrame):
        to_link_np1 = to_link1[["frame", "y", "x"]].to_numpy()
    else:
        to_link_np1 = to_link1
    
    if isinstance(to_link2, pd.DataFrame):
        to_link_np2 = to_link2[["frame", "y", "x"]].to_numpy()
    else:
        to_link_np2 = to_link2
    
    to_link_np1 = to_link_np1.astype("float32")
    to_link_np2 = to_link_np2.astype("float32")
    result = connect_rust(to_link_np1, to_link_np2, search_range)

    if isinstance(to_link1, pd.DataFrame):
        output1 = to_link1.copy()
        output1["connect particle"] = result[0]
    else:
        output1 = result[0]
        
    if isinstance(to_link2, pd.DataFrame):
        output2 = to_link2.copy()
        output2["connect particle"] = result[1]
    else:
        output2 = result[1]
    if not merge:
        return output1, output2
    else:
        return output1.merge(output2, how = "outer", on = "connect particle")


def mean_from_file(path, channel = None):
    """
    Takes the average across frames of the provided video returns a numpy array with the result.
    
    Args:
        path: The file path to the video to mean
        channel: In the case of .vsi / .ets files, the channel to mean.
    
    Returns:
        video: The numpy array containing the average of all frames in the video.
    """
    return mean_from_file_rust(path, channel)

def load(path, frames = None, channel = None):
    """
    Loads a tiff or .vsi / .ets a the provided path and returns it as a 3-dimensional numpy array.
    
    Args:
        path: The file path to the video to mean
        frames: A sequence that specifies what frames from the video to load. For example, to only load the first 50 frames of a video, frames = range(50) can be supplied.
        channel: In the case of .vsi / .ets files, the channel to mean.
    
    Returns:
        video: The numpy array containing the average of all frames in the video.
    """
    return load_rust(path, frames, channel) 
    

def annotate_image(image, tracked_df, figax = None, r = None, frame = None, imshow_kw = {}, circle_kw = {}, subplot_kw = {}):
    import matplotlib.pyplot as plt

    circle_kw = {"fill": False, **circle_kw}
    if frame is not None:
        subset_df = tracked_df[tracked_df["frame"] == frame]
    else:
        subset_df = tracked_df
    
    if r is None and not "r" in subset_df:
        r = 5
        print(f"Using default r of {r}")
    if figax is None:
        fig, ax = plt.subplots(**subplot_kw)
    else:
        fig, ax = figax
    ax.imshow(image, **imshow_kw)

    for _idx, row in subset_df.iterrows():
        if r is None:
            inner_r = row["r"]
        else:
            inner_r = r
        x, y = row["x"], row["y"]
        ax.add_patch(plt.Circle((x, y), inner_r, **circle_kw))
    return (fig, ax)

def annotate_image_plotly(image, tracked_df, r = None, frame = None, imshow_kw = {}, circle_color = "white", color_scale = "viridis", circle_kw = {}):
    from plotly import express as px
    
    if frame is not None:
        subset_df = tracked_df[tracked_df["frame"] == frame]
    else:
        subset_df = tracked_df
    
    if r is None and "r" not in subset_df:
        r = 5
        print(f"Using default r of {r}")
    fig = px.imshow(image, color_continuous_scale = color_scale, **imshow_kw)

    for _idx, row in subset_df.iterrows():
        if r is None:
            inner_r = row["r"]
        else:
            inner_r = r
        x, y = row["x"], row["y"]
        fig.add_shape(
            type = "circle", xref = "x", yref = "y",
            x0 = x - inner_r, y0 = y - inner_r,
            x1 = x + inner_r, y1 = y + inner_r,
            line_color = circle_color, line_width = 1, **circle_kw
        )
    return fig

def annotate_video(video, tracked_df, frame = 0, **kwargs):
    image = video[frame]
    return annotate_image(image, tracked_df, frame = frame, **kwargs)

def annotate_file(path, tracked_df, ets_channel = 0, frame = 0, **kwargs):
    image = load(path, ets_channel = ets_channel, keys = [frame])[0]
    return annotate_image(image, tracked_df, frame = frame, **kwargs)

def unique_particle_ids(df, column = "particle"):
    id = uuid.uuid4()
    df[column] = df[column].as_type("str") + id
    
