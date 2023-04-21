Tour
====


This is a small tour that will briefly cover the layout of options available in gpu_tracking

.. image:: Overview.png
	:align: center

The app is window based, with each window able to display a single video at a time. An arbitrary number of windows can be opened,
but performance might start to suffer if very many are opened. Windows can be coupled to eachother, allowing a window to show not
just the data from tracking its own video, but also the data from another window. The other window might be a different tracking,
or it could be a static dataset loaded as a .csv file. This is a very flexible approach, that allows for easily comparing a lot of
data in the same window on the same video.

Below is a short description of each of the areas enumerated in the above screenshot

1 - New button
   Creates a new window without any couplings

2 - Help button
   Opens the documentation you are currently reading in a native browser window.

3 - Input
   Here, the file path to a video can be provided in the writable field, or the video can be selected from the browse button, which
   will open a file-picker. vsi / ets channel chooses which channel to read for .vsi / .ets files and defaults to 0 if nothing is written.

4 - Play and export
   Here the video can be played with a requested fps. If the requested framerate is faster than the refresh rate of the app,
   the video is instead played at the refresh rate. The "frame step" input field determines how many frames should be advanced when changing
   frame. Setting this to e.g. 5 will skip over 4 frames at a time and display the 5th, allowing the video to be played faster if the refresh
   rate is not fast enough. Setting this to -1 also allows to play the video in reverse.

   The video can be captured and exported as mp4 by clicking record and choosing an output path

5 - Video slicing
   These settings control what should be tracked. Off means no tracking, while One means track the current frame only (requiring to retrack
   every time the frame is changed), All tracks the whole video (which might take some time, depending on settings) while Range restricts
   the video to a subset defined by two additional fields that pop up when Range is selected. When the video is restricted by Range,
   tracking is only performed for the current range. Range also has additional interactions, for example when playing / recording a video
   set to Range, only the selected range will be played / recorded.

6 - Primary detection mechanism
   Here the primary flavor of tracking can be selected - either :ref:`Trackpy-like <batch>` or :ref:`Laplacian of the Gaussian <LoG>`. This
   Choice affects what other options are displayed.

7 - All Options
   All Options toggles whether to show all options for the chosen tracking method, Or to limit shown options to the bare minimum. The
   detailed options reflect the parameters to the python functions :ref:`LoG <LoG>` and :ref:`batch <batch>`

8 - Primary size options
   Here, the primary size options are shown. Diameter in the case of Trackpy and min_radius / max_radius in the case of LoG.

9 - Thresholding and linking options
   Here, SNR and Area SNR can be chosen, corresponding to the "snr" and "minmass_snr" parameters of the python API, respectively. For more
   information, refer to the :ref:`Python API <batch>`.

10 - Submit
   In order for changes to the tracking parameters to take effect, they need to be submitted by clicking submit or hitting Enter on the keyboard.

11 - Zoom and reset zoom
   The video can be zoomed by holding left click while dragging the mouse to form a zoom window. To reset zoom, hit this button or rightclick anywhere
   in the video field

12 - Color Options
   Toggles the display of the color options, listed here as 15, 16, 17 and 18.

13 - Tracks for all particles
   Toggles if all particle tracks up until the current frame should be displayed. If off, only the tracks for currently "alive" particles will be drawn.
   A particle is alive if its track started before the current frame and ends after the current frame.

14 - Image colormap min / max
   Here, the minimum and maximum of the image colormap can be set. This is fairly similar to adjusting brightness in Fiji, and is important
   to set in many cases, as they default to the current frame's min/max if not set. This can cause flickering of the video as it is played,
   as there will always be some variation in the minimum and maximum of each frame. It is especially important to set these if there is a
   tendency in the brightness of the video, as a general increase / decrease. These effects will be visually lost if each frame is allowed
   to set its own min/max.

15 - Circle color
   This sets the color of the circles drawn around detections. It is especially helpful to set this to different colors when windows are
   coupled to eachother, as it will be impossible to distinguish which detections are from which video if they are all white. The alpha
   can be turned all the way down to hide detections.

16 - Image colormap
   The colormap used for the image data. The image is always 1-dimensional intensity values. These can be mapped to colors by using setting
   a minimum and maximum for the intensities (point 14 here) and using a lookup table known as a colormap, which maps this range to a range
   of colors. All available colormaps in matplotlib have been included in gpu_tracking, but popular choices are viridis (the default), greys
   and inverted greys, both of which show the data in grey-scale rather than color.

17 & 18 - Track colormap
   Like the image, tracks can be colored by a colormap. 17 chooses which colormap is used, while 18 controls whether to color each segment in
   the track according to the tracks local lifetime (such that the first segment is always 0 in the colormap, and the last segment 1), or according
   to the length of the video (all tracks observed in the middle of the video will be approximately the same color, but different from those observed
   in the beginning or end of the video).

19 - Clone
   Clones the window, producing a new video with the same settings. The new window will be coupled to the original window, showing the data of
   both windows in the new one. By changing the circle color and video or tracking settings of the new (or old) window, it is possible compare
   the same tracking of two different videos, e.g. different channels in the same experiment. One can also compare different tracking settings
   for the same video.

20 - Copy python command
   Exports the current settings by copying a command to the clipboard that does the same tracking as the currently active one in the python API.
   Simply ctrl-V to use these settings in a script.

21 - Output data to csv
   Saves the current tracking to a csv file, which can later be loaded by e.g. pandas or any other csv reader.
