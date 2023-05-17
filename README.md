# gpu_tracking - Portable GPU accelerated single particle tracking

gpu_tracking attempts to make single particle tracking as easy and user friendly as possible, while still maximally leveraging the available hardware
to perform the computation-heavy image processing needed for the tracking task. Key features are

1. A GUI to more easily work with the microscopy data - detections can be made and visualized extremely fast, enabling iteratively tuning parameters
by visually inspecting the detections as they are made.
2. A convenient python API, making it very easy to automate dataprocessing across many videos
3. Ability to link particles while detecting, connecting them through frames and forming tracks
4. Written in Rust with wgpu, allowing gpu_tracking to run on any operating system and on GPUs from any vendor.
Performant on anything from an HPC to a laptop

# Installation
## Python package & GUI
The primary way to install gpu_tracking is through pip, which will install both the python package that exposes gpu_tracking, as well as the GUI.
As development is ongoing, it is recommended to install with `-U`, to upgrade the package if it is already installed.

```
pip install gpu_tracking -U
```

After installing, the GUI can be launched with the terminal command
```
gpu_tracking
```
File names can optionally be provided to the command to automatically open them in the GUI.

## Rust crate
Currently, gpu_tracking is not yet exposed as a Rust crate, as there has not yet been a use for that. If this is something you would like to see,
feel free to start an issue, and I'll try to get it up on crates.io as soon as possible. Until such time, feel free to clone or fork the repo and
adapt the package to your use case.

# Documentation
Documentation for gpu_tracking can be found at https://amfaber.github.io/gpu_tracking_docs/
