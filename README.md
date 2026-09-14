# Geometric&3DComputerVision

**Course:** Geometric and 3D Computer Vision (a.y. 2025–2026)
**Author:** *Lorenzo Pastore*

## Overview

This project implements a simple **structured-light 3D laser scanner** using a fixed camera,
a hand-swept laser line, and two printed rectangular markers acting as reference planes. For
every video frame, the laser line is segmented and split into the portion falling on each
marker and the portion falling on the object. The marker portions are used to estimate the
laser plane in 3D for that frame; the object portion is then triangulated against that plane to
recover colored 3D points, which are accumulated over the whole video into a single point
cloud and exported as a `.ply` file.

## Repository Structure

```
.
├── README.md
├── functions.py              # All core algorithms (see "Pipeline" below)
├── main.ipynb                # Notebook driving the scanning pipeline end-to-end
├── LaserScanner_project_data/   # (not submitted) provided videos + calibration + marker
│   ├── calibration/
│   │   ├── K.txt
│   │   └── dist.txt
│   └── data/
│       ├── cup1.mp4
│       ├── cup2.mp4
│       ├── puppet.mp4
│       └── soap.mp4
└── output1.ply                # Reconstructed point cloud (generated)
```


## Requirements

- Python 3.9+
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib (used for `matplotlib.path.Path`, point-in-polygon tests)
- Pandas
- [PyntCloud](https://github.com/daavoo/pyntcloud) (PLY export)

```bash
pip install numpy opencv-python matplotlib pandas pyntcloud
```

## Usage

1. Place the provided videos and calibration files under `LaserScanner_project_data/` as
   shown in the folder structure above.
2. Open `main.ipynb` and run all cells, or adapt the driving loop into a standalone script.
3. Edit the paths/constants in the initialization cell to select which video to process:

   ```python
   K_PATH = "LaserScanner_project_data/calibration/K.txt"
   DIST_PATH = "LaserScanner_project_data/calibration/dist.txt"
   VIDEO_PATH = "LaserScanner_project_data/data/cup1.mp4"
   ```

4. At the end of the run, the reconstructed colored point cloud is written to `output1.ply`
   (validated to open correctly in [MeshLab](http://www.meshlab.net)).