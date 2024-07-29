# Face Recognition Demo

This repository contains a demo for face recognition using InsightFace and OpenCV. The demo includes three main files:

- `installer_script.sh`: A shell script to set up the required environment.
- `demo.py`: A Python script to perform face recognition on a given video source.
- `config.json`: A configuration file to set up parameters for the demo.

## Requirements

Before running the scripts, ensure you have the following installed:

- Python 3.8 or higher

## Setup

### 1. installer_script.sh

This script will set up the required Python packages. It checks if OpenCV is already installed and handles the installation accordingly.

To run the setup script, execute:

```bash
bash installer_script.sh
```

### 2. config.json

This configuration file sets up parameters for the demo. Ensure the file `config.json` is present in the same directory as `demo.py` with the following content:

```json
{
  "data_dir": "./data/",
  "result_dir": "./result/",
  "ctx_id": 0,
  "video_source": 0,
  "fps": 10,
  "video_length_seconds": 10,
  "show_video": true
}
```

### 3. demo.py

This script performs face recognition on a given video source. It requires pre-saved images of known faces in the `./data/` directory.

#### Required Files

- **Known Faces**: Images of known faces should be placed in the `./data/` directory. The images should be in `.png` format, and the filenames (without extension) will be used as the names of the individuals.
- **Video Source**: The video source is specified in the `config.json` file under the `video_source` key. It can be a device index (e.g., 0 for the default camera) or a video file path.

#### Running the Script

To run the face recognition demo, execute:

```bash
python demo.py
```

The script will:

1. Load known face images from the `./data/` directory and build a database of face embeddings.
2. Open the video source specified by `video_source` in `config.json` and process it frame by frame.
3. Detect faces in each frame, compare them with the known face embeddings, and display the results.
