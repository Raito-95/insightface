#!/bin/bash

if python -c "import cv2" &> /dev/null; then
    echo "OpenCV is already installed."

    pip install insightface
    pip install onnxruntime

    pip uninstall -y opencv-python-headless
else
    echo "OpenCV is not installed."

    pip install insightface
    pip install onnxruntime

    pip uninstall -y opencv-python-headless
    pip install opencv-python
fi
