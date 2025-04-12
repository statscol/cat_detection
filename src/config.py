import os
from typing import Dict, List

import cv2
from pydantic_settings import BaseSettings


class CameraConfig(BaseSettings):
    URL: str = os.getenv("CAMERA_URL", "")
    USERNAME: str = os.getenv("CAMERA_USER", "")
    PASSWORD: str = os.getenv("CAMERA_PASS", "")


class DetectorInferenceConfig(BaseSettings):

    YOLO_MODEL_NAME: str = "yolo11m.pt"
    # if using yoloworld
    # YOLO_DEFAULT_CLASSES: List = ["cat"]
    YOLO_DEFAULT_CLASSES: List = [15]
    FRAMES_UPDATE_ON_VIDEO: int = 30
    CV2_DEFAULT_TEXT_ARGS: Dict = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1,
        "color": (255, 255, 255),
        "thickness": 2,
    }
    CV2_DEFAULT_TEXT_BOX_ARGS: Dict = {
        "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
        "fontScale": 1,
        "thickness": 2,
    }

    OFFSET_BOX_CV2: int = 4


DEFAULT_VID_SIZE_IN_MB = 50
