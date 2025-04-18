import copy
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from torch.cuda import is_available
from ultralytics import YOLO

from base import BaseDetector
from classifier import CatClassifier
from config import CameraConfig, DetectorInferenceConfig
from utils import get_logger, preprocess_yolo_boxes

inference_config = DetectorInferenceConfig()


class YoloDetector(BaseDetector):
    def __init__(
        self,
        model_name: Optional[str] = inference_config.YOLO_MODEL_NAME,
        detector_classes: List[int] = inference_config.YOLO_DEFAULT_CLASSES,
        out_folder: str = "./inference",
        add_classifier: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = "0" if is_available() else "cpu"
        self.out_folder = Path(out_folder)
        self.detector_classes = detector_classes
        self.add_classifier = add_classifier
        self.out_folder.mkdir(exist_ok=True, parents=True)
        # this is meant to be used/adapted with a classifier on top of
        # yolo detections. skip it for now
        self.logger = get_logger("detector")
        self.load_model()
        self.inference_method = {"image": self.model, "video": self.model.track}

    def load_model(self):
        self.model = YOLO(self.model_name)
        # if using yoloworld
        # self.model.set_classes(self.detector_classes)
        self.logger.info(
            f"Loaded Detector for {','.join(self.model.names[c] for c in self.detector_classes)}"
        )
        self.classifier = CatClassifier() if self.add_classifier else None
        self.logger.info(f"Loaded Classifier: {self.classifier}")

    def detect_image(
        self,
        image: Union[str, np.ndarray],
        save_to_disk: bool = False,
        save_to_db: bool = False,
        save_crop: bool = False,
        method: str = "image",
    ):
        """detect objects in an image provided using the @param image, can be a file path or a np.ndarray(cv2 image)"""
        image = cv2.imread(image) if isinstance(image, str) else image
        frame = copy.deepcopy(image)
        detections = self.inference_method[method](
            source=frame,
            device=self.device,
            persist=method == "video",
            conf=0.2,
            verbose=False,
            classes=self.detector_classes,
            save_dir=str(self.out_folder),
            project=str(self.out_folder),
        )

        boxes, names = detections[0].boxes, detections[0].names
        # account for frames where there are no detections
        if boxes is not None:
            boxes_id = (
                None
                if boxes.id is None
                else {
                    int(x.id.cpu().item()): names[int(x.cls.cpu().item())]
                    for x in boxes
                }
            )
            boxes = preprocess_yolo_boxes(boxes.xywh.cpu().numpy().astype(int))
            frame = self.render_boxes(
                image,
                boxes,
                ids=boxes_id,
                save_to_disk=save_to_disk,
                save_to_db=save_to_db,
                save_crop=save_crop,
                face=False,
            )
        return frame

    def process_videoframe(self, frame, **kwargs):
        return self.detect_image(frame, **kwargs)


if __name__ == "__main__":
    detector = YoloDetector(
        model_name=inference_config.YOLO_MODEL_NAME,
        add_classifier=True,
    )
    # save_to_disk to save frames
    # save_crop to save crops
    # save_to_db to use sqllite for logging
    config = {
        "save_to_disk": False,
        "method": "video",
        "save_crop": False,
        "save_to_db": False,
    }

    results = detector.detect_camera(camera_config=CameraConfig(), **config)
