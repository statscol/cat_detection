import logging


def percent_crop(image, pct: float = 0.7):
    height, width = image.shape[:2]

    # Calculate new dimensions (70% of original)
    new_width = int(width * pct)
    new_height = int(height * pct)

    # Calculate top-left corner of the crop (to keep it centered)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2

    # Crop the image
    cropped_image = image[
        start_y : (start_y + new_height), start_x : (start_x + new_width)
    ]
    return cropped_image


def preprocess_yolo_boxes(boxes):
    """preprocess yolov8 ultralytics boxes in format xywh which have x center,y_center width, height"""
    boxes_xywh = [
        [int(x_c - (w / 2)), int(y_c - (h / 2)), w, h] for (x_c, y_c, w, h) in boxes
    ]
    return boxes_xywh


def crop_image(box, image):
    "box: list| array with xyxy positions"
    x_i, y_i, x_f, y_f = box.astype(int)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[y_i:y_f, x_i:x_f]


def get_logger():
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger
