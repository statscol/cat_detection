from collections import Counter

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from utils import get_logger, percent_crop

logger = get_logger("classifier")

# top 2 colors found in the cat
NUM_CLUSTERS = 2
CROP_MAX_SIZE = (250, 250)


class CatClassifier:
    def __init__(self, center_crop: bool = True):
        self.id2label = {0: "gandalf", 1: "kratos", 2: "monstruo", 3: "jupyter"}

        # defined by taking a look at pictures and finding
        self.color_centers = np.array(
            [[170, 170, 170], [10, 10, 10], [40, 40, 40], [60, 60, 60]]
        )
        # in case we wanted to use the top-2 most frequent colors of the cat in RGB, we can also just use one of the colors
        # self.color_centers=np.array([[[160,160,160],[150,150,150]],[[10,10,10],[60,60,60]],[[40,40,40],[100,100,100]],[[60,60,60],[150,150,150]]]) # noqa: E501
        self.default_label = "unidentified"
        self.center_crop = center_crop

    @staticmethod
    def find_color_clusters(img_array):
        clusters = KMeans(n_clusters=NUM_CLUSTERS)
        # clusters using pixel values for RGB channels
        clusters.fit(img_array.reshape(-1, 3))
        return clusters

    def get_most_similar(self, clusters):
        """uses most common color cluster and gets the most similar cat
        finding the min euclidean distance between each other"""
        try:
            # get the top 1 most frequent colors
            most_common = Counter(clusters.labels_.tolist()).most_common()[:1]
            cluster_center = np.stack(
                [clusters.cluster_centers_[idx] for idx, center in most_common]
            )
            # euclidean distance, equivalent to np.sqrt((np.abs(self.color_centers-cluster_center)**2).sum(axis=1))
            # here we could have used two color centers and take the mean euclidean distance from reference colors to cluster colors # noqa: E501
            # dist=np.linalg.norm(self.color_centers-cluster_center,axis=1).mean(axis=1)
            dist = np.linalg.norm(self.color_centers - cluster_center, axis=1)
            prediction = int(dist.argmin())
            return self.id2label[prediction]
        except Exception as e:
            logger.exception(f"Exception during inference {e}")
            return self.default_label

    @staticmethod
    def valid_input(img: str | np.ndarray):
        assert isinstance(img, (str, np.ndarray)), ValueError(
            f"img input expected str or np.ndarray, got {img.__class__}"
        )
        if isinstance(img, str):
            img = Image.open(img)
            if not all(d <= CROP_MAX_SIZE[0] for d in img.size):
                # make sure we only use max 250x250 crops
                img = img.resize(CROP_MAX_SIZE)
            img = np.asarray(img)
        if isinstance(img, np.ndarray):
            if not all(d <= CROP_MAX_SIZE[0] for d in img.shape):
                img = cv2.resize(img, CROP_MAX_SIZE, interpolation=cv2.INTER_AREA)
        return img

    def __call__(self, img: str | np.ndarray):
        img = self.valid_input(img)
        if self.center_crop:
            img = percent_crop(img)
        clusters_img = self.find_color_clusters(img)
        return self.get_most_similar(clusters_img)

    def __str__(self):
        return "CatClassifier()"


if __name__ == "__main__":
    clf = CatClassifier()

    from pathlib import Path

    from tqdm import tqdm

    files = list(Path("data/images/jupyter").glob("*.*g"))
    preds = []
    for file in tqdm(files, desc="Generating Labels"):
        if file.is_file():
            preds.append(clf(str(file)))
    message = (
        f"Accuracy: {sum([True for p in preds if p=='jupyter'])/len(list(files)):.3%}"
    )
    logger.info(message)
