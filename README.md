# Cat Detection using Zero shot object detection

Want to keep track of your cats when you're not at home? This repo demonstrates how to perform cat detection using YOLO-World as a zero-shot object detector, along with a color-based classifier using K-means clustering and OpenCV.

<br>

![demo](./data/demo_detection.gif)

<br>

- In this approach, the object detector is responsible for detecting the cat. Then, a classifier predicts the cat label by finding the most similar color among the clusters identified in the image.
- This approach requires manually inspecting the pixels of several images and calculating the average RGB values expected for a particular cat. For example, white cats should have a color center close to [160, 160, 160], while black cats should have a color center near [5, 5, 5].
- For inference, a center crop is applied, focusing on the central 70% of the image (this avoids adding noise to the clusters)

# Setup

- Using uv `pip install uv` and then `uv sync`
- Setup a .env with your CameraConfig (see src/config.py)


- Running object detector + classifier

```bash
uv run --env-file=.env src/detector.py
```

- Test classifier on image crops:

```bash
uv run --env-file=.env src/classifier.py
```


# Limitations

- Using color clusters will not work at night, training a classifier for this is suggested for 24/7 usage.
