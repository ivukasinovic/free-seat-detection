import os
import cv2
import numpy as np
from pathlib import Path

import skimage.io

ROOT_DIR = Path(".")

weights = os.path.join(ROOT_DIR, "yolo.weights")
config = os.path.join(ROOT_DIR, "yolo.cfg")
classesTxt = os.path.join(ROOT_DIR, "yolo.txt")
image = os.path.join(ROOT_DIR, "miami/miami3.jpg")
IMAGE_DIR = os.path.join(ROOT_DIR, "miami")



def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

file_names = next(os.walk(IMAGE_DIR))[2]

for file in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
    image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.imread(image)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classesTxt, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    borders = []
    conf_threshold = 0.6
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id in [56, 60,0]:
                confidence = scores[class_id]
                if confidence > 0.6:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    borders.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(borders, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        border = borders[i]
        draw_prediction(image, class_ids[i], confidences[i], round(border[0]), round(border[1]), round(border[0] + border[2]), round(border[1] + border[3]))

    cv2.imshow("chair,person detection", image)
    cv2.waitKey()
