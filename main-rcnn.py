import os
import numpy as np
import cv2
import mrcnn.config
from pathlib import Path

import mrcnn.utils
from mrcnn.model import MaskRCNN

import skimage.io

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.6

ROOT_DIR = Path(".")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = os.path.join(ROOT_DIR, "miami")


model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

model.load_weights(COCO_MODEL_PATH, by_name=True)

def get_chair_boxes(boxes, class_ids):
    chair_boxes = []

    for i, box in enumerate(boxes):
        if class_ids[i] in [57, 61]:
            chair_boxes.append(box)

    return np.array(chair_boxes)

def get_people_boxes(boxes, class_ids):
    people_boxes = []

    for i, box in enumerate(boxes):
        if class_ids[i] in [1]:
            people_boxes.append(box)

    return np.array(people_boxes)



file_names = next(os.walk(IMAGE_DIR))[2]

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

chair_boxes = None

image = skimage.io.imread(os.path.join(IMAGE_DIR, "init/miami.jpg"))
results = model.detect([image], verbose=1)
r = results[0]

#initialize free chairs
chair_boxes = get_chair_boxes(r['rois'], r['class_ids'])

for file in file_names:
    occupied = 0
    print("Processing image... ", file)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, file))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.detect([rgb_image], verbose=1)
    r = results[0]
    if chair_boxes is None:
        chair_boxes = get_chair_boxes(r['rois'], r['class_ids'])
    else:
        people_boxes = get_people_boxes(r['rois'], r['class_ids'])
        print('People detected..', people_boxes)
        overlaps = mrcnn.utils.compute_overlaps(chair_boxes, people_boxes)
        print("Overlaping matrix", overlaps)

    print("Chairs found in image:, length", len(chair_boxes))


    for person_box in people_boxes:
        y1, x1, y2, x2 = person_box
        cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(rgb_image, 'person', (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for chair_area, overlap_areas in zip(chair_boxes, overlaps):
        max_IoU_overlap = np.max(overlap_areas)
        y1, x1, y2, x2 = chair_area

        if max_IoU_overlap < 0.1:
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rgb_image, 'chair', (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            free_space = True
        else:
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            occupied +=1
    print("Total seats: ", len(chair_boxes), ". Free seats: ", len(chair_boxes)-occupied)
    cv2.imshow('image', rgb_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
