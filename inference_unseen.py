import logging
from typing import List, Optional
import numpy as np

from matplotlib.patches import Rectangle
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

from detectron2.engine.defaults import DefaultPredictor
from mask2former.modeling import *
from detectron2.config import get_cfg
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
import cv2
import torch, sys

from detectron2.config import LazyConfig, instantiate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = get_cfg()
cfg.set_new_allowed(True) 
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

cfg.merge_from_file("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
cfg.MODEL.DEVICE = "cuda"

# set input image size
# NEW TRAINING PIPELINE
# cfg.INPUT.MIN_SIZE_TEST = 1024
# cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.freeze()

# init predictor
predictor = DefaultPredictor(cfg)
# DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask-rcnn-dragonfly/output_512_dragonfly_2025-09-28_13-30-18/model_final.pth")
DetectionCheckpointer(predictor.model).load("/home/mrajaraman/master-thesis-dragonfly/external/mask2former-dragonfly/output_512_dragonfly_2025-10-02_01-01-28/model_final.pth")
print("Model loaded successfully.") 

category_mapping={"1": "head", "2": "torso", "3": "tail", "4": "wings"}
category_names = list(category_mapping.values())

test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/test_batch.csv")

for image_path in test_images['Path']:
    image = np.array(cv2.imread(image_path, cv2.IMREAD_COLOR))
    np_image = image.copy()
    image_id = image_path[47:-4]
    print("Inference done on ", image_id)

    prediction_result = predictor(image)
    print(prediction_result)
    original_predictions = prediction_result

    # PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
    print(predictor.aug)
    print()

    # Note that our data loader ONLY USES THE cfg.INPUT.IMAGE_SIZE key, MIN_SIZE_TRAIN and MAX_SIZE_TRAIN are ignored 
    # print(predictor.cfg.INPUT.IMAGE_SIZE)
    # print()

    # Output sample mask predictions:
    sample_preds = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
    print(sample_preds)
    # print(sample_preds.shape)
    # print(sample_preds.dtype)

    outputs = prediction_result["instances"]

    # print(type(outputs))
    # print(type(image))

    v = Visualizer(image[:, :, ::-1], metadata=None, scale=1.0)
    out = v.draw_instance_predictions(outputs.to("cpu"))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.show()

    plt.title("Inference Result of Mask2Former on image")
    plt.savefig(f"inferences_28/inference_{image_id}.png")