import logging
from typing import List, Optional
import numpy as np

from matplotlib.patches import Rectangle
from PIL import Image
import matplotlib.pyplot as plt

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask, get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements
from detectron2.checkpoint import DetectionCheckpointer

logger = logging.getLogger(__name__)
# import sys
# sys.path.insert(0, "/home/mrajaraman/do-not-modify/MassID45/Mask2Former/Mask2Former-MassID45")

from detectron2.engine.defaults import DefaultPredictor
from mask2former.modeling import *
from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
import cv2
import torch, sys

from detectron2.config import LazyConfig, instantiate

cfg = get_cfg()
cfg.set_new_allowed(True) 
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("/home/mrajaraman/do-not-modify/MassID45/Mask2Former/Mask2Former-MassID45/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
# cfg.MODEL.WEIGHTS = "/h/jquinto/Mask2Former/model_final_R50.pkl"

# set model device
# cfg.MODEL.DEVICE = "self.device.type"
cfg.MODEL.DEVICE = "cuda"

# set input image size
# NEW TRAINING PIPELINE
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.freeze()

# init predictor
predictor = DefaultPredictor(cfg)
DetectionCheckpointer(predictor.model).load("/home/mrajaraman/Code/model_checkpoints/model_final_mask2former.pth") #, weights_only=False) #, map_location=device)
print("Model loaded successfully.")
category_mapping={"1": "b"}

# detectron2 category mapping
category_names = list(category_mapping.values())
image = np.array(
    cv2.imread(
        "/home/mrajaraman/dataset/originals/img_931176390.jpg",  
        cv2.IMREAD_COLOR
))

# if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
#     # convert RGB image to BGR format
#     image = image[:, :, ::-1]

prediction_result = predictor(image)

img = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
bbox = []
print(prediction_result['instances']._fields['pred_boxes'].tensor)
print("Number of detected instances is ", len(prediction_result['instances']._fields['pred_boxes'].tensor))
for i in prediction_result['instances']._fields['pred_boxes'].tensor:
        print("i is ", i)
        bbox.append(i.cpu().detach().numpy())

# Display the image
plt.imshow(Image.open('/home/mrajaraman/dataset/originals/img_931176390.jpg'))

# Add the patch to the Axes
for i in range(len(bbox)):
        # print("length is ", len(bbox))
        plt.gca().add_patch(Rectangle((bbox[i][0], bbox[i][1]), bbox[i][2]-bbox[i][0], bbox[i][3]-bbox[i][1], linewidth=1, edgecolor='r', facecolor='none'))
        print(f"Instance {i+1} added to image")

# plt.gca().add_patch(Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none'))
# from PIL import Image
plt.title("Inference Result of Mask2Former on image")
plt.savefig("inference_mask2former.png")

# original_predictions = pre
# original_predictions = prediction_result

# # PROOF THAT RESIZING WORKS AS EXPECTED DURING INFERENCE
# print(predictor.aug)
# print()

# # Note that our data loader ONLY USES THE cfg.INPUT.IMAGE_SIZE key, MIN_SIZE_TRAIN and MAX_SIZE_TRAIN are ignored 
# print(predictor.cfg.INPUT.IMAGE_SIZE)
# print()

# Output sample mask predictions:
sample_preds = prediction_result['instances'][0]._fields['pred_masks'].cpu().detach().numpy()
# print(sample_preds)
# print(sample_preds.shape)
# print(sample_preds.dtype)