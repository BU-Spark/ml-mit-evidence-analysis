import tensorflow as tf
from models.segmentation.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
import torch
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

def blood_classifier(image, model, transform):
    result = inference_detector(model, image)
    
    pred_instances = result.pred_instances

    CLASSES = ('class1', 'class2', 'blood', 'class4', 'class5', 'class6')
    highest_confidence = 0
    detected_class = None

    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()

    for score, label in zip(scores, labels):
        if score > highest_confidence:
            highest_confidence = score
            detected_class = CLASSES[label]

    if detected_class == 'blood':
        classification = True
    else:
        classification = False

    return classification