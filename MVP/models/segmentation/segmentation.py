import cv2
import numpy as np

def crop_feature(image, bbox):
    # Ensure bbox coordinates are within image bounds
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1]-1, x_max), min(image.shape[0]-1, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None  # Return None if the bounding box is invalid

    # Crop the image using the bounding box coordinates
    crop = image[y_min:y_max, x_min:x_max]
    return crop

def extract_features(mask):
    features = {}
    
    # Area
    features['area'] = np.sum(mask)
    
    # Centroid
    M = cv2.moments(mask.astype(np.uint8))
    features['centroid'] = (M['m10'] / M['m00'], M['m01'] / M['m00']) if M['m00'] != 0 else (0, 0)
    
    # Bounding Box
    y, x = np.where(mask)
    features['bounding_box'] = [min(x), min(y), max(x), max(y)]  # minX, minY, maxX, maxY
    
    # Aspect Ratio
    bbox_width = max(x) - min(x)
    bbox_height = max(y) - min(y)
    features['aspect_ratio'] = bbox_width / bbox_height if bbox_height > 0 else 0
    
    return features['bounding_box']


def segment(image, mask_generator):
    masks = mask_generator.generate(image)

    bbox_list = []
    for i in range(len(masks)):
        mask_data = masks[i]['segmentation']
        bbox_list.append(extract_features(mask_data))
    
    return bbox_list



