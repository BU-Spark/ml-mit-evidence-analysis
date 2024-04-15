import tensorflow as tf
from models.segmentation.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def create_image_name(labels):
    return '_'.join(labels) + '.jpg'


def crop_feature(image, bbox):

    # Ensure bbox coordinates are within image bounds
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(image.shape[1]-1, x_max), min(image.shape[0]-1, y_max)

    if x_max <= x_min or y_max <= y_min:
        return None  # Return None if the bounding box is invalid

    crop = image[y_min:y_max, x_min:x_max]
    return crop

def load_models():

    # Load alcohol model
    #alcohol_model = tf.keras.models.load_model("models/alcohol/alcohol_model.keras")

    # Load SAM model
    sam_checkpoint = "models/segmentation/SAM_vit_model.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = SamAutomaticMaskGenerator(sam)

    # Load knife model (98% accuracy)
    with open('models/weapon/knife_model.pt', 'rb') as file:
        knife_model = torch.load(file, map_location='cpu')

    # Load knife model (97% accuracy)
    with open('models/weapon/pistol_model.pt', 'rb') as file:
        pistol_model = torch.load(file, map_location='cpu')

    # Define the transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


    # Face-classifier model
    # Load the model
    with open('models/face/face_model.pkl', 'rb') as f:
        face_clf, face_labels = pickle.load(f)


    return sam, knife_model, pistol_model, transform_pipeline, face_clf, face_labels