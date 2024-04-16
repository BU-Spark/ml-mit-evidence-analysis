import cv2
import torch
from PIL import Image

def classify_knife(image, model, transform):
    # Read the image using cv2
    # image = cv2.imread(image_path)
    # Convert the BGR image from cv2 to RGB, and then to PIL Image for compatibility with torchvision transforms
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Apply the preprocessing transformations
    image = transform(image)
    # Add an extra batch dimension since pytorch expects batches
    image = image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model = model.to(device)
    
    # Set the model to evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        # Model inference
        outputs = model(image)
        # Assuming your model outputs logits, apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get the predicted class
        _, predicted = torch.max(probabilities, 1)
    
    # 1 = 'knife', 0 = 'non_knife'
    return True if predicted.item() == 1 else False


def classify_pistol(image, model, transform):
    # Read the image using cv2
    # image = cv2.imread(image_path)
    # Convert the BGR image from cv2 to RGB, and then to PIL Image for compatibility with torchvision transforms
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Apply the preprocessing transformations
    image = transform(image)
    # Add an extra batch dimension since pytorch expects batches
    image = image.unsqueeze(0)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    image = image.to(device)
    model = model.to(device)
    
    # Set the model to evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        # Model inference
        outputs = model(image)
        # Assuming your model outputs logits, apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get the predicted class
        _, predicted = torch.max(probabilities, 1)
    
    # 1 = 'pistol', 0 = 'non_pistol'
    return True if predicted.item() == 1 else False