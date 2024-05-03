import cv2
import numpy as np
import tensorflow

def get_prediction_alcohol(image: np.ndarray, alcohol_model : tensorflow.keras.models.Model) -> bool:
    """
    Predicts if the passed image contains alcohol, using passed model.

    Parameters:
    - image (np.ndarray): The image to be classified, as a NumPy array (cv2).
    – alcohol model (tensorflow.keras.models.Model) : Trained binary classification model
        that makes a prediction if there is alcohol in the image.


    Returns:
    - bool: True if the image contains alcohol, False otherwise.
    """
    image = cv2.resize(image, (160, 160))
    image = np.expand_dims(image, axis = 0)
    prediction = alcohol_model.predict(image)
    return prediction[0][0] > 0.5