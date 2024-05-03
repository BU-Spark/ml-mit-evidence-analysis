import pickle
import cv2
from models.face.face_recognition import face_locations, face_encodings
import pandas as pd
import numpy as np

COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCALE = 1

def classify_face(image, clf, labels):
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    locs = face_locations(image, number_of_times_to_upsample=N_UPSCALE)
    if len(locs) == 0:
        return None, None
    face_encodings_array = face_encodings(image, known_face_locations=locs)

    # Predict attributes for detected faces
    pred = pd.DataFrame(clf.predict_proba(face_encodings_array), columns=labels)
    pred = pred.loc[:, COLS]

    # Determine the gender
    genders = ['Male' if gender_prob >= 0.5 else 'Female' for gender_prob in pred['Male']]

    # Determine the race
    max_value = pred.iloc[:, 1:4].max(axis=1)
    race = pred.iloc[0, 1:4].idxmax()

    # Pair up genders and races for each face
    # attributes = list(zip(genders, races))
    
    return genders[0], race

# Example usage:
# attributes = classify_face(image)
# print(attributes)
