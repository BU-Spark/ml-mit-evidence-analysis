# Standard Libaries
import math
import cv2
import os

# External Libraries
from models.segmentation.segmentation import segment
from lib.helper import *

from models.alcohol.alcohol_model import get_prediction_alcohol
from models.weapon.weapon_model import classify_knife, classify_pistol
from models.face.face_model import classify_face


# Model Loaders

segmentation_model, knife_model, pistol_model, transform_pipeline, face_clf, face_labels, alcohol_model= load_models()



def main(directory):


    # create results directory if it doesnt exist
    results_dir = os.path.join(directory, "results")

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)


    label_importance ={
        'alcohol': 100,
        'weapon': 100,
        'blood': 100,
    }

    result_dict = {}

    # List to keep track of non-image or corrupted files
    non_image_files = []  

    for filename in os.listdir(directory):

        # Filtering common image extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"File not supported: {filename}")
            non_image_files.append(filename)
            continue

        file_path = os.path.join(directory, filename)
        image = cv2.imread(file_path)

        # cv2.imread() returns None if it fails to read the image
        if image is None:
            print(f"File is not an image or is corrupted: {filename}")
            non_image_files.append(filename)
        else:
            print(f"Processing image: {filename}")


            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            labels = []


            # Segment the image into objects
            bbox_list = segment(image, segmentation_model)

        #Iterate through all the objects
        for ind, bbox in enumerate(bbox_list):

            try:
                # Crop the objects out of the image
                cropped_image = crop_feature(image, bbox)
                
                if (ind == 3):
                    cv2.imwrite("cropped_pistol.jpg", cropped_image)

                if cropped_image is not None:
                    
                    # if successful, run through all the models

                    # has_weapon = weapon(cropped_image)

                    has_alcohol = get_prediction_alcohol(image, alcohol_model)

                    has_pistol = classify_pistol(cropped_image, pistol_model, transform_pipeline)
                    has_knife = classify_knife(cropped_image, knife_model, transform_pipeline)
                    face_classifier = classify_face(cropped_image, face_clf, face_labels)


                    #     TESTING
                    #print(has_knife)
                    #print(has_pistol)
                    #print(face_classifier)
                    #print(has_alcohol)
                    #print('-------')

                    if(has_alcohol) and ('alcohol' not in labels):
                        labels.append('alcohol')
                    
                    if(has_knife) and ('knife' not in labels):
                        labels.append('knife')

                    if(has_pistol)and ('gun' not in labels):
                        labels.append('gun')

                    if(face_classifier != (None, None)):
                        labels.append(face_classifier[0], face_classifier[1])
            

            except Exception as e:
                # If an error occurred during cropping, skip this mask and continue
                (f"Skipping mask due to error: {e}")


        if(labels):
            image_name = create_image_name(labels)
        else:
            image_name = 'nothing.jpg'
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(results_dir, image_name), image)


            # Placeholder for image processing code
            # You can perform your operations on the image here.
            # Example: cv2.imshow("Image", image)
            # cv2.waitKey(0)
            # Don't forget to release windows with cv2.destroyAllWindows() if you use imshow

    if non_image_files:
        print("\nSummary of non-image or corrupted files:")
        for file in non_image_files:
            print(file)
    else:
        print("\nAll files were successfully processed.")


main("sample_images")

# image = cv2.imread('cropped_pistol.jpg')

# print(type(image))

# has_pistol = classify_pistol(image, pistol_model, transform_pipeline)

# print(has_pistol)