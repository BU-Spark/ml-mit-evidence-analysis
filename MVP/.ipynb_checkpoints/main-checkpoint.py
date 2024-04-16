# standard libaries
import cv2
import os

# external libraries
from models.segmentation.segmentation import segment
from lib.models_util import *
from lib.output_util import *

# custom models
from models.alcohol.alcohol_model import get_prediction_alcohol
from models.weapon.weapon_model import classify_knife, classify_pistol
from models.face.face_model import classify_face
from models.blood.blood_detection import blood_classifier



classes = ["knife", "pistol", "alcohol", "male", "female", "asian", "white", "black","blood"]

# loading models
segmentation_model, knife_model, pistol_model, transform_pipeline, face_clf, face_labels,blood_model = load_models()

def main(directory):

    create_output_folders(directory, classes)

    # keeping track of non-image or corrupted files
    non_image_files = []  

    for filename in os.listdir(directory):

        if filename == "results":
            continue

        # filtering common image extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"File not supported: {filename}")
            non_image_files.append(filename)
            continue

        file_path = os.path.join(directory, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"File is not an image or is corrupted: {filename}")
            non_image_files.append(filename)
            continue
        print(f"Processing image: {filename}")


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = {}
        for cl in classes:
            labels[cl] = False

        print("Now we start the segmentation.")

        # segmenting the image into objects
        bbox_list = segment(image, segmentation_model)
        print("Now segmentation finished.")

        # iterating through all the objects
        for ind, bbox in enumerate(bbox_list):

            try:
                # cropping the objects out of the image
                cropped_image = crop_feature(image, bbox)
                
                if (ind == 3):
                    cv2.imwrite("cropped_pistol.jpg", cropped_image)

                if cropped_image is not None:
                    #has_alcohol = get_prediction_alcohol(image, alcohol_model)
                    has_pistol = classify_pistol(cropped_image, pistol_model, transform_pipeline)
                    has_knife = classify_knife(cropped_image, knife_model, transform_pipeline)
                    image_gender, image_race = classify_face(cropped_image, face_clf, face_labels)
                    has_blood = blood_classifier(cropped_image,blood_model,transform_pipeline)

                    if has_knife:
                        labels["knife"] = True
                    if has_pistol:
                        labels["pistol"] = True
                    #if has_alcohol:
                        ##labels["alcohol"] = True
                    labels[image_gender] = True
                    labels[image_race] = True
                    if has_blood:
                        labels["blood"] = True

            except Exception as e:
                # if an error occurred during cropping, skip this mask and continue
                (f"Skipping mask due to error: {e}")

        for cl in labels:
            if labels[cl]:
                if cl is None:
                    print("Label is None, skipping.")
                    continue
                target_folder = os.path.join(directory, "results", cl)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                shutil.copy(os.path.join(directory, filename), os.path.join(target_folder, filename))


    if non_image_files:
        print("\nSummary of non-image or corrupted files:")
        for file in non_image_files:
            print(file)
    else:
        print("\nAll files were successfully processed.")

main("sample_images")