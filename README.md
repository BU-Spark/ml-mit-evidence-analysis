# TEMPLATE-base-repo

Create a new branch from dev, add changes on the new branch you just created.

Open a Pull Request to dev. Add your PM and TPM as reviewers. 

At the end of the semester during project wrap up open a final Pull Request to main from dev branch.

# Spring 2024 Milestone

**Team member: Ilya Bulygin, Nurassyl Medeu, Jiawei Sun, Qinghui Zhu**

## Environment Requirements
- Python version: 3.11.1
- Modules requirements: requirements.txt
- To install all modules, please run ```pip install -r requirements.txt```

## Project Architecture

<img src="structure.png">

## Proof of Concept

This folder contains all Jupyter Notebooks for individual models training.
- ```ImageSegmentation``` folder contains Segment Anything Model (**SAM**) from Meta. This model is used for input image segmentation.
- ```face_classification``` folder contains **race** and **gender** classification model.
    - Original face classificaton repo can be found here [face-classification](https://github.com/wondonghyeon/face-classification)
    - Our face classification model has modified some training codes due to python version differences. Please consider using the training code as need.
- ```selfie``` folder contains facial feature classification model.
    - This model is not embedded in the filal deliverables, since the accuracy is not ideal. Further training on different dataset may have significant improvement on the model.
- ```Blood_detection(FasterCNN).ipynb``` file aims for FasterCNN model training on blood detection. Original github repo can be found here [HOD-Benchmark-Dataset](https://github.com/poori-nuna/HOD-Benchmark-Dataset/tree/main)
- ```HumanCharacterModel.ipynb``` file aims for
- ```Weapon.ipynb``` file aims for **knife** and **pistol** model training. Several approaches are applied regarding high accuracy model production.
  - Self defined ```CNN``` model was selected with highest accuracy (98%), but comes with a problem of overfitting. This model is a binary detection model, without relativity percentage functionality.
  - ```YOLOv8n``` pretrained model are tesed and specialized with knife and pistol detection, but only reaches accuracy around 60%.
        - knife.yaml file for knife dataset can be found [here](https://drive.google.com/drive/u/1/folders/1qqBHAOiLvLZebG4Ss23rq30j9ELt3rKY)
        - More YOLOv8 pretrained models can be downloaded [here](https://docs.ultralytics.com/tasks/detect/#models)
  - ```MMDetection``` toolbox are utilized, with [```faster_rcnn```](https://drive.google.com/drive/u/1/folders/1qqBHAOiLvLZebG4Ss23rq30j9ELt3rKY) model tested but not spcialy trained for weapon detection. Further work can be done here with the purpose of model specialization.
- ```alcohol_model.ipynb``` file aims for **alcohol** detection. 

## MVP

### Introduction

This folder contains complete final deliverable. Input images will be segmented, filtered, then passed into recognization model. After evaluation, output of each model will be combined and used for labeling and classification. All images will eventually be classified into different folders in ```results``` folder.

Currently supported classification types are: ```knife```, ```pistol```, ```alcohol```, ```male```, ```female```, ```asian```, ```white```, ```black```, ```blood```.

### Input Restriction

Currently, only ```png```, ```jpeg```, ```jpg``` image types are supported. After collecting images, put all into a folder under ```MVP``` folder for classification.

### Launch Program
To launch program, run ```python main.py``` in ```MVP``` folder.

## Models

All models can be downloaded from [here](https://drive.google.com/drive/u/1/folders/1w0zNPWbahkFEmBiexCbiwHZe9Yd0uCgk)
