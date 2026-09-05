## Idea 1 – VLM for Low-Vision Users

### Project Summary

Images taken by low-vision users may be blurry, dark, poorly framed, or missing important information. However, current VLMs may still give a confident answer even when they cannot clearly understand the image. The idea is to develop a VLM that uses the image, the user’s question, and OCR text to answer when possible and abstain when there is not enough visual information.

**Dataset:** [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/)

**Use Case:** Help low-vision users ask questions about objects, labels, signs, and other things in an image.

**Gap Filled:** Current VLMs can hallucinate answers when an image is unclear or incomplete. This project focuses on making the model recognize when it is uncertain instead of always forcing an answer.

**Target Users:** Blind and low-vision users who use visual-assistance applications.

### Workflow

[Image + Question] → [OCR Text Extraction] → [VLM] → [Confidence Check] → [Answer or Abstain]

### Roadmap

**Phase 1: Data Preparation**

Load the VizWiz images, questions, answers, and answerability labels.

**Phase 2: Baseline VLM**

Test a pretrained VLM such as BLIP-2 or PaliGemma to see how well it answers the questions without fine-tuning.

**Phase 3: OCR Augmentation**

Extract text from the images using OCR and provide it as extra information to the VLM.

**Phase 4: Fine-Tuning and Confidence**

Fine-tune the model using LoRA and train it to predict whether an image contains enough information to answer the question.

**Phase 5: Evaluation**

Compare VQA accuracy, answerability detection, hallucination rate, and confidence calibration.

## Idea 2 – Building-Damage Assessment

### Project Summary

After a wildfire, earthquake, flood, or hurricane, checking the damage to every building manually can take a lot of time. This project would use before-and-after satellite images to locate buildings and classify them as having no damage, minor damage, major damage, or being destroyed.

**Dataset:** xBD/xView2 disaster dataset

**Use Case:** Create an initial damage map that helps response teams see which areas may need to be inspected first.

**Gap Filled:** A damage-detection model may work on locations and disasters included in its training data but perform poorly on a new disaster. This project would focus on whether the model can generalize to disaster events and locations it has not seen before.

**Target Users:** Disaster-response teams, emergency-management agencies, and humanitarian organizations.

### Workflow

[Before Image + After Image] → [Siamese Image Encoder] → [Feature Comparison] → [Building Segmentation] → [Damage Classification]

### Roadmap

**Phase 1: Data Preparation**

Load the before-and-after xBD images, building locations, and damage labels.

**Phase 2: Building Segmentation**

Train a U-Net or SegFormer model to locate buildings in the satellite images.

**Phase 3: Damage Classification**

Use a Siamese model to compare the before-and-after images and classify each building’s damage level.

**Phase 4: Generalization Testing**

Train the model on some disaster events and test it on a completely separate event or disaster type.

**Phase 5: Evaluation**

Measure building segmentation IoU, damage-classification F1, and performance on disasters not included in training.
