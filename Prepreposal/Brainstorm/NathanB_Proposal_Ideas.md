## Idea 1 – Context and Quality-Aware VLM for Low-Vision Users

### Project Summary

Images taken by blind or low-vision users may be blurry, dark, poorly framed, rotated, or partially blocked. Current VLMs may still give a confident answer even when they cannot clearly understand the image. This project would build a system that answers the user’s question when possible, but when it cannot, it identifies the specific problem with the image and explains how the user could take a better one.

What interests me about this project is trying to make AI reason more like a person. Humans use what they see, the question being asked, visible text, and previous experience to understand a situation. We also recognize when important information is missing instead of always guessing.

**Datasets:**

- [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/) for images, questions, answers, and answerability labels
- [VizWiz-QualityIssues](https://vizwiz.org/tasks-and-datasets/image-quality-issues/) for real images labeled as blurry, dark, too bright, poorly framed, obstructed, rotated, or unrecognizable

**Main Technology:**

- PaliGemma, BLIP-2, or another pretrained VLM
- ConvNeXt or Vision Transformer for multi-label image-quality classification
- PaddleOCR or EasyOCR for reading text
- LoRA for efficient VLM fine-tuning
- Confidence calibration and answerability prediction
- PyTorch and Hugging Face Transformers
- Gradio for the final demo

**Use Case:** A user could take a picture of a product, sign, package label, appliance setting, or other object and ask a question about it. The system would either answer or explain that the image is too dark, blurry, obstructed, or poorly framed and provide a specific recommendation.

**Gap Filled:** Existing visual-assistance applications can describe images and read text, but they may still provide unreliable answers when the visual evidence is unclear. This project focuses on recognizing why an image is not usable and communicating what information is missing instead of forcing an answer.

**Target Users:** Blind and low-vision users who use visual-assistance applications.

### Workflow

[Image + User Question] → [Image-Quality Classifier] → [VLM + OCR] → [Answerability and Confidence Check] → [Answer or Reason for Failure + Guidance]

### Roadmap

**Phase 1: Dataset Preparation**
Load and connect the VizWiz-VQA and VizWiz-QualityIssues annotations. Analyze the different quality problems and answerable versus unanswerable questions.

**Phase 2: Image-Quality Classification**
Train a multi-label classifier to detect blur, darkness, overexposure, poor framing, obstruction, and rotation.

**Phase 3: VLM and OCR Baseline**
Evaluate a pretrained VLM on VizWiz-VQA and add OCR text as extra context for questions involving signs, labels, or documents.

**Phase 4: Answerability and Guidance**
Train an answerability component that uses the VLM confidence and predicted image-quality problems. If the image is unanswerable, return the most likely reason and an appropriate suggestion.

**Phase 5: Evaluation and Demo**
Compare the original VLM against the quality-aware system using VQA accuracy, answerability average precision, quality-classification F1, hallucination rate, and confidence calibration.

## Idea 2 – Damage-and-Accessibility-Aware Disaster Mapping

### Project Summary

After a wildfire, earthquake, hurricane, or flood, response teams cannot immediately inspect every affected building or road. Once new satellite, airplane, or drone images become available, a model could compare them with images from before the disaster and create an initial map of damaged buildings and blocked roads.

Instead of only showing which buildings are damaged, this project would also examine whether those areas can be reached through the road network. The final system could highlight areas with severe damage and limited road access so response teams know where alternative routes or closer human review may be needed.

**Datasets:**

- xBD/xView2 for paired before-and-after satellite images, building locations, and damage labels such as no damage, minor damage, major damage, and destroyed
- [SpaceNet 8](https://spacenet.ai/sn8-challenge/) for before-and-after satellite imagery with buildings, roads, flooded buildings, and flooded-road labels
- Optional OpenStreetMap road data for additional road-network information

**Main Technology:**

- Siamese U-Net or Siamese SegFormer
- Change-detection and feature-fusion models
- Semantic segmentation for buildings, roads, and flooding
- Ordinal classification for building-damage severity
- NetworkX for road-graph construction and route analysis
- Rasterio and GeoPandas for satellite and geographic data
- PyTorch for model training
- Folium or Kepler.gl for the final map

**Use Case:** Emergency teams could use the generated map to identify areas with severe building damage, see which roads may be flooded or blocked, and determine where normal access routes may not be available.

**Gap Filled:** Many projects estimate building damage or road flooding separately. However, knowing where damage occurred does not automatically tell responders whether they can reach that area. This project combines both types of information to create a more useful disaster-response map.

**Target Users:** Emergency-management agencies, first-response teams, humanitarian organizations, and disaster-response researchers.

### Workflow

[Before + After Satellite Images] → [Building-Damage Model + Road/Flood Model] → [Damage and Road Predictions] → [Road-Network Graph] → [Accessibility Analysis] → [Damage-and-Access Map]

### Roadmap

**Phase 1: Dataset Preparation**
Load the xBD and SpaceNet 8 imagery and labels. Prepare aligned before-and-after image pairs and separate training sets for building damage and flooded-road detection.

**Phase 2: Building Detection and Damage Classification**
Train a Siamese U-Net or SegFormer to locate buildings, compare their before-and-after features, and classify their damage level.

**Phase 3: Road and Flood Segmentation**
Train a segmentation model using SpaceNet 8 to detect roads and determine which road segments are flooded or blocked.

**Phase 4: Accessibility Analysis**
Convert the predicted roads into a graph. Remove or penalize flooded road segments and estimate whether severely damaged areas remain reachable.

**Phase 5: Evaluation and Visualization**
Evaluate building segmentation IoU, damage-classification F1, flooded-road F1, and changes in route connectivity. Display the results on a map showing building damage and possible access routes.

The two deep-learning modules would first be evaluated separately because xBD and SpaceNet 8 cover different disaster imagery. Combining them into a single response map would be the final integration and demonstration.
