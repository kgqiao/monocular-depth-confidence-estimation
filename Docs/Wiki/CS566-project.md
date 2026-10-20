---
id: are-ncvb2c
type: topic
kind: overview
description: "The CSCI 566 group project, Fall 2026: a learned per-pixel confidence head that tells when monocular depth from Depth Anything V2 is wrong in adverse driving conditions."
tags: [machine-learning/cs566-deeplearning]
aliases: [CS566 project, Knowing When Not to Trust Depth]
created: 2026-09-28T00:00:00
updated: 2026-10-19T14:42:31
refreshed: 2026-10-19T14:42:31
status: stable
sources: []
defines: machine-learning/cs566-deeplearning/cs566-project
---

> [!overview] The page of #machine-learning/cs566-deeplearning/cs566-project · under [[CS566-DeepLearning]]
> 0 sources · refreshed 2026-10-19 · #machine-learning/cs566-deeplearning

## Summary

The CSCI 566 group project, Fall 2026: a learned per-pixel confidence head that tells when monocular depth from Depth Anything V2 is wrong in adverse driving conditions.

## Context

Course rules for the project (team size, deliverables, course deadlines) are on [[CS566-DeepLearning]]. The subject is perception and sensor fusion for driving, so the project also belongs to [[Autonomous Driving]], which links here. The repository [[kqiao-cs566-project-repo]] holds the team's deliverables, like pre-preposal reports, tests, and the confidence head API.

## Team

Nathan Aday, Alejandro Arteaga, Nathan Bao, Katherine Qiao (KQ), and Varad Jaideep Tawde ([[CS566 Project Pre-proposal]], p. 1).

On October 9, 2026, the team assigned its roles with a fair matching script ([[Notes - CS566 Project Huddle (Recurring)]], Assignments; [[CS566 Project Roles]]):

- **Datasets and Training:** Nathan Aday and Nathan Bao.
- **Evaluation:** Varad Jaideep Tawde and Katherine Qiao.
- **Cross Team Coordinator:** Alejandro Arteaga.

Each team meets once a week with the coordinator, and everybody meets at the weekly huddle ([[Notes - CS566 Project Huddle (Recurring)]], Team Responsibilities). See [[CS566 Project Team Roles]] for the duties of each role and the matching rule.

## The project

The pre-proposal is titled "Knowing When Not to Trust Depth: A Learned Per-Pixel Confidence Head for Monocular Depth in Adverse Driving Conditions" ([[CS566 Project Pre-proposal]], p. 1). It proposes a small convolutional head on top of a frozen [[Depth Anything V2]]. The head predicts, per pixel, whether the depth error exceeds a threshold. It reads the depth model's features, temporal depth variance, and radar signals, and lidar projected into the image gives the training labels. See [[Per-Pixel Depth Confidence Head]].

## My Journal

The following are my own notes as we progress through the project. I will continue to capture new ideas, findings, and document understanding here. 

We have proposed a self driving application where we want to train a sensor fusion head to improve the accuracy of depth calculations. The main contribution is to assign confidence score to depth anything outputs. At the moment, depth anything confidently produces a pixel map of depth estimates, with no way to know which parts of trust worthy or not trust worthy. 

In theory, you can do this training with any monocular image dataset that also has a depth ground truth. We are kind of bringing the scope closer with the insight that NuScenes and other self-driving datasets provide what we believe are the right foundations for this training: we have a set of RGB images and corresponding RADAR/LiDAR values that act as ground truth.

A [[Head (machine learning)|head]] is a small neural network attached to a larger pre-trained model. It reads the model's outputs. Useful as you can keep the larger pre-trained model frozen while still adding to its capabilities or behavior.

Radar vs. Lidar

We have only 10-30 radar points per image, and all the radar points are at the mounting height of the radar: there is no height dimension.

For the sake of this project, we are assuming the data collection car has lidar, and the road car does not have lidar. This means the road car will rely on our depth anything head for the true depth eval.

Possible extension: assign the confidence score to entire objects, not just the pixel. Recognize and segment them, and take the depth measurement to the nearest point of the object, so you now have a confidence prediction per object. So once you have the confidence head, there are other ways you can play with the backbone to keep improving upon it.

## Timeline

- **Pre-proposal.** The group submitted [[CS566 Project Pre-proposal]], and the TA answered with [[CS566 Project Pre-preposal Feedback]]. The course sources disagree on its deadline; see [[CS566-DeepLearning]].
- **Roles.** The team defined its focus roles at the Oct 5 huddle and assigned them on Oct 9 ([[Notes - CS566 Project Huddle (Recurring)]]). See [[CS566 Project Team Roles]].
- **Dataset plan and first training run.** The Datasets and Training role wrote both on Oct 12, after the TA feedback. See [[CS566 Dataset Strategy]] and [[CS566 Confidence Head Training Run 1]].
- **Plan in the pre-proposal** ([[CS566 Project Pre-proposal]], p. 3, §5):
  - Weeks 1–3: form the team, set goals, and survey work on monocular depth confidence, radar-camera fusion, and adverse-condition driving datasets.
  - Weeks 4–6: preprocess [[nuScenes]], cache Depth Anything V2 decoder features and depth, and project lidar sweeps into per-pixel error labels.
  - Weeks 7–12: build the cheap baselines, train the head, and evaluate with sparsification curves, AUSE, and AURG across day, night, and rain.
  - Dissemination: compile results, stretch-goal experiments (frame selector, occupancy weighting), and analysis into a final report and repository.

## Reporting rules

The project proposal raises these points. These points are useful to stay grounded on the original idea, but as work on the project progresses, these are subject to change. Treat these as guiding points and NOT hard requirements. If we ever find a better methodology or improvement, even if it is not stated in the pre-proposal, we can still adopt it for the project.

- Report every result separately for day, night, and rain on the nuScenes validation split ([[CS566 Project Pre-proposal]], p. 2, §3).
- If the head does not beat raw temporal variance, report the negative result in the same table ([[CS566 Project Pre-proposal]], p. 2, §3.1).
- Let the ablations without radar channels and on decoder features alone decide what stays in the model ([[CS566 Project Pre-proposal]], p. 2, §3.1).
- Put the scene count of each split next to each number ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § How it is scored).
- Report lidar label density per weather split, so that missing labels in rain are visible ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Things to keep in mind).
- Break out confidence maps by textureless surfaces and object edges, not only by weather tag ([[CS566 Brainstorm - Project Scope Visual (Form B)]], § Things to keep in mind).

## Pages

- [[Per-Pixel Depth Confidence Head]]: the method.
- [[CS566 Project Team Roles]]: the roles, who holds each, and how the team assigned them.
- [[CS566 Dataset Strategy]]: the datasets in order of use, and the splits.
- [[CS566 Confidence Head Training Run 1]]: the plan of the first training run.
- [[Lidar and Radar Projection into Camera Images]]: the label and radar masks, built and measured.
- [[Sparsification Curve]] (in [[Machine Learning]]): the evaluation method.
- Background: [[Depth Anything V2]], [[nuScenes]], [[HyDRa]], [[LiCROcc]].

## Map

```base
filters:
  and:
    - file.inFolder("tool/source-core/documents")
    - 'file.hasTag("machine-learning/cs566-deeplearning/cs566-project", "machine-learning/cs566-deeplearning/cs566-project/autonomous-vehicle-sandbox", "machine-learning/cs566-deeplearning/cs566-project/kqiao-cs566-project-repo")'
views:
  - type: table
    name: Map
    groupBy:
      property: type
      direction: ASC
    order:
      - file.name
      - description
      - status
      - refreshed
```

## Sources

- [[CS566 Project Pre-proposal]]
- [[CS566 Brainstorm - Project Scope Visual (Form B)]]
- [[Notes - CS566 Project Huddle (Recurring)]]
- [[CS566 Project Roles]]
- [[CS566 Project Pre-preposal Feedback]]
