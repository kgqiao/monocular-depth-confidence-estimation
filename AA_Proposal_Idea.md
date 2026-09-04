# Alejandro Arteaga — Proposal Idea: Rare-Scenario Detection as a Data-Curation Engine for AV Perception

## Main idea / pitch

Use anomaly/out-of-distribution detection over vision-foundation-model embeddings
to automatically surface the rare, novel, or hard driving scenes hidden in a large
unlabeled driving dataset — and prove the scores matter by showing that labeling
*those* frames first trains a better perception model per labeling dollar than
random or uncertainty-based selection.

## Problem being tackled

AV fleets record far more data than anyone can label or even look at, and the
frames that actually improve a perception model are the rare ones: night + rain,
construction zones, unusual objects, odd geometry. Today's pipelines mostly
treat data selection passively (label a random slice) or with weak signals
(model uncertainty), and OOD detection research is mostly benchmarked on
class-level toy datasets (CIFAR-style "is this a ship?") rather than
scene-level "is this driving situation novel?" — so there is no established
recipe for turning anomaly scores into labeling decisions on real driving data.

## Datasets

### BDD100K (primary)

100k driving images with weather / scene / time-of-day attribute labels plus
2D detection labels. The attributes give us *free ground truth for rarity*
(e.g., hold out night+rain as the "rare" slice) and the detection labels let
us measure downstream curation value.

- **Access:** free with registration at <http://bdd-data.berkeley.edu/>
  (note: plain-http only — the site's HTTPS cert is currently broken).
  Toolkit and docs: <https://github.com/bdd100k/bdd100k>. Unofficial
  mirrors exist on Hugging Face if the portal is down. Downloads are
  plain zip archives — no SDK required.
- **Format:** non-relational, document-style. Images are JPEG frames
  (1280×720); labels are per-split JSON documents, one record per image,
  each holding scalar scene attributes (`weather`, `scene`, `timeofday` as
  categorical strings) and a list of objects (category string + float bbox
  coordinates). Loads with nothing fancier than `json.load` into
  pandas/pyarrow if we want it tabular.
- **Size / feasibility:** the 100k-images split is ~5–6 GB plus ~100 MB of
  labels — fits on a laptop or Colab disk. (The 40-second source *videos*
  are ~1.8 TB; we do not need them.) No official cloud-streaming endpoint,
  so the workflow is: download once, extract embeddings once, then work
  with the small embedding + label files. Very feasible.

### CODA (held-out evaluation)

CODA — Corner Case Dataset for Autonomous Driving — ~10k real road frames
containing corner-case objects/scenarios, built specifically to evaluate
rare-scenario detection. We use it purely as a frozen test set.

- **Access:** <https://coda-dataset.github.io/> (download links —
  Google Drive / cloud mirrors — plus format docs on the site).
- **Format:** non-relational: JPEG/PNG images sourced from KITTI, nuScenes,
  and ONCE, with COCO-style JSON annotations (per-image records, object
  category strings, float bboxes, corner-case flags).
- **Size / feasibility:** a few GB total — trivially local, evaluation is
  inference-only.

### nuScenes (stretch — the 3D/spatial angle)

Full AV sensor suite (6 cameras, LiDAR, 5 radars) that lets us add
LiDAR/radar-derived spatial signals to the anomaly score and connect to 3D
reconstruction work.

- **Access:** free for non-commercial use with registration at
  <https://www.nuscenes.org/nuscenes> (per-user signup and license
  acceptance). Loaded through the official devkit:
  `pip install nuscenes-devkit`
  (<https://github.com/nutonomy/nuscenes-devkit>).
- **Format:** unlike the two above, nuScenes is effectively **relational**:
  the metadata is a set of normalized JSON tables (`scene`, `sample`,
  `sample_data`, `sample_annotation`, `ego_pose`, `calibrated_sensor`, …)
  linked by foreign-key tokens, which the devkit joins for you. Sensor
  payloads are binary blobs referenced from those tables: JPEG camera
  frames, LiDAR sweeps as `.pcd.bin` float arrays (x, y, z, intensity,
  ring), radar point clouds as `.pcd`. The data is **log-derived**: 1,000
  scenes of 20 s each, cut from continuous driving logs, with annotated
  keyframes at 2 Hz — so it behaves like a time-ordered stream you index
  by scene → sample.
- **Size / feasibility:** the full *trainval* release is ~300+ GB —
  **not** something we should pull casually. The **v1.0-mini split
  (~4 GB, 10 scenes)** has the identical schema and is the right target
  for the course timeline: prototype every 3D signal on mini locally or
  in Colab, and only pull selected full-set scenes if the stretch goal
  becomes central. This is the main feasibility risk of the 3D angle,
  which is why it is scoped as a stretch, not the core.

## Main tech / deep learning models

- **Embedding backbones:** DINOv2 and CLIP (frozen) to represent each frame;
  no training needed for the scoring stage, so compute stays cheap.
- **Anomaly / OOD scoring layer:** classical and deep detectors — kNN-distance,
  Mahalanobis, energy score, MSP, and PyOD's detector zoo — benchmarked
  head-to-head on the same embeddings. (Anomaly detection is the instructor's
  research area — PyOD is his library — so this is a lane where we can get
  substantive feedback.)
- **Downstream curation evaluation:** a small detector (YOLOv8/RT-DETR variant)
  trained under a fixed labeling budget, comparing selection strategies:
  random vs. entropy/uncertainty vs. our anomaly-ranked selection.
  Headline figure: budget-vs-mAP curves; target claim of the form
  *"same mAP with 40% of the labels."*
- **Optional VLM layer:** an open VLM (e.g., Qwen-VL / LLaVA) that *names* why
  a flagged scene is novel ("construction zone at night in rain"), turning raw
  scores into an interpretable mining report.

## Practical application and the gap it closes

**Application:** an open-source "data engine" component for AV/robotics teams —
point it at an unlabeled driving log, get back (a) a ranked queue of the frames
most worth labeling and (b) a natural-language digest of the novel scenarios
found. This is the core loop of scenario-mining / data-curation teams across
the AV industry, rebuilt on public data and open models.

**Gap / novel part:** OOD detection papers report AUROC on class-level toy
benchmarks and stop there; active-learning papers use uncertainty but rarely
scene-level novelty. The gap this project closes is the *bridge*: evaluating
scene-level anomaly scores by their downstream labeling-budget value on real
driving data — treating "is this scene weird?" not as an end in itself but as
a data-valuation signal, with an honest ablation of which scoring family
actually buys mAP.

## How this fits the team's interests

- **Anomaly detection** is the core method; **vision** is the domain.
- **3D perception / spatial world models:** the stretch goal scores novelty in
  spatial terms, not just image embeddings — LiDAR point-distribution
  statistics, depth-consistency signals, or BEV-space features on nuScenes —
  asking "is the *geometry* of this scene unusual?" rather than only its pixels.
- The frame-scoring output is deliberately generic: any pipeline that needs to
  *select* frames (for labeling, training, or reconstruction) can consume the
  ranked queue, so it composes naturally with the other ideas in this repo.

## Feasibility notes

- Scoring stage is inference-only over frozen embeddings — runs on Colab or a
  single consumer GPU; the detector-training stage uses small models on
  subsets, sized for a course timeline.
- The benchmark structure guarantees a results table even if no method "wins";
  the curation experiment and the 3D-signal ablation are where the novelty
  lives.
- Modular enough for 5 people: embeddings + scoring / curation loop + detector
  training / CODA + rare-slice evaluation / VLM reporting / 3D-spatial signals.
