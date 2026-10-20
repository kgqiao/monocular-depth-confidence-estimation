
---

## Figure 1. The pipeline based on Google Doc Plan (9-8-2026)


```mermaid
flowchart LR
    subgraph inputs ["Inputs per frame"]
        cam["6 camera images"]
        rad["Radar returns"]
        lid["Lidar sweep"]
    end

    da["Depth Anything V2<br/>metric variant"]
    var["Depth variance<br/>5-frame sliding window"]
    proj["Project radar<br/>into the image"]
    sel["Frame selector<br/>keep top 10-15 percent"]
    bev["Depth to BEV projection"]
    occ["Occupancy grid<br/>free / occupied / unknown"]
    carve["Lidar ray-carving<br/>occluded or observed per voxel"]
    visdb[("nuScenes visibility<br/>attribute, 4 buckets")]
    bucket["Occlusion-stratified scoring"]
    table["mIoU, stock vs ours<br/>by occlusion bucket"]

    cam --> da --> var --> sel
    rad --> proj --> sel
    sel --> bev --> occ
    var -. "confidence signal" .-> occ
    proj -. "confidence signal" .-> occ
    occ --> bucket --> table
    lid --> carve --> bucket
    visdb --> bucket

    class da blue
    class cam,rad,lid,visdb data
    class var,proj,sel,carve,bucket ours
    class bev,occ open
    class table metric
    class inputs group

    classDef blue fill:#e0ecfb,stroke:#2563eb,stroke-width:2px,color:#111
    classDef group fill:#f7f8fa,stroke:#cbd5e1,stroke-width:1px,color:#111
    classDef ours fill:#ede9fe,stroke:#7c3aed,stroke-width:3px,color:#111
    classDef data fill:#eef0f2,stroke:#6b7280,stroke-width:2px,color:#111
    classDef metric fill:#e3f5e6,stroke:#16a34a,stroke-width:2px,color:#111
    classDef open fill:#fdeaea,stroke:#dc2626,stroke-width:2px,stroke-dasharray:4 3,color:#111
```

Only one borrowed model appears: Depth Anything V2. Everything between the depth
map and the score is ours. Lidar enters the pipeline only on the scoring side,
for ray-carving, not as a model input.

---

## Figure 2. What the before-and-after compares

The doc's final item is "standard Depth Anything vs our improved Depth Anything."
Both arms have to reach a grid before there is a number to compare.

```mermaid
flowchart LR
    cam["6 camera images"]
    rad["Radar returns"]

    subgraph baseline ["Baseline arm"]
        da1["Depth Anything V2<br/>all frames, no selection"]
        bev1["Depth to BEV"]
        occ1["Occupancy grid"]
    end

    subgraph improved ["Our arm"]
        da2["Depth Anything V2"]
        var2["Depth variance"]
        sel2["Cross-modality<br/>frame selector"]
        bev2["Depth to BEV<br/>confidence weighted"]
        occ2["Uncertainty-aware grid<br/>free / occupied / unknown"]
    end

    gt[("Occupancy ground truth")]
    score["mIoU, both arms"]

    cam --> da1 --> bev1 --> occ1 --> score
    cam --> da2 --> var2 --> sel2 --> bev2 --> occ2 --> score
    rad --> sel2
    var2 -.-> bev2
    gt --> score

    class da1,da2 blue
    class cam,rad,gt data
    class var2,sel2 ours
    class bev1,bev2,occ1,occ2 open
    class score metric
    class baseline,improved group

    classDef blue fill:#e0ecfb,stroke:#2563eb,stroke-width:2px,color:#111
    classDef group fill:#f7f8fa,stroke:#cbd5e1,stroke-width:1px,color:#111
    classDef ours fill:#ede9fe,stroke:#7c3aed,stroke-width:3px,color:#111
    classDef data fill:#eef0f2,stroke:#6b7280,stroke-width:2px,color:#111
    classDef metric fill:#e3f5e6,stroke:#16a34a,stroke-width:2px,color:#111
    classDef open fill:#fdeaea,stroke:#dc2626,stroke-width:2px,stroke-dasharray:4 3,color:#111
```

The baseline arm needs the same depth-to-BEV and grid code as our arm, or the
comparison measures the grid builder instead of the depth model. That code is
one component used twice, not two components.

---

## Figure 3. Where the occlusion contribution attaches

Contribution 4 is not a block in the model. It sits after prediction, on the
scoring side, and it changes how one number becomes several.

```mermaid
flowchart TD
    pred["Predicted grid, either arm"]
    lid["Lidar sweep"]
    carve["Ray-carving:<br/>clear line of sight to sensor?"]
    visdb[("nuScenes visibility attribute<br/>0-40, 40-60, 60-80, 80-100 percent")]
    label["Per-voxel occlusion label"]
    split["Split voxels into buckets"]
    r1["mIoU, mostly visible"]
    r2["mIoU, partly occluded"]
    r3["mIoU, mostly occluded"]

    lid --> carve --> label
    visdb --> label
    label --> split
    pred --> split
    split --> r1
    split --> r2
    split --> r3

    class lid,visdb data
    class carve,label,split ours
    class pred open
    class r1,r2,r3 metric

    classDef ours fill:#ede9fe,stroke:#7c3aed,stroke-width:3px,color:#111
    classDef data fill:#eef0f2,stroke:#6b7280,stroke-width:2px,color:#111
    classDef metric fill:#e3f5e6,stroke:#16a34a,stroke-width:2px,color:#111
    classDef open fill:#fdeaea,stroke:#dc2626,stroke-width:2px,stroke-dasharray:4 3,color:#111
```

The doc says occlusion buckets replace day/night as the primary split, because
buckets pool many objects while weather pools a handful of scenes. Both splits
come out of the same scoring script, so running both costs almost nothing.

Two different things carry the name "occlusion" here. Ray-carving gives a label
per voxel. The nuScenes visibility attribute gives a label per annotated object.
Joining them means deciding what an object-level percentage says about the voxels
inside that object's box.

---

## Figure 4. Where the ground truth comes from

The pipeline does not score against Occ3D. It builds its own labels, because the
grid it predicts has a state Occ3D does not contain.

```mermaid
flowchart LR
    lid["Lidar sweep"]
    ret["Lidar returns per cell"]
    carve["Ray-carving:<br/>clear line of sight to sensor?"]
    visdb[("nuScenes visibility attribute")]
    gt["Our GT grid<br/>free / occupied / occluded"]
    pred["Our predicted grid<br/>free / occupied / unreliable"]
    score["mIoU over 3 states"]

    lid --> ret --> gt
    lid --> carve --> gt
    visdb --> carve
    gt --> score
    pred --> score

    class lid,visdb data
    class carve,gt ours
    class ret,pred open
    class score metric

    classDef ours fill:#ede9fe,stroke:#7c3aed,stroke-width:3px,color:#111
    classDef data fill:#eef0f2,stroke:#6b7280,stroke-width:2px,color:#111
    classDef metric fill:#e3f5e6,stroke:#16a34a,stroke-width:2px,color:#111
    classDef open fill:#fdeaea,stroke:#dc2626,stroke-width:2px,stroke-dasharray:4 3,color:#111
```

The doc specifies how ray-carving marks a cell occluded. It does not say what
marks a cell occupied or free in the ground truth. Lidar returns are the obvious
source, so that node is drawn but is not in the doc.

---

## Two decisions this pipeline makes

Neither is stated as a decision in the meeting doc. Both follow from it.

**We construct the grid, we do not predict a labeled one.** Depth goes into
bird's-eye-view cells and each cell gets a state. There is no image encoder, no
lifting of features into voxels, no 3D head, and no training. The grid is the
output of a projection and a threshold rule.

**We bring our own ground truth.** Occ3D labels every nuScenes keyframe with 17
semantic classes and ships visibility masks. Our grid has three states and no
classes, so Occ3D has nothing to compare against a cell we call unreliable. The
ray-carving in Contribution 4 exists to produce labels that match our output.

Both follow from the same choice, so they stand or fall together.

The cost is that no published number is comparable to ours. mIoU over free,
occupied and unreliable is not the mIoU in occupancy papers, which averages over
17 classes. The before-and-after against stock Depth Anything still works, since
both arms use our GT. Nothing outside the project does.

The alternative, a fusion network trained on Occ3D, is in
`route-b-occ3d.md`. It is not what the meeting doc describes.

**BEV Grid States** 

Free and occupied are observations. 

| State | Set by | Meaning |
|---|---|---|
| Free | Depth projection | Nothing here |
| Occupied | Depth projection | Something here |
| Unreliable | Depth variance, radar disagreement | We saw it, we do not trust it |
| Occluded | Lidar ray-carving | We could not see it |

Unreliable is a prediction-side state from Contribution 3. Occluded is a
scoring-side label from Contribution 4. Collapsing them into one "unknown"
would lose the distinction the two contributions are built on.

---

## Component inventory

| Component | Borrowed or ours | Source or note |
|---|---|---|
| Depth Anything V2, metric | Borrowed | Running in the sandbox already |
| Depth variance over a sliding window | Ours | Needs a frame rate; see below |
| Radar projection into the image | Ours | In the sandbox viewer already |
| Cross-modality frame selector | Ours | The threshold rule is hand-tuned |
| Depth to BEV projection | Ours | Not specified in the doc |
| Free / occupied / unknown classifier | Ours | Not specified in the doc |
| Confidence to "unknown" rule | Ours | Not specified in the doc |
| Lidar ray-carving | Ours | Not specified in the doc |
| nuScenes visibility attribute | Borrowed | Ships with the dataset |
| Occlusion-bucketed scoring script | Ours | One script, several rows |

---
