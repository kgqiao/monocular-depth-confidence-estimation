
# Katherine Qiao Project Proposal Ideas


# (1) Project: Active Cross-Modality Adverse-Weather Selection:

This repository contains a hardware-agnostic, multi-sensor active perception pipeline designed for autonomous vehicle (AV) data engines. It processes raw multi-modal sensor inputs from large-scale driving datasets (e.g., **nuScenes**, **Waymo Open Dataset**) to isolate environmental degradation, reconstruct photorealistic 3D digital twins, and execute adversarial closed-loop simulation stress testing.

---
## Project Summary & Core Revision
Traditional autonomous vehicle pipelines treat data collection passively—streaming and storing every frame regardless of visual quality. Under adverse conditions like severe rain, fog, or snow, standard **3D Gaussian Splatting (3DGS)** pipelines fail because they try to model transient weather particles as permanent 3D structures, leaving behind heavy "ghosting" artifacts and blurry "mid-air floaters." Conversely, modern generative world models (such as *RealWeather*) attempt to solve this by synthesizing or "hallucinating" clean pixels over noisy ones, introducing high safety risks for autonomous systems.

## 🎯 Application, Gap, & Users

*   **Application / Use Case:** Transforming noisy, real-world adverse weather logs into clean, reusable 3D simulation assets for autonomous software verification.
*   **Gap Filled:** Bridges the gap between artifact-ridden passive 3D reconstruction pipelines and hallucination-prone 2D generative video models. This framework provides **the first physics-grounded, de-weathered 3D asset engine capable of true closed-loop sensor simulation**.
*   **Target Users:** AV Simulation Engineers, Perception Validation Teams, and Robotic State-Estimation Researchers.

---
**Workflow Process:**
[Raw AV Data Log] ──> [Depth Anything v2] ──> [Compute Temporal Variance (σ²)]
                                                         │
                                               (If Variance Spikes)
                                                         │
[Optimized 3DGS Map] <── [Pass Selected ~15%] <── [Cross-Reference Radar Depth]


## Comprehensive Implementation Roadmap

### Phase 1: Data Ingestion & Cross-Sensor Projections
*Initialize multi-sensor pipelines and establish metric spatial alignment.*
*   **Primary Stack:** Python, `nuScenes-devkit` / Waymo API, NumPy, Open3D.
*   **Implementation:** Ingest raw multi-view camera streams, precise vehicle ego-trajectories ($SE(3)$ poses), and sparse Radar point clouds from adverse weather sequences. Project the all-weather Radar frames into the 2D image coordinate matrices using native extrinsic and intrinsic calibration tensors.

### Phase 2: Vectorized Spatial Architecture
*Design a high-performance Python matrix core to analyze dense frames simultaneously.*
*   **Primary Stack:** Python, NumPy, PyTorch.
*   **Implementation:** Build a pure Python spatial processor utilizing multi-dimensional broadcasting arrays. Align pixel-level sensor structures simultaneously (`[B, C, H, W]`). This architectural foundation completely replaces slow Python loops, allowing full-frame radar returns to map against camera pixels instantly.

### Phase 3: Temporal Uncertainty & Vision Disagreement
*Quantify frame-to-frame geometric instability induced by weather noise.*
*   **Primary Stack:** PyTorch, Depth Anything V2 (Metric variant).
*   **Implementation:** Process sequential RGB frames through *Depth Anything V2*. Because raindrops or heavy fog distort lens elements dynamically, depth predictions for static structures will fluctuate violently over time. Implement a temporal sliding-window filter to calculate the statistical pixel-wise depth variance ($\sigma^2$) across a 5-frame interval. High variance flags transient weather distortion.

### Phase 4: The Active Cross-Modality Selector *(Core Contribution)*
*Formulate a cross-modality data pruning policy that cross-examines vision with physics.*
*   **Primary Stack:** Python, NumPy, SciPy (Information Theory module).
*   **Implementation:** Build the active selector logic. For regions where visual depth variance spikes, look up the projected Radar depths in that exact spatial cluster. Since Radar waves pierce rain and snow without degradation, treat the Radar as an absolute structural sanity check. If the camera's depth prediction fundamentally contradicts the Radar data, drop the frame. If they match despite the rain, keep the frame. Prune the raw video data down to an optimized sub-set of the top ~10–15% highest-fidelity frames.

### Phase 5: 3D Urban Gaussian Splatting Reconstruction
*Compile your optimized, de-weathered dataset into an explicit 3D digital twin.*
*   **Primary Stack:** Nerfstudio, Street Gaussians, or DrivingGaussian architectures.
*   **Implementation:** Export your custom, pruned image manifest alongside the corresponding JSON camera trajectory formats. Train an explicit 3D/4D Gaussian Splatting model of the driving corridor. Because your algorithm stripped away the weather-polluted frames, the engine yields a photorealistic, architecturally sound static map without blurry "mid-air floaters."

### Phase 6: Real2Sim Closed-Loop Stress Testing *(Core Contribution)*
*Repurpose the clean digital twin as a simulation substrate and execute edge-case stress tests.*
*   **Primary Stack:** OpenCV Shaders, PyTorch, YOLOv10 or modern 3D Object Detectors (BEVFormer).
*   **Implementation:** Take your pristine 3D Gaussian Splat asset and turn it into a **Real2Sim simulator**:
    1.  **Trajectory Alteration:** Program a virtual vehicle to deviate from the original car's path, forcing the 3DGS engine to render entirely novel camera view vectors.
    2.  **Adversarial Hazard Injection:** Attach custom post-processing shaders to the simulation camera to inject controlled, synthetic weather back into the clean scene (e.g., dial up a mathematical fog density model using Koschmieder's Law or add virtual heavy snow).
    3.  **Photometric Relighting:** Shift the environmental light source to create an artificial, low-angle sunset, casting intense solar glare straight into the virtual car's lens.








---
# (2) Project: Causal-Policy Urban Forecasting VLM (C-PUF)

### Project Summary
*   **Concept**: A multimodal deep learning pipeline that shifts urban change analysis from passive pixel extrapolation to **generative causal forecasting**.
*   **What It Does**: Ingests irregular multi-year satellite images alongside text-based municipal zoning policies to generate predictive future 3D urban assets and text justification reports.
*   **Foundation Layer**: Built on top of **TEOChat** (Stanford Ermon Group), a LLaVA-style foundation model pre-trained on multi-temporal Earth Observation sequence data.

---

### Core Framework Blueprint

#### Gap Filled & Contribution
*   **Technical Gap**: Standard models assume uniform frame rates (failing on irregular gaps due to cloud cover) and passively extrapolate pixels based on visual momentum alone, ignoring human planning regulations.
*   **Core Contribution**: Embed a **Continuous Neural Time-Deformable Positional Encoding Layer** and a **Cross-Modal Policy Projection Network**.
*   **How It Works**: Converts irregular calendar day counts ($\Delta t$) into continuous sinusoidal vectors. This forces the transformer to align and evaluate the **exact same pixel coordinates** across different years ($T_1, T_2, T_3$), anchoring textual zoning regulations directly to localized spatial coordinate patches to compute true urban change vectors.

#### 1) Datasets Used
*   **Sentinel-2 & Landsat-8**: Multi-temporal Earth Observation imagery arrays programmatically pulled and cloud-masked via the **Google Earth Engine (GEE) Python API**.
*   **TEOChatlas**: Used for initial model pre-training weights, containing 500k+ instruction-following geospatial examples.
*   **OpenStreetMap (OSM)**: Regional transportation networks and boundary geometries extracted programmatically using `osmnx`.

#### 2) Main Tech Involved
*   **Deep Learning Architecture**: Spatial-Temporal Transformers, Multi-Frame Cross-Attention, Parameter-Efficient Fine-Tuning (`QLoRA` via `PEFT` / `bitsandbytes`).
*   **Geospatial Data Engineering**: Cloud-Native Geospatial Ingestion (`ee` API, `rioxarray`, `odc-stac`), Vector-to-Raster Processing (`shapely`, `geopandas`).
*   **Neural Rendering Primitives**: 3D Scene Reconstruction and Spatial Graphics Engine bindings (`Nerfstudio` / `TCLC-GS Engine`, `PyTorch3D`).

#### 3) Application / Use Case
*   **Target Application**: Urban Planning Digital Twin Platforms.
*   **Narrow Use Case**: Allows engineers to stress-test hypothetical zoning shifts (e.g., *"allow high-density zoning within 500m of a newly proposed transit line"*) to simulate precisely where real estate and structural development will realign over a 5-year horizon.
*   **Target Users**: Smart City Simulation Engineers, Municipal Infrastructure Planners, and Remote Sensing Data Scientists.
