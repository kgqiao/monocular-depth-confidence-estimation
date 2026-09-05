# Active Cross-Modality Selection for Adverse Weather 3D Reconstructions & Real2Sim Stress Testing

This repository contains a hardware-agnostic, multi-sensor active perception pipeline designed for autonomous vehicle (AV) data engines. It processes raw multi-modal sensor inputs from large-scale driving datasets (e.g., **nuScenes**, **Waymo Open Dataset**) to isolate environmental degradation, reconstruct photorealistic 3D digital twins, and execute adversarial closed-loop simulation stress testing.

## Project Summary & Core Revision (Our Contribution)

Traditional autonomous vehicle pipelines treat data collection passively—streaming and storing every frame regardless of visual quality. Under adverse conditions like severe rain, fog, or snow, standard **3D Gaussian Splatting (3DGS)** pipelines fail because they try to model transient weather particles as permanent 3D structures, leaving behind heavy "ghosting" artifacts and blurry "mid-air floaters." Conversely, modern generative world models (such as *RealWeather*) attempt to solve this by synthesizing or "hallucinating" clean pixels over noisy ones, introducing high safety risks for autonomous systems.

### 🌟 Where We Are Adding Our Contributions (The Core Revision)
Instead of trying to digitally fix corrupted pixels after the fact, **our core contribution introduces an active selection filter based on cross-modality physics grounding.** 

1. **The Vectorized Disagreement Engine (Phase 2 & 3):** We calculate frame-to-frame geometric instability using a sliding window over a monocular depth model (**Depth Anything v2**). 
   
2. **The Physics Sanity Check (Phase 4):** Where visual depth fluctuates wildly due to weather distortion (e.g., rain blur on the lens), our pipeline looks at co-located, projected **Radar returns**. Because Radar waves pierce rain, fog, and snow without metric degradation, our algorithm cross-examines the camera's visual uncertainty with ground-truth radar physics. If the sensors radically disagree, the frame is flagged as weather-corrupted and dropped. 
   
3. **The Real2Sim Simulation Testbed (Phase 6):** Once our active selector passes a pristine, un-hallucinated dataset to a 3D street-mapping engine (**Street Gaussians**), our final contribution repurposes the clean 3D asset as a customizable substrate. We dynamically alter the driving trajectory, inject shader-based synthetic hazards (e.g., fog via Koschmieder's Law, sunset glare), and programmatically chart the exact operational thresholds where downstream object detection models break down.


### 🌟 What We Are Improving (The Core Innovation)
Instead of trying to digitally fix or hallucinate corrupted pixels after the fact, **our framework shifts the paradigm from generative 2D synthesis to physics-grounded 3D data filtering.** We introduce a custom Active Perception filter that acts as an intelligent algorithmic critic to optimize data fidelity before reconstruction:

*   **Replacing Passive Accumulation with Active Pruning:** Instead of dumping 100% of a noisy video into a 3DGS pipeline, our engine automatically downsamples and filters the data feed down to an optimized sub-set of the **top ~15% highest-fidelity frames**.
*   **Replacing Generative Hallucinations with Multi-Modal Physics Grounding:** We calculate frame-to-frame geometric instability across a monocular depth model (**Depth Anything v2**). Where visual depth calculations fluctuate violently due to water droplets distorting the lens, our core algorithm cross-examines the visual uncertainty with co-located, projected **Radar returns**. Because Radar waves pierce rain, fog, and snow without metric degradation (unlike LiDAR, which suffers from severe atmospheric backscatter), we use actual hard radar physics to audit the AI depth map. If the sensors radically disagree, the frame is flagged as weather-corrupted and dropped.
*   **Transforming Static Output into a Closed-Loop Stress Testbed:** Standard 3DGS pipelines produce static visual captures. Our framework takes the pristine, "de-weathered" 3D street asset and repurposes it as an interactive *Real2Sim* simulator substrate. We dynamically alter the virtual driving trajectory to view the scene from completely novel vantage points, apply shader-based atmospheric effects (e.g., explicit fog using Koschmieder's Law), and programmatically map out the exact performance drop-off thresholds of safety-critical object detection models.




---

## 🎯 Application, Gap, & Users

*   **Application / Use Case:** Transforming noisy, real-world adverse weather logs into clean, reusable 3D simulation assets for autonomous software verification.
*   **Gap Filled:** Bridges the gap between artifact-ridden passive 3D reconstruction pipelines and hallucination-prone 2D generative video models. This framework provides **the first physics-grounded, de-weathered 3D asset engine capable of true closed-loop sensor simulation**.
*   **Target Users:** AV Simulation Engineers, Perception Validation Teams, and Robotic State-Estimation Researchers.

---

## 🚀 Language Strategy
This framework is engineered entirely in **Python (NumPy, PyTorch, SciPy)**. All spatial computations, raycasting approximations, and data filtering blocks are written using **highly optimized tensor broadcasting and vectorized matrix operations** to bypass native Python loop bottlenecks.

---

## 📂 Repository Directory Structure

```text
weather-active-perception/
│
├── config/
│   └── nuscenes_config.yaml     # Camera intrinsics, paths, sensor parameters
│
├── data_loaders/
│   ├── __init__.py
│   ├── nuscenes_unpacker.py     # Python API to unpack images, poses, Radar
│   └── coordinate_transforms.py # Handles World -> SE(3) Camera -> 2D Image plane
│
├── core_planner/
│   ├── __init__.py
│   ├── vector_raycast.py        # Vectorized PyTorch/NumPy spatial mapping core
│   ├── uncertainty_metrics.py   # Information Theory math (Entropy/Variance)
│   └── active_selector.py       # Cross-modality evaluation policy (Our Main Contribution)
│
├── perception_models/
│   └── depth_anything_v2.py     # PyTorch wrapper for sliding-window depth inference
│
├── simulator/
│   └── shaders/
│       ├── fog.glsl             # Koschmieder's Law shader for OpenCV/WebGL (Our Main Contribution)
│       └── rain_particles.glsl  # Synthetic precipitation shader (Our Main Contribution)
│
├── main.py                      # Global pipeline execution script
├── requirements.txt
└── README.md
```

---

## 🛠️ Comprehensive Implementation Roadmap

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

### Phase 4: The Active Cross-Modality Selector 
*Formulate a cross-modality data pruning policy that cross-examines vision with physics.*
*   **Primary Stack:** Python, NumPy, SciPy (Information Theory module).
*   **Implementation:** Build the active selector logic. For regions where visual depth variance spikes, look up the projected Radar depths in that exact spatial cluster. Since Radar waves pierce rain and snow without degradation, treat the Radar as an absolute structural sanity check. If the camera's depth prediction fundamentally contradicts the Radar data, drop the frame. If they match despite the rain, keep the frame. Prune the raw video data down to an optimized sub-set of the top ~10–15% highest-fidelity frames.

### Phase 5: 3D Urban Gaussian Splatting Reconstruction
*Compile your optimized, de-weathered dataset into an explicit 3D digital twin.*
*   **Primary Stack:** Nerfstudio, Street Gaussians, or DrivingGaussian architectures.
*   **Implementation:** Export your custom, pruned image manifest alongside the corresponding JSON camera trajectory formats. Train an explicit 3D/4D Gaussian Splatting model of the driving corridor. Because your algorithm stripped away the weather-polluted frames, the engine yields a photorealistic, architecturally sound static map without blurry "mid-air floaters."

### Phase 6: Real2Sim Closed-Loop Stress Testing 
*Repurpose the clean digital twin as a simulation substrate and execute edge-case stress tests.*
*   **Primary Stack:** OpenCV Shaders, PyTorch, YOLOv10 or modern 3D Object Detectors (BEVFormer).
*   **Implementation:** Take your pristine 3D Gaussian Splat asset and turn it into a **Real2Sim simulator**:
    1.  **Trajectory Alteration:** Program a virtual vehicle to deviate from the original car's path, forcing the 3DGS engine to render entirely novel camera view vectors.
    2.  **Adversarial Hazard Injection:** Attach custom post-processing shaders to the simulation camera to inject controlled, synthetic weather back into the clean scene (e.g., dial up a mathematical fog density model using Koschmieder's Law or add virtual heavy snow).
    3.  **Photometric Relighting:** Shift the environmental light source to create an artificial, low-angle sunset, casting intense solar glare straight into the virtual car's lens.
    4.  **Evaluation:** Stream these newly corrupted simulation frames into an autonomous object detection model. Quantify the exact threshold (e.g., specific visibility boundary or glare index) where the safety-critical network fails to track pedestrians or crosswalks.

---

## 📦 Tangible Project Outputs

Upon completion of the execution pipeline, this framework yields four high-value deliverables:
1. **The Interactive 3D Digital Twin**: A fully trained, photorealistic 3D/4D Gaussian Splat asset (.ply file) representing a clean, structurally accurate recreation of a complex intersection, ready to be flown through via virtual camera in 360 degrees.

2. **The Active Selector Engine**: A robust, modular Python/PyTorch repository that ingests multi-sensor AV formats (Waymo/nuScenes), calculates spatiotemporal vision uncertainty, and executes cross-modality pruning.

3. **The Frame-Selection JSON Manifest**: An optimized file mapping the specific high-fidelity frames selected by your algorithm, proving a ~85% data reduction while preserving or improving geometric accuracy.

4. **The Closed-Loop Simulation Testbed & Metrics**: A suite of Matplotlib/OpenCV benchmarks mapping how perception models degrade against your injected synthetic hazards (fog, snow, sunset glare). This includes a split-screen presentation video demonstrating the original blurry rain sequence on the left, and a smooth, re-weathered adversarial simulation on the right.
