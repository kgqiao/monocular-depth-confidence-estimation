 Idea 1 - Puddle / Occlusion-Aware Drivable-Space Estimator

Project Summary
Standard drivable-space estimators commit to a single confident prediction per pixel, even when the road surface is ambiguous. Puddles and wet asphalt reflect sky/trees, causing camera-based depth and LiDAR/stereo returns to disagree — the model either misreads a puddle as an obstacle or a hole as flat ground. Instead of forcing a confident (and sometimes wrong) answer, this project builds an occupancy model that explicitly flags "I'm not sure" regions where sensor modalities conflict, prioritizing safety over false confidence.


Use Case: Detect reflective/ambiguous road surfaces where vision and depth sensors disagree, surfacing a low-confidence region instead of a committed (potentially wrong) drivable-space prediction.
Gap Filled: Most occupancy pipelines optimize for accuracy on "normal" road conditions and silently fail on rare reflective/wet surfaces. This fills the gap between confident-but-wrong occupancy maps and a conservative, uncertainty-aware fallback signal.
Target Users: AV planning teams needing a conservative fallback signal, not full puddle-depth estimation.

Workflow:
[Multi-view Camera + Stereo/LiDAR frames] → [BEV Occupancy Network] → [MC Dropout Forward Passes (N samples)] → [Compute Per-Cell Variance] → (If Variance Spikes) → [Flag Low-Confidence Region] / (Else) → [Commit Occupancy Prediction]

Roadmap

Phase 1: Data Ingestion & BEV Projection

Stack: Python, Waymo API, KITTI devkit, NumPy
Ingest weather-tagged camera + stereo depth frames; project into a shared bird's-eye-view (BEV) grid using calibration tensors.

Phase 2: Lightweight BEV Occupancy Network

Stack: PyTorch
Train a compact occupancy network predicting drivable vs. non-drivable cells from BEV features; keep the backbone small for fast iteration.

Phase 3: Uncertainty Estimation via MC Dropout

Stack: PyTorch
Run multiple stochastic forward passes per input; compute per-cell prediction variance as the uncertainty signal, rather than a single deterministic output.

Phase 4: Cross-Modality Conflict Check

Stack: NumPy
Where variance spikes, cross-reference stereo/LiDAR depth in that cell. If modalities disagree, mark the region low-confidence; if they agree despite visual ambiguity, keep the prediction.

Phase 5 (Optional): Synthetic Data & Deployment

Stack: NVIDIA Isaac Sim/Omniverse, TensorRT
Generate extra wet-road/reflective scenes in sim to cover rare weather conditions; optimize the trained model with TensorRT for faster inference in a demo pipeline.


Idea 2: 3D Occupancy World Model for Household Clutter Prediction

Project Summary 
Home robots typically navigate using a static snapshot of a room, but real rooms change throughout the day — dishes pile up near the sink, shoes collect by the door, laundry appears on the floor. A robot that only knows "the room as it was this morning" ends up surprised, re-planning reactively instead of anticipating clutter. This project builds a lightweight world model that learns where and when clutter tends to accumulate in a specific room, so a robot can predict likely future occupancy instead of only reacting to what it currently sees.


Use Case: Predict short-term changes in a static indoor space (object rearrangement only, not people) to help a home robot plan navigation before clutter appears.
Gap Filled: Most indoor 3D occupancy work focuses on one-time scene reconstruction or static mapping; few models treat everyday clutter accumulation as a predictable, learnable temporal pattern tied to a specific room's habits.
Target Users: Home-robotics researchers/hobbyists; more academic/lab-scale exploration than a shipped commercial product.


Workflow:
[Repeated Phone 3D Scans of One Room, over days] → [Convert Each Scan to Voxel Occupancy Grid] → [Align Grids in Time (Temporal Stack)] → [Small Temporal Model learns Occupancy Deltas] → [Predict Future Occupancy Grid (e.g., "by evening")] → [Flag High-Clutter-Probability Zones]

 Roadmap

Phase 1: Data Collection & 3D Scanning

Stack: Phone LiDAR/ARKit or ARCore, Open3D
Scan one room repeatedly over several days at different times (morning/evening) to capture natural clutter cycles.

Phase 2: Voxel Occupancy Grid Construction

Stack: Open3D, NumPy
Convert each raw 3D scan into a fixed-resolution voxel grid marking occupied vs. empty space, aligned to a common room coordinate frame.

Phase 3: Temporal Alignment & Dataset Construction

Stack: Python, NumPy
Stack aligned grids into a time-series per room; construct (past-grid → future-grid) training pairs at varying time gaps (e.g., morning → evening).

Phase 4: Small Temporal Prediction Model

Stack: PyTorch (CUDA-accelerated)
Train a lightweight model (e.g., small 3D ConvLSTM or simple temporal transformer over voxel patches) to predict occupancy deltas — which cells are likely to become occupied — rather than full scene regeneration.

Phase 5: Evaluation & Visualization

Stack: Open3D, Matplotlib
Compare predicted future occupancy against actual later scans; visualize high-probability clutter zones (e.g., near sink, by door) as a heatmap overlay on the room mesh.
Use the predicted occupancy grid to bias a toy path planner away from likely-clutter zones, demonstrating the practical navigation benefit.

