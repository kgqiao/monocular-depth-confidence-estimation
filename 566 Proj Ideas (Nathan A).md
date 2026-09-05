

### Photo Scan to 3D World - Fill in the Gaps

>[!note]
>Topics: 3D, AR, Video World Models


- Scan a room with an iPhone. Reconstruct what the camera saw using an off-the-shelf feed-forward model (π³ or MapAnything).
- Render the partial reconstruction from viewpoints it didn't cover. You get images with holes.
- A camera-conditioned video diffusion model fills the holes. Lift those generated views back into the 3D map.
- Output: a complete room, with every surface labeled observed or generated.


### Self-Supervised Anomaly Detection using Video World Model for AV

>[!note]
>Topics: Video World Model, Autonomous Driving

- Learn "normal" behavior from the scene over time
- Event clustering and statistical mapping
- Flag anomalous behavior observed: new road closure, wet spill, etc.


### Learned Frame Budgeting for VLMs

>[!note]
>Topics: Vision Language Model, Edge Engineering

- The bottleneck for running VLMs on the edge is how many frames are pushed through the vision encoder
- Everyone picks a low FPS and lives with it
- Instead, we train a tiny gating model (class CV) to score the interest of the frame, then pass it to the full VLM for encoding
- Create a set of benchmarks where high-value but high-speed events are still caught in near-realtime, even on a slow edge system


### Limits of Cosmos 2B

>[!note]
>Topics: Video World Model, Edge Engineering

- NVIDIA Cosmos world model for the edge
- Take Cosmos Reason2-2B and sweep FP16 → FP8 → W4A16 → W4A8 → W4A4. 
- Build a small eval set of physical-plausibility questions: object permanence, contact, gravity, temporal ordering
- Measure which capability (spatial, language, temporal grounding) breaks down first in compression
- Gives a good sense of how good of a performance you can get on the edge
