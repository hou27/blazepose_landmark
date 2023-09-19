# BlazePose Model을 사용한 Pose Extraction

### Reference

[MediaPipe](https://google.github.io/mediapipe/solutions/pose.html)

## MediaPipe Pose

```bash
pip install -q mediapipe==0.10.0
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

This repo hosts the official MediaPipe samples with a goal of showing the fundamental steps involved to create apps with our machine learning platform.

External PRs for fixes are welcome, however new sample/demo PRs will likely be rejected to maintain the simplicity of this repo for ongoing maintenance. It is strongly recommended that contributors who are interested in submitting more complex samples or demos host their samples in their own public repos and create written tutorials to share with the community. Contributors can also submit these projects and tutorials to the [Google DevLibrary](https://devlibrary.withgoogle.com/)

MediaPipe Solutions streamlines on-device ML development and deployment with flexible low-code / no-code tools that provide the modular building blocks for creating custom high-performance solutions for cross-platform deployment. It consists of the following components:

- MediaPipe Tasks (low-code): create and deploy custom e2e ML solution pipelines
- MediaPipe Model Maker (low-code): create custom ML models from advanced solutions
- MediaPipe Studio (no-code): create, evaluate, debug, benchmark, prototype, deploy advanced production-level solutions
