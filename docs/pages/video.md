# Video

## Inference

```bash
psifx video pose mediapipe inference --video Videos/Right.mp4 --poses Poses/Right.tar.xz --masks Masks/Right.mp4
psifx video pose mediapipe inference --video Videos/Left.mp4 --poses Poses/Left.tar.xz --masks Masks/Left.mp4

psifx video face openface inference --video Videos/Right.mp4 --features Faces/Right.tar.xz
psifx video face openface inference --video Videos/Left.mp4 --features Faces/Left.tar.xz
```

## Visualization

```bash
psifx video pose mediapipe visualization --video Videos/Right.mp4 --poses Poses/Right.tar.xz --visualization Visualizations/Right.mediapipe.mp4
psifx video pose mediapipe visualization --video Videos/Left.mp4 --poses Poses/Left.tar.xz --visualization Visualizations/Left.mediapipe.mp4

psifx video face openface visualization --video Videos/Right.mp4 --features Faces/Right.tar.xz --visualization Visualizations/Right.openface.mp4
psifx video face openface visualization --video Videos/Left.mp4 --features Faces/Left.tar.xz --visualization Visualizations/Left.openface.mp4
```