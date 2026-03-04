# Sample_AegisXR

An iOS app for **real-time object detection** using the camera, built with SwiftUI and Apple's on-device ML stack. Detected objects (people, vehicles, animals, everyday items) are shown with bounding boxes and labels over the live camera feed.

---

## What We've Done So Far

### Core Features Implemented

1. **Real-time camera capture** – Live video from the back camera at 1280×720
2. **Object detection** – YOLO11n model (80 COCO classes) running on-device with the Neural Engine
3. **Object tracking** – IoU-based tracking with EMA smoothing for stable labels across frames
4. **Visual overlay** – Bounding boxes and confidence labels drawn over the camera preview
5. **Camera permission handling** – Requests and handles camera access with clear feedback

### Technical Highlights

- **100% Swift** – No Python or PyTorch in the app
- **Swift concurrency** – Actors and `async/await` for camera and detection pipelines
- **Apple frameworks only** – SwiftUI, AVFoundation, Vision, Core ML
- **Optimized for on-device** – Core ML uses Neural Engine when available

---

## Tech Stack

| Framework    | Purpose                                                |
|-------------|--------------------------------------------------------|
| **SwiftUI** | UI, layout, and state management                       |
| **AVFoundation** | Camera capture, `AVCaptureSession`, pixel buffers |
| **Vision**  | ML inference (`VNCoreMLRequest`, `VNImageRequestHandler`) |
| **Core ML** | Model loading and execution (Neural Engine)            |
| **UIKit**   | `UIViewRepresentable` for `AVCaptureVideoPreviewLayer` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ObjectDetectionView                          │
│  (SwiftUI: permission, camera preview, overlay, detection loop)  │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  CameraManager  │  │ DetectionService │  │   OverlayView   │
│  (Actor)        │  │ (Actor)          │  │ (SwiftUI)       │
│                 │  │                  │  │                 │
│ • AVFoundation  │  │ • Vision         │  │ • Canvas        │
│ • AsyncStream   │  │ • Core ML        │  │ • Labels        │
│ • Frame throttle│  │ • YOLO11n        │  │ • Bounding box  │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         │  CVPixelBuffer     │  [DetectedObject]  │  [TrackedObject]
         └───────────────────►│                    │◄──────────────────┘
                              │  ObjectTracker     │
                              │  (Actor)           │
                              │  • IoU matching    │
                              │  • EMA smoothing   │
                              │  • Confirm/hold    │
                              └───────────────────┘
```

### Key Components

| Component        | File                 | Description                                                                 |
|-----------------|----------------------|-----------------------------------------------------------------------------|
| **CameraManager** | `CameraManager.swift` | Actor that configures and runs `AVCaptureSession`, streams `CVPixelBuffer` frames via `AsyncStream` with optional throttling. |
| **DetectionService** | `DetectionService.swift` | Actor that loads the YOLO11n Core ML model, runs Vision inference, parses detections (including raw YOLO output), runs NMS, and returns tracked objects. |
| **ObjectTracker** | `ObjectTracker.swift` | Actor that matches detections across frames using IoU, smooths boxes and confidence with EMA, and uses confirm/hold logic for stable display. |
| **ObjectDetectionView** | `ObjectDetectionView.swift` | Main SwiftUI view: permission flow, camera preview, overlay, detection loop, and object count. |
| **OverlayView** | `OverlayView.swift` | SwiftUI view that draws bounding boxes and labels on a `Canvas`, converting Vision (bottom-left) coordinates to view coordinates. |

---

## Project Structure

```
Sample_AegisXR/
├── Sample_AegisXRApp.swift      # App entry point
├── ContentView.swift            # Root view → ObjectDetectionView
├── ObjectDetectionView.swift    # Main detection UI + camera preview
├── CameraManager.swift          # AVFoundation camera capture
├── DetectionService.swift       # Vision + Core ML detection
├── ObjectTracker.swift          # Tracking & smoothing
├── OverlayView.swift            # Bounding box overlay
├── Models/
│   └── yolo11n.mlpackage       # YOLO11n Core ML model (80 COCO classes)
└── Assets.xcassets/            # App icons, accent color
```

---

## Model: YOLO11n

- **Model:** YOLO11n (YOLOv11 nano)
- **Format:** Core ML (`.mlpackage`)
- **Classes:** 80 COCO categories (person, car, dog, etc.)
- **Thresholds:** Confidence ≥ 0.25, NMS IoU 0.45
- **Compute:** `.all` (Neural Engine preferred on supported devices)

The model is bundled with the app. If you replace it, ensure it is converted to Core ML and added to the project (e.g. via `python convert.py` if you use a conversion script).

---

## Requirements

- **Xcode** 15+ (or 26.x per project settings)
- **iOS** 17+ (or your deployment target)
- **Device** with camera (physical device recommended for real-time performance)
- **Camera permission** for object detection

---

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/shubhamjakhete/sample_aegisXR.git
   cd sample_aegisXR
   ```

2. **Open in Xcode**
   ```bash
   open Sample_AegisXR.xcodeproj
   ```

3. **Select a device** – Choose a physical iPhone or iPad (simulator has no camera)

4. **Run** – Build and run (⌘R). Grant camera access when prompted.

5. **Use the app** – Point the camera at people, vehicles, or common objects; labels and boxes appear in real time.

---

## Possible Next Steps

- [ ] Depth estimation for objects/vehicles (e.g. ARKit, AVFoundation depth, or Core ML depth model)
- [ ] Additional object classes or custom models
- [ ] Export/snapshot of detections
- [ ] AR overlay (RealityKit / ARKit)
- [ ] Performance tuning (throttling, resolution)

---

## Author

**Shubham Jakhete**

---

## License

See project license if applicable.
