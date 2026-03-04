//
//  ObjectDetectionView.swift
//  Sample_AegisXR
//
//  Main SwiftUI view for real-time object detection with camera feed.
//

import AVFoundation
import SwiftUI

struct ObjectDetectionView: View {
    @State private var permissionGranted = false
    @State private var permissionDenied = false
    @State private var errorMessage: String?
    @State private var captureSession: AVCaptureSession?
    @State private var currentTrackedObjects: [TrackedObject] = []
    @State private var isDetecting = false

    private let cameraManager = CameraManager()
    private let detectionService = DetectionService()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                if permissionGranted {
                    CameraPreviewView(session: captureSession)
                        .ignoresSafeArea()

                    OverlayView(trackedObjects: currentTrackedObjects, viewSize: geo.size)
                        .ignoresSafeArea()

                VStack {
                    Spacer()
                    HStack {
                        Text("\(currentTrackedObjects.count) objects")
                            .font(.caption)
                            .padding(8)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                        Spacer()
                    }
                    .padding()
                }
            } else if permissionDenied {
                permissionDeniedView
            } else {
                ProgressView("Checking camera access...")
            }
            }
        }
        .task {
            await requestPermissionAndStart()
        }
        .onDisappear {
            Task { await cameraManager.stop() }
        }
    }

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill")
                .font(.system(size: 60))
                .foregroundStyle(.secondary)
            Text("Camera Access Required")
                .font(.headline)
            Text("Please enable camera access in Settings for real-time object detection.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
    }

    private func requestPermissionAndStart() async {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            permissionGranted = true
            await startCameraAndDetection()
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            await MainActor.run {
                permissionGranted = granted
                permissionDenied = !granted
            }
            if granted {
                await startCameraAndDetection()
            }
        default:
            await MainActor.run {
                permissionDenied = true
            }
        }
    }

    private func startCameraAndDetection() async {
        do {
            try await cameraManager.start()
            let session = await cameraManager.captureSession
            await MainActor.run {
                captureSession = session
            }
            await runDetectionLoop()
        } catch {
            await MainActor.run {
                errorMessage = error.localizedDescription
                permissionDenied = true
            }
        }
    }

    private func runDetectionLoop() async {
        isDetecting = true
        await cameraManager.consumeFrames(throttle: 2) { [detectionService] buffer in
            let tracked = await detectionService.detect(pixelBuffer: buffer)
            await MainActor.run {
                currentTrackedObjects = tracked
            }
        }
    }
}

/// UIViewRepresentable for displaying the camera preview
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession?

    func makeUIView(context: Context) -> PreviewUIView {
        PreviewUIView()
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.previewLayer.session = session
    }
}

final class PreviewUIView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        previewLayer.videoGravity = .resizeAspectFill
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        previewLayer.frame = bounds
    }
}
