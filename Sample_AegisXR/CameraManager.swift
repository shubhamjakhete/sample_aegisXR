//
//  CameraManager.swift
//  Sample_AegisXR
//
//  Real-time camera capture using AVFoundation with Swift Concurrency.
//

import AVFoundation
import SwiftUI

/// Actor-isolated camera manager for efficient frame capture.
actor CameraManager: Sendable {
    private(set) var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private let (stream, streamContinuation) = AsyncStream.makeStream(of: CVPixelBuffer.self)
    private var isRunning = false

    /// Process camera frames - call from outside actor; stream never leaves actor
    func consumeFrames(throttle: Int = 2, _ process: @escaping (CVPixelBuffer) async -> Void) async {
        var count = 0
        for await buffer in stream {
            count += 1
            if count % throttle == 0 {
                await process(buffer)
            }
        }
    }

    /// Configure and start the camera session
    func start() async throws {
        guard !isRunning else { return }

        let session = AVCaptureSession()
        session.sessionPreset = .hd1280x720
        session.beginConfiguration()

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw CameraError.noCamera
        }

        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else {
            throw CameraError.cannotAddInput
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        await MainActor.run {
            output.setSampleBufferDelegate(DelegateRelay.shared, queue: DelegateRelay.queue)
        }

        guard session.canAddOutput(output) else {
            throw CameraError.cannotAddOutput
        }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            connection.videoRotationAngle = 90
            connection.isVideoMirrored = false
        }

        session.commitConfiguration()
        captureSession = session
        videoOutput = output
        await MainActor.run {
            DelegateRelay.shared.setContinuation { [weak self] buffer in
                Task { await self?.deliver(buffer) }
            }
        }
        session.startRunning()
        isRunning = true
    }

    private func deliver(_ buffer: CVPixelBuffer) {
        streamContinuation.yield(buffer)
    }

    /// Stop the camera session
    func stop() {
        captureSession?.stopRunning()
        captureSession = nil
        videoOutput = nil
        streamContinuation.finish()
        isRunning = false
    }

    enum CameraError: Error {
        case noCamera
        case cannotAddInput
        case cannotAddOutput
        case permissionDenied
    }
}

// MARK: - Delegate relay (AVCaptureVideoDataOutputSampleBufferDelegate is not Sendable)
private final class DelegateRelay: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    static let shared = DelegateRelay()
    static let queue = DispatchQueue(label: "com.sampleaegisxr.camera", qos: .userInitiated)

    private var handler: ((CVPixelBuffer) -> Void)?

    func setContinuation(_ handler: @escaping (CVPixelBuffer) -> Void) {
        self.handler = handler
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        handler?(pixelBuffer)
    }
}
