//
//  DetectionService.swift
//  Sample_AegisXR
//
//  Object detection using Vision and Core ML with Neural Engine optimization.
//

import CoreML
import Vision
import SwiftUI

/// A single detected object with bounding box and label
struct DetectedObject: Sendable {
    let label: String
    let confidence: Float
    let boundingBox: CGRect
}

/// Actor-isolated detection service using Vision + Core ML
actor DetectionService: Sendable {
    private var visionModel: VNCoreMLModel?
    private var request: VNCoreMLRequest?
    private let modelName = "yolo11n"
    private let confidenceThreshold: Float = 0.25
    private let nmsThreshold: Float = 0.45

    // COCO class names (80 classes)
    private static let cocoLabels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    init() {}

    private static var hasLoggedModelNotFound = false

    private func loadModel() {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine preferred for optimal performance

        // Try multiple locations: Xcode may compile .mlpackage → .mlmodelc, or keep as .mlpackage
        var modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlpackage", subdirectory: "Models")
            ?? Bundle.main.url(forResource: modelName, withExtension: "mlpackage")
            ?? Bundle.main.url(forResource: modelName, withExtension: "mlmodelc", subdirectory: "Models")
            ?? Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")

        if modelURL == nil, let urls = Bundle.main.urls(forResourcesWithExtension: "mlpackage", subdirectory: nil) {
            modelURL = urls.first { $0.lastPathComponent.contains("yolo11") }
        }
        if modelURL == nil, let urls = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil) {
            modelURL = urls.first { $0.lastPathComponent.contains("yolo11") }
        }

        // Fallback: search bundle recursively
        if modelURL == nil, let bundleURL = Bundle.main.resourceURL {
            if let found = findModel(in: bundleURL, name: "yolo11n") {
                modelURL = found
            }
        }

        guard let modelURL else {
            if !Self.hasLoggedModelNotFound {
                Self.hasLoggedModelNotFound = true
                print("YOLO model not found. Add yolo11n.mlpackage to the app: run 'python convert.py' then drag the model into Xcode's Models group.")
            }
            return
        }

        do {
            let urlToLoad: URL
            let path = modelURL.standardizedFileURL.path
            let needsCompile = path.contains("mlpackage") && !path.contains("mlmodelc")
            if needsCompile {
                // .mlpackage must be compiled to .mlmodelc before loading on device
                let packageURL = modelURL.standardizedFileURL
                urlToLoad = try compileIfNeeded(packageURL: packageURL)
            } else {
                urlToLoad = modelURL
            }

            let mlModel = try MLModel(contentsOf: urlToLoad, configuration: config)
            let vnModel = try VNCoreMLModel(for: mlModel)
            visionModel = vnModel

            let req = VNCoreMLRequest(model: vnModel) { _, _ in }
            req.imageCropAndScaleOption = .scaleFill
            request = req
        } catch {
            if !Self.hasLoggedModelNotFound {
                Self.hasLoggedModelNotFound = true
                print("Failed to load model: \(error.localizedDescription)")
            }
        }
    }

    /// Compile .mlpackage to .mlmodelc (required on device). Caches result for subsequent launches.
    private func compileIfNeeded(packageURL: URL) throws -> URL {
        let fileManager = FileManager.default
        let cacheDir = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let compiledDir = cacheDir.appendingPathComponent("yolo11n.mlmodelc", isDirectory: true)

        if fileManager.fileExists(atPath: compiledDir.path) {
            return compiledDir
        }

        let compiledURL = try MLModel.compileModel(at: packageURL)
        try fileManager.copyItem(at: compiledURL, to: compiledDir)
        return compiledDir
    }

    private func findModel(in directory: URL, name: String) -> URL? {
        guard let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { return nil }
        for case let url as URL in enumerator {
            let last = url.lastPathComponent
            if last == "\(name).mlpackage" || last == "\(name).mlmodelc" {
                return url
            }
        }
        return nil
    }

    private let tracker = ObjectTracker()

    /// Run detection on a pixel buffer; returns stabilized tracked objects
    func detect(pixelBuffer: CVPixelBuffer) async -> [TrackedObject] {
        if request == nil { loadModel() }
        guard request != nil else { return [] }

        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let imageSize = CGSize(width: width, height: height)

        let rawDetections: [DetectedObject] = await withCheckedContinuation { continuation in
            let req = VNCoreMLRequest(model: visionModel!) { request, error in
                if let error = error {
                    print("Detection error: \(error)")
                    continuation.resume(returning: [])
                    return
                }
                Task { [weak self] in
                    guard let self = self else { return }
                    let results = await self.processResults(request.results ?? [], imageSize: imageSize)
                    continuation.resume(returning: results)
                }
            }
            req.imageCropAndScaleOption = .scaleFill
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            do {
                try handler.perform([req])
            } catch {
                print("Handler error: \(error)")
                continuation.resume(returning: [])
            }
        }

        return await tracker.update(detections: rawDetections)
    }

    private func processResults(_ results: [Any], imageSize: CGSize) -> [DetectedObject] {
        var detections: [(label: String, confidence: Float, box: CGRect)] = []

        for result in results {
            if let obs = result as? VNRecognizedObjectObservation,
               let label = obs.labels.first, label.confidence >= confidenceThreshold {
                let box = obs.boundingBox
                let name: String
                if let idx = Int(label.identifier), (0..<Self.cocoLabels.count).contains(idx) {
                    name = Self.cocoLabels[idx]
                } else {
                    name = label.identifier
                }
                detections.append((name, label.confidence, box))
            }
        }

        // If we get VNRecognizedObjectObservation, return directly
        if !detections.isEmpty {
            return detections.map { DetectedObject(label: $0.label, confidence: $0.confidence, boundingBox: $0.box) }
        }

        // Fallback: parse VNCoreMLFeatureValueObservation (NMS-free YOLO raw output)
        for result in results {
            guard let obs = result as? VNCoreMLFeatureValueObservation,
                  let mlFeature = obs.featureValue.multiArrayValue else { continue }

            let parsed = parseYOLOOutput(mlFeature, imageSize: imageSize)
            detections.append(contentsOf: parsed)
        }

        let nmsFiltered = runNMS(detections)
        return nmsFiltered.map { DetectedObject(label: $0.label, confidence: $0.confidence, boundingBox: $0.box) }
    }

    private func parseYOLOOutput(_ output: MLMultiArray, imageSize: CGSize) -> [(label: String, confidence: Float, box: CGRect)] {
        var detections: [(label: String, confidence: Float, box: CGRect)] = []
        // YOLO output: (1, 84, 8400) - 4 bbox + 80 classes, 8400 proposals
        let shape = output.shape
        guard shape.count >= 2 else { return [] }
        let numClasses = 80
        let numPredictions: Int
        if shape.count == 3 {
            numPredictions = shape[2].intValue
        } else {
            return []
        }

        guard shape.count == 3 else { return [] }

        for i in 0..<Swift.min(numPredictions, 8400) {
            var maxScore: Float = 0
            var maxIndex = 0
            for c in 0..<numClasses {
                let score = output[[0, 4 + c, i] as [NSNumber]].floatValue
                if score > maxScore {
                    maxScore = score
                    maxIndex = c
                }
            }
            guard maxScore >= confidenceThreshold else { continue }

            let cx = output[[0, 0, i] as [NSNumber]].floatValue
            let cy = output[[0, 1, i] as [NSNumber]].floatValue
            let w = output[[0, 2, i] as [NSNumber]].floatValue
            let h = output[[0, 3, i] as [NSNumber]].floatValue

            // YOLO: (cx, cy, w, h) normalized 0-1, top-left origin
            // Vision: (x, y, w, h) normalized 0-1, origin bottom-left
            let x = CGFloat(cx - w / 2)
            let y = CGFloat(1 - (cy + h / 2))
            let box = CGRect(x: x, y: y, width: CGFloat(w), height: CGFloat(h))
            let label = maxIndex < Self.cocoLabels.count ? Self.cocoLabels[maxIndex] : "\(maxIndex)"
            detections.append((label, maxScore, box))
        }
        return detections
    }

    private func runNMS(_ detections: [(label: String, confidence: Float, box: CGRect)]) -> [(label: String, confidence: Float, box: CGRect)] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var result: [(label: String, confidence: Float, box: CGRect)] = []

        for det in sorted {
            var keep = true
            for kept in result {
                if det.label == kept.label && iou(det.box, kept.box) > nmsThreshold {
                    keep = false
                    break
                }
            }
            if keep { result.append(det) }
        }
        return Array(result.prefix(20))
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let union = a.union(b)
        return Float((inter.width * inter.height) / (union.width * union.height))
    }
}
