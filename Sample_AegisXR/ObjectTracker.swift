//
//  ObjectTracker.swift
//  Sample_AegisXR
//
//  Stabilization and tracking layer: IoU matching, EMA smoothing, confirm/hold logic.
//

import CoreGraphics
import Foundation

/// Output type for the UI: stable tracked object with persistent ID
struct TrackedObject: Sendable, Identifiable {
    let id: Int  // trackId - use for SwiftUI identity
    let label: String
    let confidence: Float  // smoothed
    let boundingBox: CGRect  // smoothed, Vision normalized (origin bottom-left)
}

/// Internal track state: bbox/confidence are smoothed (EMA)
private struct Track: Sendable {
    let trackId: Int
    var label: String
    var bboxNormalized: CGRect  // smoothed, Vision normalized
    var confidence: Float       // smoothed
    var ageFrames: Int          // frames since creation
    var missedFrames: Int       // consecutive frames without detection
    var isConfirmed: Bool       // shown only after confirmFrames
}

/// IoU-based tracker with EMA smoothing, confirm-before-show, and hold-last-seen
actor ObjectTracker: Sendable {
    private var tracks: [Int: Track] = [:]
    private var nextTrackId = 1

    // Config
    private let maxMissedFrames = 5
    private let confirmFrames = 2
    private let holdLastSeenFrames = 2
    private let emaAlphaBbox: Float = 0.3   // lower = smoother boxes
    private let emaAlphaConf: Float = 0.4
    private let iouMatchThreshold: Float = 0.25

    /// Process raw detections into stable tracked objects
    func update(detections: [DetectedObject]) -> [TrackedObject] {
        // 1. Build (trackId, detIdx, iou) candidates for matching
        typealias Match = (trackId: Int, detIdx: Int, iou: Float)
        var candidates: [Match] = []
        for (trackId, track) in tracks {
            guard track.missedFrames <= holdLastSeenFrames else { continue }
            for (idx, det) in detections.enumerated() {
                guard sameCategory(track.label, det.label) else { continue }
                let iouVal = iou(track.bboxNormalized, det.boundingBox)
                if iouVal >= iouMatchThreshold {
                    candidates.append((trackId, idx, iouVal))
                }
            }
        }
        // Greedy matching by highest IoU first
        candidates.sort { $0.iou > $1.iou }
        var matchedTrackIds: Set<Int> = []
        var matchedDetIndices: Set<Int> = []
        for c in candidates {
            if matchedTrackIds.contains(c.trackId) || matchedDetIndices.contains(c.detIdx) { continue }
            matchedTrackIds.insert(c.trackId)
            matchedDetIndices.insert(c.detIdx)
            let det = detections[c.detIdx]
            var t = tracks[c.trackId]!
            t.bboxNormalized = emaRect(t.bboxNormalized, det.boundingBox, alpha: emaAlphaBbox)
            t.confidence = emaFloat(t.confidence, det.confidence, alpha: emaAlphaConf)
            t.missedFrames = 0
            t.ageFrames += 1
            if t.ageFrames >= confirmFrames { t.isConfirmed = true }
            tracks[c.trackId] = t
        }

        // 2. Increment missedFrames for unmatched tracks
        for (tid, t) in tracks where !matchedTrackIds.contains(tid) {
            var track = t
            track.missedFrames += 1
            tracks[tid] = track
        }

        // 3. Create new tracks for unmatched detections
        for (idx, det) in detections.enumerated() {
            if matchedDetIndices.contains(idx) { continue }
            let tid = nextTrackId
            nextTrackId += 1
            tracks[tid] = Track(trackId: tid, label: det.label, bboxNormalized: det.boundingBox, confidence: det.confidence, ageFrames: 1, missedFrames: 0, isConfirmed: false)
        }

        // 4. Delete tracks when missedFrames > maxMissedFrames
        let toRemove = tracks.filter { $0.value.missedFrames > maxMissedFrames }.map(\.key)
        for tid in toRemove {
            tracks.removeValue(forKey: tid)
        }

        // 5. Output: confirmed tracks within hold window
        return tracks.compactMap { _, t -> TrackedObject? in
            guard t.isConfirmed else { return nil }
            guard t.missedFrames <= holdLastSeenFrames else { return nil }
            return TrackedObject(id: t.trackId, label: t.label, confidence: t.confidence, boundingBox: t.bboxNormalized)
        }
    }

    private func sameCategory(_ a: String, _ b: String) -> Bool {
        if a == b { return true }
        let personLabels: Set<String> = ["person"]
        let vehicleLabels: Set<String> = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        let aPerson = personLabels.contains(a)
        let bPerson = personLabels.contains(b)
        let aVeh = vehicleLabels.contains(a)
        let bVeh = vehicleLabels.contains(b)
        if aPerson && bPerson { return true }
        if aVeh && bVeh { return true }
        return false
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let union = a.union(b)
        return Float((inter.width * inter.height) / (union.width * union.height))
    }
}

private func emaRect(_ old: CGRect, _ new: CGRect, alpha: Float) -> CGRect {
    let a = CGFloat(alpha)
    return CGRect(
        x: old.origin.x * (1 - a) + new.origin.x * a,
        y: old.origin.y * (1 - a) + new.origin.y * a,
        width: old.width * (1 - a) + new.width * a,
        height: old.height * (1 - a) + new.height * a
    )
}

private func emaFloat(_ old: Float, _ new: Float, alpha: Float) -> Float {
    old * (1 - alpha) + new * alpha
}
