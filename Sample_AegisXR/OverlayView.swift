//
//  OverlayView.swift
//  Sample_AegisXR
//
//  SwiftUI overlay for drawing bounding boxes and labels over camera feed.
//

import SwiftUI

struct OverlayView: View {
    let trackedObjects: [TrackedObject]
    let viewSize: CGSize

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                for obj in trackedObjects {
                    let rect = denormalize(obj.boundingBox, in: size)
                    context.stroke(
                        Path(roundedRect: rect, cornerRadius: 4),
                        with: .color(.green),
                        lineWidth: 3
                    )
                }
            }
            .overlay(alignment: .topLeading) {
                ForEach(trackedObjects) { obj in
                    let rect = denormalize(obj.boundingBox, in: geo.size)
                    Text("\(obj.label) \(Int(obj.confidence * 100))%")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)
                        .padding(4)
                        .background(Color.green.opacity(0.8), in: RoundedRectangle(cornerRadius: 4))
                        .position(x: rect.minX + 40, y: max(rect.minY - 10, 14))
                }
            }
        }
        .allowsHitTesting(false)
    }

    /// Convert Vision normalized rect (origin bottom-left) to view coordinates
    private func denormalize(_ rect: CGRect, in size: CGSize) -> CGRect {
        // Vision: origin bottom-left, normalized 0-1
        let x = rect.origin.x * size.width
        let y = (1 - rect.origin.y - rect.height) * size.height
        let w = rect.width * size.width
        let h = rect.height * size.height
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

#Preview {
    OverlayView(
        trackedObjects: [
            TrackedObject(id: 1, label: "person", confidence: 0.92, boundingBox: CGRect(x: 0.2, y: 0.2, width: 0.3, height: 0.5)),
            TrackedObject(id: 2, label: "car", confidence: 0.85, boundingBox: CGRect(x: 0.5, y: 0.5, width: 0.25, height: 0.2))
        ],
        viewSize: CGSize(width: 375, height: 812)
    )
    .frame(width: 375, height: 812)
}
