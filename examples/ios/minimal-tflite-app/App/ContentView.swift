import SwiftUI

struct ContentView: View {
    @State private var statusText: String = "Loading TFLite bundle from app resources..."

    var body: some View {
        ScrollView {
            Text(statusText)
                .font(.body.monospaced())
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(24)
        }
        .task {
            statusText = await runSampleInference()
        }
    }

    private func runSampleInference() async -> String {
        do {
            let detector = try IOSTfliteDetector.fromBundle()
            let results = try detector.detectSampleImage(named: "sample_input")
            if results.isEmpty {
                return "No detections above the configured score threshold."
            }
            var lines = ["Top detections", ""]
            for (index, result) in results.enumerated() {
                lines.append(
                    "\(index + 1). \(result.label) " +
                    "(score=\(String(format: "%.3f", result.score))) " +
                    "bbox=\(result.bboxXywhNorm)"
                )
            }
            return lines.joined(separator: "\n")
        } catch {
            return [
                "Bundle load or inference failed.",
                "",
                error.localizedDescription,
                "",
                "Make sure App/Resources contains:",
                "- model.tflite",
                "- label.txt",
                "- export_metadata.json",
                "- sample_input.jpg",
            ].joined(separator: "\n")
        }
    }
}
