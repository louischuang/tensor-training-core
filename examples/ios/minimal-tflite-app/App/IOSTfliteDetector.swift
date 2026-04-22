import Foundation
import TensorFlowLite
import UIKit

struct IOSDetectionResult {
    let labelID: Int
    let label: String
    let score: Float
    let bboxXywhNorm: [Float]
}

private struct IOSAnchor: Decodable {
    let cx: Float
    let cy: Float
    let w: Float
    let h: Float
}

private struct IOSNMSMetadata: Decodable {
    let scoreThreshold: Float
    let iouThreshold: Float

    private enum CodingKeys: String, CodingKey {
        case scoreThreshold = "score_threshold"
        case iouThreshold = "iou_threshold"
    }
}

private struct IOSPostprocessingMetadata: Decodable {
    let nms: IOSNMSMetadata
}

private struct IOSExportMetadata: Decodable {
    let imageSize: [Int]
    let anchors: [IOSAnchor]
    let postprocessing: IOSPostprocessingMetadata

    private enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case anchors
        case postprocessing
    }
}

final class IOSTfliteDetector {
    private let interpreter: Interpreter
    private let labels: [String]
    private let metadata: IOSExportMetadata

    private init(interpreter: Interpreter, labels: [String], metadata: IOSExportMetadata) {
        self.interpreter = interpreter
        self.labels = labels
        self.metadata = metadata
    }

    static func fromBundle() throws -> IOSTfliteDetector {
        guard
            let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite"),
            let labelPath = Bundle.main.path(forResource: "label", ofType: "txt"),
            let metadataPath = Bundle.main.path(forResource: "export_metadata", ofType: "json")
        else {
            throw NSError(domain: "IOSTfliteDetector", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Required model bundle files are missing from the app bundle."
            ])
        }

        let labels = try String(contentsOfFile: labelPath, encoding: .utf8)
            .split(separator: "\n")
            .map(String.init)
            .filter { !$0.isEmpty }
        let metadata = try JSONDecoder().decode(
            IOSExportMetadata.self,
            from: Data(contentsOf: URL(fileURLWithPath: metadataPath))
        )

        var interpreter = try Interpreter(modelPath: modelPath, options: Interpreter.Options())
        try interpreter.allocateTensors()
        return IOSTfliteDetector(interpreter: interpreter, labels: labels, metadata: metadata)
    }

    func detectSampleImage(named imageName: String) throws -> [IOSDetectionResult] {
        guard let image = UIImage(named: imageName) else {
            throw NSError(domain: "IOSTfliteDetector", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Unable to load sample image \(imageName).jpg from the app bundle."
            ])
        }
        return try detect(image: image)
    }

    func detect(image: UIImage) throws -> [IOSDetectionResult] {
        let inputTensor = try interpreter.input(at: 0)
        let inputData = try makeInputData(image: image, tensor: inputTensor)
        try interpreter.copy(inputData, toInputAt: 0)
        try interpreter.invoke()

        var classScores: [[Float]]?
        var boxOffsets: [[Float]]?
        for outputIndex in 0..<interpreter.outputTensorCount {
            let tensor = try interpreter.output(at: outputIndex)
            let values = try readTensorValues(tensor)
            let shape = tensor.shape.dimensions
            if shape.last == 4 {
                boxOffsets = reshape2D(values: values, rows: shape[1], cols: shape[2])
            } else {
                classScores = reshape2D(values: values, rows: shape[1], cols: shape[2])
            }
        }

        guard let classScores, let boxOffsets else {
            throw NSError(domain: "IOSTfliteDetector", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Expected both class and bbox output tensors."
            ])
        }

        return runNMS(classScores: classScores, boxOffsets: boxOffsets)
    }

    private func makeInputData(image: UIImage, tensor: Tensor) throws -> Data {
        let width = metadata.imageSize[0]
        let height = metadata.imageSize[1]
        guard let resized = resize(image: image, width: width, height: height) else {
            throw NSError(domain: "IOSTfliteDetector", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Unable to resize the input image."
            ])
        }
        guard let cgImage = resized.cgImage else {
            throw NSError(domain: "IOSTfliteDetector", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Unable to access CGImage data."
            ])
        }

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var rawBytes = [UInt8](repeating: 0, count: height * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(
            data: &rawBytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw NSError(domain: "IOSTfliteDetector", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Unable to create bitmap context."
            ])
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        let quantization = tensor.quantizationParameters
        var data = Data(capacity: width * height * 3 * 4)
        for pixelIndex in stride(from: 0, to: rawBytes.count, by: bytesPerPixel) {
            let rgb = [
                Float(rawBytes[pixelIndex]) / 255.0,
                Float(rawBytes[pixelIndex + 1]) / 255.0,
                Float(rawBytes[pixelIndex + 2]) / 255.0,
            ]
            for channel in rgb {
                switch tensor.dataType {
                case .float32:
                    var value = channel
                    data.append(Data(bytes: &value, count: MemoryLayout<Float>.size))
                case .uInt8:
                    let value: UInt8
                    if quantization.scale != 0 {
                        let quantized = Int((channel / quantization.scale) + Float(quantization.zeroPoint))
                        value = UInt8(max(0, min(255, quantized)))
                    } else {
                        value = UInt8(max(0, min(255, Int(channel * 255.0))))
                    }
                    data.append(value)
                default:
                    throw NSError(domain: "IOSTfliteDetector", code: 7, userInfo: [
                        NSLocalizedDescriptionKey: "Unsupported input tensor type: \(tensor.dataType)"
                    ])
                }
            }
        }
        return data
    }

    private func readTensorValues(_ tensor: Tensor) throws -> [Float] {
        let quantization = tensor.quantizationParameters
        switch tensor.dataType {
        case .float32:
            let count = tensor.data.count / MemoryLayout<Float>.size
            return tensor.data.withUnsafeBytes { buffer in
                let pointer = buffer.bindMemory(to: Float.self)
                return Array(pointer.prefix(count))
            }
        case .uInt8:
            return tensor.data.map { raw in
                if quantization.scale != 0 {
                    return (Float(raw) - Float(quantization.zeroPoint)) * quantization.scale
                }
                return Float(raw)
            }
        default:
            throw NSError(domain: "IOSTfliteDetector", code: 8, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported output tensor type: \(tensor.dataType)"
            ])
        }
    }

    private func reshape2D(values: [Float], rows: Int, cols: Int) -> [[Float]] {
        var output = Array(repeating: Array(repeating: Float(0), count: cols), count: rows)
        for row in 0..<rows {
            for col in 0..<cols {
                output[row][col] = values[(row * cols) + col]
            }
        }
        return output
    }

    private func runNMS(classScores: [[Float]], boxOffsets: [[Float]]) -> [IOSDetectionResult] {
        var candidates: [IOSDetectionResult] = []
        for anchorIndex in classScores.indices {
            let scores = classScores[anchorIndex]
            guard scores.count > 1 else { continue }
            var bestIndex = 1
            var bestScore = scores[1]
            if scores.count > 2 {
                for index in 2..<scores.count where scores[index] > bestScore {
                    bestIndex = index
                    bestScore = scores[index]
                }
            }
            guard bestScore >= metadata.postprocessing.nms.scoreThreshold else { continue }
            let labelID = bestIndex
            let label = labels.indices.contains(labelID - 1) ? labels[labelID - 1] : "class_\(labelID)"
            candidates.append(
                IOSDetectionResult(
                    labelID: labelID,
                    label: label,
                    score: bestScore,
                    bboxXywhNorm: decodeBox(offsets: boxOffsets[anchorIndex], anchor: metadata.anchors[anchorIndex])
                )
            )
        }

        return candidates.sorted { $0.score > $1.score }.reduce(into: []) { selected, candidate in
            let overlaps = selected.contains { current in
                iou(candidate.bboxXywhNorm, current.bboxXywhNorm) >= metadata.postprocessing.nms.iouThreshold
            }
            if !overlaps {
                selected.append(candidate)
            }
        }
    }

    private func decodeBox(offsets: [Float], anchor: IOSAnchor) -> [Float] {
        let boxCenterX = (offsets[0] * anchor.w) + anchor.cx
        let boxCenterY = (offsets[1] * anchor.h) + anchor.cy
        let boxWidth = exp(offsets[2]) * anchor.w
        let boxHeight = exp(offsets[3]) * anchor.h
        let x = max(0, min(1, boxCenterX - (boxWidth / 2)))
        let y = max(0, min(1, boxCenterY - (boxHeight / 2)))
        return [x, y, max(0, min(1, boxWidth)), max(0, min(1, boxHeight))]
    }

    private func iou(_ a: [Float], _ b: [Float]) -> Float {
        let ax0 = a[0]
        let ay0 = a[1]
        let ax1 = a[0] + a[2]
        let ay1 = a[1] + a[3]
        let bx0 = b[0]
        let by0 = b[1]
        let bx1 = b[0] + b[2]
        let by1 = b[1] + b[3]

        let interX0 = max(ax0, bx0)
        let interY0 = max(ay0, by0)
        let interX1 = min(ax1, bx1)
        let interY1 = min(ay1, by1)
        let interWidth = max(0, interX1 - interX0)
        let interHeight = max(0, interY1 - interY0)
        let intersection = interWidth * interHeight
        let union = (a[2] * a[3]) + (b[2] * b[3]) - intersection
        return union <= 0 ? 0 : intersection / union
    }

    private func resize(image: UIImage, width: Int, height: Int) -> UIImage? {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height))
        return renderer.image { _ in
            image.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        }
    }
}
