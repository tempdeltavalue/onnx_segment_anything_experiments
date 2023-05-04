/**
 * Copyright 2019 The TensorFlow Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Portions Copyright (c) Microsoft Corporation

import Accelerate
import AVFoundation
import CoreImage
import Darwin
import Foundation
import UIKit
import CoreML


extension Array {
    var data: Data { withUnsafeBytes { .init($0) } }
}

// Result struct
struct Result {
    let processTimeMs: Double
    let inferences: [Inference]
}

// Inference struct for ssd model
struct Inference {
    let score: Float
    let className: String
    let rect: CGRect
    let displayColor: UIColor
}

// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

enum OrtModelError: Error {
    case error(_ message: String)
}

class ModelHandler: NSObject {
    // MARK: - Inference Properties

    let threadCount: Int32
    let threshold: Float = 0.5
    let threadCountLimit = 10
    
    // MARK: - Model Parameters

    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 224
    let inputHeight = 224
    
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0 / 255.0, green: 200.0 / 255.0, blue: 250.0 / 255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown
    ]
    
    private var labels: [String] = []
    
    /// ORT inference session and environment object for performing inference on the given ssd model
    private var session: ORTSession
    private var env: ORTEnv
    
    // MARK: - Initialization of ModelHandler
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int32 = 1) {
        let modelFilename = modelFileInfo.name
        
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to get model file path with name: \(modelFilename).")
            return nil
        }
        
        self.threadCount = threadCount
        do {
            // Start the ORT inference environment and specify the options for session
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.info)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.info)
            try options.setIntraOpNumThreads(threadCount)
            // Create the ORTSession
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch {
            print("Failed to create ORTSession.")
            return nil
        }
       
        super.init()
        
        labels = loadLabels(fileInfo: labelsFileInfo)
    }
    
    // This method preprocesses the image, runs the ort inferencesession and returns the inference result
    func runModel(multiArr: MLMultiArray, img_w: Float, img_h:Float) {
        let inputShape: [NSNumber] = [1 as NSNumber,
                                      256 as NSNumber,
                                      64 as NSNumber,
                                      64 as NSNumber]
        
        let input111 = try! MLMultiArray(shape: [1, 256, 64, 64], dataType: .float64)

//        input111.withUnsafeMutableBytes { ptr, strides in
            let data = NSMutableData(data: Data(count: 16777216))
            
            
            let inputTensor = try! ORTValue(tensorData: data,
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape)
            
            let point_coords_data = NSMutableData(data: [176, 68].data)
            
            let inputShape2: [NSNumber] = [1 as NSNumber,
                                           1 as NSNumber,
                                           2 as NSNumber]
            
                        
            let point_coords = try! ORTValue(tensorData: point_coords_data,
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape2)
            // Run ORT InferenceSession
            
            let point_labels_data = NSMutableData(data: [1].data)
            
            let inputShape3: [NSNumber] = [1 as NSNumber,
                                           1 as NSNumber]
            
            let point_labels = try! ORTValue(tensorData: point_labels_data,
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape3)
            
            let rows = 256
            let column = 256
            let zero_array_data = NSMutableData(data: [Float](repeating: 0, count: column * rows).data)
            
            
            let inputShape4: [NSNumber] = [1 as NSNumber,
                                           1 as NSNumber,
                                           rows as NSNumber,
                                           column as NSNumber]
            
            let mask_input = try! ORTValue(tensorData: zero_array_data,
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape4)
            
            
            
            let has_mask_input_data = NSMutableData(data: [0].data)
            
            let inputShape5: [NSNumber] = [1 as NSNumber]
            
            let has_mask_input = try! ORTValue(tensorData: has_mask_input_data,
                                           elementType: ORTTensorElementDataType.float,
                                           shape: inputShape5)
            
            let orig_im_size_data = NSMutableData(data: [img_w, img_h].data)
            
            
            let inputShape6: [NSNumber] = [2 as NSNumber]
            
            let orig_im_size = try! ORTValue(tensorData: orig_im_size_data,
                                             elementType: ORTTensorElementDataType.float,
                                           shape: inputShape6)
            
            let runOptions = try! ORTRunOptions()
            try! runOptions.setLogSeverityLevel(.info)
        
            let outputs = try! session.run(withInputs: ["image_embeddings": inputTensor,
                                                       "point_coords" : point_coords,
                                                       "point_labels" : point_labels,
                                                       "mask_input" : mask_input,
                                                       "has_mask_input" : has_mask_input,
                                                       "orig_im_size" : orig_im_size],
                                          outputNames: ["masks", "iou_predictions", "low_res_masks"],
                                          runOptions: runOptions)
 
            let tensorData = try! outputs["masks"]!.tensorData()
            let tensorInfo = try! outputs["masks"]!.tensorTypeAndShapeInfo()
        
            let tensorData1 = try! outputs["iou_predictions"]!.tensorData()
            let tensorInfo1 = try! outputs["iou_predictions"]!.tensorTypeAndShapeInfo()
            
            let tensorData2 = try! outputs["low_res_masks"]!.tensorData()
            let tensorInfo2 = try! outputs["low_res_masks"]!.tensorTypeAndShapeInfo()

            let ml_arr = try! MLMultiArray(dataPointer: tensorData2.mutableBytes,
                                           shape: [tensorInfo2.shape[2], tensorInfo2.shape[3]],
                                           dataType: MLMultiArrayDataType.float,
                                           strides:[1, 1])
            
            let t_width = tensorInfo.shape[2].intValue
            let t_height = tensorInfo.shape[3].intValue
//            for x in 0..<t_width {
//                for y in 0..<t_height {
//                    let ind = y * t_width + x
//                    if ml_arr[ind].intValue < 0 {
//                        ml_arr[ind] = 0
//                    } else {
//                        ml_arr[ind] = 255
//                    }
//                }
//            }
                  
                  
            let image = ml_arr.image(min: 0, max: 255, channel: 1)
            
            
            
//            var new_ml_arr = try! ml_arr.reshaped(to: [tensorInfo!.shape[2].intValue, tensorInfo!.shape[3].intValue]) // data is 4 larger (
            
//            var ml_arr = try! MLMultiArray(tensorData!)
//            var new_ml_arr = try! ml_arr.reshaped(to: [tensorInfo!.shape[2].intValue, tensorInfo!.shape[3].intValue]) // data is 4 larger (
            
            
            
            var pixelBuffer: UnsafeMutablePointer<CVPixelBuffer?>!
            if pixelBuffer == nil {
            pixelBuffer = UnsafeMutablePointer<CVPixelBuffer?>.allocate(capacity: MemoryLayout<CVPixelBuffer?>.size)
            }
            CVPixelBufferCreate(kCFAllocatorDefault,
                                Int(tensorInfo.shape[2]),
                                Int(tensorInfo.shape[3]),
                                kCVPixelFormatType_OneComponent8,
                                nil, pixelBuffer)

//            let temp_img = UIImage(pixelBuffer: ml_arr.pixelBuffer!)
//            let arr = MLMultiArray(pixelBuffer: pixelBuffer, shape: tensorInfo!.shape)

//        }

    }
    

    // This method preprocesses the image, runs the ort inferencesession and returns the inference result
    func runModel(onFrame pixelBuffer: CVPixelBuffer) throws -> Result? {
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
            sourcePixelFormat == kCVPixelFormatType_32BGRA ||
            sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        let imageChannels = 3
        assert(imageChannels >= inputChannels)
        
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        // Preprocess the image
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = preprocess(ofSize: scaledSize, pixelBuffer) else {
            return nil
        }
        
        let interval: TimeInterval
        
        let inputName = "input_image"
        
        
        // How to convert pixel buffer to float??
        
//        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
//
//        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
//        let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)
//
//        let index = inputWidth*3 + inputHeight*bytesPerRow
//        let b = buffer[index]
//        let g = buffer[index+1]
//        let r = buffer[index+2]
//        buffer[index]
//
        
        guard let rgbData = rgbDataFromBuffer(
            scaledPixelBuffer,
            byteCount: inputWidth * inputHeight * inputChannels
        ) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        let inputShape: [NSNumber] = [inputHeight as NSNumber,
                                      inputWidth as NSNumber,
                                      inputChannels as NSNumber]
        
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: rgbData),
                                       elementType: ORTTensorElementDataType.float,
                                       shape: inputShape)
        
        // Run ORT InferenceSession
        let startDate = Date()
        let outputs = try session.run(withInputs: [inputName: inputTensor],
                                      outputNames: ["embeddings"],
                                      runOptions: nil)
        
        interval = Date().timeIntervalSince(startDate) * 1000
        
        guard let rawOutputValue = outputs["TFLite_Detection_PostProcess"] else {
            throw OrtModelError.error("failed to get model output_0")
        }
        let rawOutputData = try rawOutputValue.tensorData() as Data
        guard let outputArr: [Float32] = Array(unsafeData: rawOutputData) else {
            return nil
        }
        
        guard let rawOutputValue_1 = outputs["TFLite_Detection_PostProcess:1"] else {
            throw OrtModelError.error("failed to get model output_1")
        }
        let rawOutputData_1 = try rawOutputValue_1.tensorData() as Data
        guard let outputArr_1: [Float32] = Array(unsafeData: rawOutputData_1) else {
            return nil
        }
        
        guard let rawOutputValue_2 = outputs["TFLite_Detection_PostProcess:2"] else {
            throw OrtModelError.error("failed to get model output_2")
        }
        let rawOutputData_2 = try rawOutputValue_2.tensorData() as Data
        guard let outputArr_2: [Float32] = Array(unsafeData: rawOutputData_2) else {
            return nil
        }
        
        guard let rawOutputValue_3 = outputs["TFLite_Detection_PostProcess:3"] else {
            throw OrtModelError.error("failed to get model output_3")
        }
        let rawOutputData_3 = try rawOutputValue_3.tensorData() as Data
        guard let outputArr_3: [Float32] = Array(unsafeData: rawOutputData_3) else {
            return nil
        }
        
        /// Output order of ssd mobileNet model: detection boxes/classes/scores/num_detection
        let detectionBoxes = outputArr
        let detectionClasses = outputArr_1
        let detectionScores = outputArr_2
        let numDetections = Int(outputArr_3[0])
        
        // Format the results
        let resultArray = formatResults(detectionBoxes: detectionBoxes,
                                        detectionClasses: detectionClasses,
                                        detectionScores: detectionScores,
                                        numDetections: numDetections,
                                        width: CGFloat(imageWidth),
                                        height: CGFloat(imageHeight))
        
        // Return ORT SessionRun result
        return Result(processTimeMs: interval, inferences: resultArray)
    }
    
    // MARK: - Helper Methods

    // This method postprocesses the results including processing bounding boxes, sort detected scores, etc.
    func formatResults(detectionBoxes: [Float32], detectionClasses: [Float32], detectionScores: [Float32],
                       numDetections: Int, width: CGFloat, height: CGFloat) -> [Inference]
    {
        var resultsArray: [Inference] = []
        
        if numDetections == 0 {
            return resultsArray
        }
        
        for i in 0 ..< numDetections {
            let score = detectionScores[i]
            
            // Filter results with score < threshold.
            guard score >= threshold else {
                continue
            }
            
            let detectionClassIndex = Int(detectionClasses[i])
            let detectionClass = labels[detectionClassIndex + 1]
            
            var rect = CGRect.zero
            
            // Translate the detected bounding box to CGRect.
            rect.origin.y = CGFloat(detectionBoxes[4 * i])
            rect.origin.x = CGFloat(detectionBoxes[4 * i + 1])
            rect.size.height = CGFloat(detectionBoxes[4 * i + 2]) - rect.origin.y
            rect.size.width = CGFloat(detectionBoxes[4 * i + 3]) - rect.origin.x
            
            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
            
            let colorToAssign = colorForClass(withIndex: detectionClassIndex + 1)
            let inference = Inference(score: score,
                                      className: detectionClass,
                                      rect: newRect,
                                      displayColor: colorToAssign)
            resultsArray.append(inference)
        }
        
        // Sort results in descending order of confidence.
        resultsArray.sort { first, second -> Bool in
            first.score > second.score
        }
        
        return resultsArray
    }
    
    // This method preprocesses the image by cropping pixel buffer to biggest square
    // and scaling the cropped image to model dimensions.
    private func preprocess(
        ofSize size: CGSize,
        _ buffer: CVPixelBuffer
    ) -> CVPixelBuffer? {
        let imageWidth = CVPixelBufferGetWidth(buffer)
        let imageHeight = CVPixelBufferGetHeight(buffer)
        let pixelBufferType = CVPixelBufferGetPixelFormatType(buffer)
        
        assert(pixelBufferType == kCVPixelFormatType_32BGRA ||
            pixelBufferType == kCVPixelFormatType_32ARGB)
        
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let imageChannels = 4
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        // Find the biggest square in the pixel buffer and advance rows based on it.
        guard let inputBaseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        // Get vImage_buffer
        var inputVImageBuffer = vImage_Buffer(data: inputBaseAddress,
                                              height: UInt(imageHeight),
                                              width: UInt(imageWidth),
                                              rowBytes: inputImageRowBytes)
        
        let scaledRowBytes = Int(size.width) * imageChannels
        guard let scaledImageBytes = malloc(Int(size.height) * scaledRowBytes) else {
            return nil
        }
                
        var scaledVImageBuffer = vImage_Buffer(data: scaledImageBytes,
                                               height: UInt(size.height),
                                               width: UInt(size.width),
                                               rowBytes: scaledRowBytes)
        
        // Perform the scale operation on input image buffer and store it in scaled vImage buffer.
        let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &scaledVImageBuffer, nil, vImage_Flags(0))
        
        guard scaleError == kvImageNoError else {
            free(scaledImageBytes)
            return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = { _, pointer in
            
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var scaledPixelBuffer: CVPixelBuffer?
        
        // Convert the scaled vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(
            nil, Int(size.width), Int(size.height), pixelBufferType, scaledImageBytes,
            scaledRowBytes, releaseCallBack, nil, nil, &scaledPixelBuffer
        )
        
        guard conversionStatus == kCVReturnSuccess else {
            free(scaledImageBytes)
            return nil
        }
        
        return scaledPixelBuffer
    }
    
    private func loadLabels(fileInfo: FileInfo) -> [String] {
        var labelData: [String] = []
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            print("Labels file not found in bundle. Please add a labels file with name " +
                "\(filename).\(fileExtension)")
            return labelData
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labelData = contents.components(separatedBy: .newlines)
        } catch {
            print("Labels file named \(filename).\(fileExtension) cannot be read.")
        }

        return labelData
    }
    
    private func colorForClass(withIndex index: Int) -> UIColor {
        // Assign variations to the base colors for each object based on its index.
        let baseColor = colors[index % colors.count]
        
        var colorToAssign = baseColor
        
        let percentage = CGFloat((10 / 2 - index / colors.count) * 10)
        
        if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
            colorToAssign = modifiedColor
        }
        
        return colorToAssign
    }
    
    
    
    // Return the RGB data representation of the given image buffer.
    func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool = true
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)
        
        switch pixelBufferFormat {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32RGBA:
            vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        default:
            // Unknown pixel format.
            return nil
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        
        return byteData
    }
}

// MARK: - Extensions

extension Data {
    // Create a new buffer by copying the buffer pointer of the given array.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    // Create a new array from the bytes of the given unsafe data.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
        #if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
        #else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
        #endif // swift(>=5.0)
    }
}

extension UIColor {
    // This method returns colors modified by percentage value of color represented by the current object.
    func getModified(byPercentage percent: CGFloat) -> UIColor? {
        var red: CGFloat = 0.0
        var green: CGFloat = 0.0
        var blue: CGFloat = 0.0
        var alpha: CGFloat = 0.0
        
        guard getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return nil
        }
        
        // Return the color comprised by percentage r g b values of the original color.
        let colorToReturn = UIColor(displayP3Red: min(red + percent / 100.0, 1.0),
                                    green: min(green + percent / 100.0, 1.0),
                                    blue: min(blue + percent / 100.0, 1.0),
                                    alpha: 1.0)
        
        return colorToReturn
    }
}


import Accelerate


extension CVPixelBuffer {
    func vectorNormalize( targetVector: UnsafeMutableBufferPointer<Float>) -> [Float] {
        // range = max - min
        // normalized to 0..1 is (pixel - minPixel) / range

        // see Documentation "Using vDSP for Vector-based Arithmetic" in vDSP under system "Accelerate" documentation

        // see also the Accelerate documentation section 'Vector extrema calculation'
        // Maximium static func maximum<U>(U) -> Float
        //      Returns the maximum element of a single-precision vector.

        //static func minimum<U>(U) -> Float
        //      Returns the minimum element of a single-precision vector.


        let maxValue = vDSP.maximum(targetVector)
        let minValue = vDSP.minimum(targetVector)

        let range = maxValue - minValue
        let negMinValue = -minValue

        let subtractVector = vDSP.add(negMinValue, targetVector)
            // adding negative value is subtracting
        let result = vDSP.divide(subtractVector, range)

        return result
    }

    func setUpNormalize() -> CVPixelBuffer {
        // grayscale buffer float32 ie Float
        // return normalized CVPixelBuffer

        CVPixelBufferLockBaseAddress(self,
                                     CVPixelBufferLockFlags(rawValue: 0))
        let width = CVPixelBufferGetWidthOfPlane(self, 0)
        let height = CVPixelBufferGetHeightOfPlane(self, 0)
        let count = width * height

        let bufferBaseAddress = CVPixelBufferGetBaseAddressOfPlane(self, 0)
            // UnsafeMutableRawPointer

        let pixelBufferBase  = unsafeBitCast(bufferBaseAddress, to: UnsafeMutablePointer<Float>.self)

        let depthCopy  =   UnsafeMutablePointer<Float>.allocate(capacity: count)
        depthCopy.initialize(from: pixelBufferBase, count: count)
        let depthCopyBuffer = UnsafeMutableBufferPointer<Float>(start: depthCopy, count: count)

        let normalizedDisparity = vectorNormalize(targetVector: depthCopyBuffer)

        pixelBufferBase.initialize(from: normalizedDisparity, count: count)
            // copy back the normalized map into the CVPixelBuffer

        depthCopy.deallocate()
//        depthCopyBuffer.deallocate()

        CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))

        return self

    }
    
    func pixelFormatName() -> String {
        let p = CVPixelBufferGetPixelFormatType(self)
        switch p {
        case kCVPixelFormatType_1Monochrome:                   return "kCVPixelFormatType_1Monochrome"
        case kCVPixelFormatType_2Indexed:                      return "kCVPixelFormatType_2Indexed"
        case kCVPixelFormatType_4Indexed:                      return "kCVPixelFormatType_4Indexed"
        case kCVPixelFormatType_8Indexed:                      return "kCVPixelFormatType_8Indexed"
        case kCVPixelFormatType_1IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_1IndexedGray_WhiteIsZero"
        case kCVPixelFormatType_2IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_2IndexedGray_WhiteIsZero"
        case kCVPixelFormatType_4IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_4IndexedGray_WhiteIsZero"
        case kCVPixelFormatType_8IndexedGray_WhiteIsZero:      return "kCVPixelFormatType_8IndexedGray_WhiteIsZero"
        case kCVPixelFormatType_16BE555:                       return "kCVPixelFormatType_16BE555"
        case kCVPixelFormatType_16LE555:                       return "kCVPixelFormatType_16LE555"
        case kCVPixelFormatType_16LE5551:                      return "kCVPixelFormatType_16LE5551"
        case kCVPixelFormatType_16BE565:                       return "kCVPixelFormatType_16BE565"
        case kCVPixelFormatType_16LE565:                       return "kCVPixelFormatType_16LE565"
        case kCVPixelFormatType_24RGB:                         return "kCVPixelFormatType_24RGB"
        case kCVPixelFormatType_24BGR:                         return "kCVPixelFormatType_24BGR"
        case kCVPixelFormatType_32ARGB:                        return "kCVPixelFormatType_32ARGB"
        case kCVPixelFormatType_32BGRA:                        return "kCVPixelFormatType_32BGRA"
        case kCVPixelFormatType_32ABGR:                        return "kCVPixelFormatType_32ABGR"
        case kCVPixelFormatType_32RGBA:                        return "kCVPixelFormatType_32RGBA"
        case kCVPixelFormatType_64ARGB:                        return "kCVPixelFormatType_64ARGB"
        case kCVPixelFormatType_48RGB:                         return "kCVPixelFormatType_48RGB"
        case kCVPixelFormatType_32AlphaGray:                   return "kCVPixelFormatType_32AlphaGray"
        case kCVPixelFormatType_16Gray:                        return "kCVPixelFormatType_16Gray"
        case kCVPixelFormatType_30RGB:                         return "kCVPixelFormatType_30RGB"
        case kCVPixelFormatType_422YpCbCr8:                    return "kCVPixelFormatType_422YpCbCr8"
        case kCVPixelFormatType_4444YpCbCrA8:                  return "kCVPixelFormatType_4444YpCbCrA8"
        case kCVPixelFormatType_4444YpCbCrA8R:                 return "kCVPixelFormatType_4444YpCbCrA8R"
        case kCVPixelFormatType_4444AYpCbCr8:                  return "kCVPixelFormatType_4444AYpCbCr8"
        case kCVPixelFormatType_4444AYpCbCr16:                 return "kCVPixelFormatType_4444AYpCbCr16"
        case kCVPixelFormatType_444YpCbCr8:                    return "kCVPixelFormatType_444YpCbCr8"
        case kCVPixelFormatType_422YpCbCr16:                   return "kCVPixelFormatType_422YpCbCr16"
        case kCVPixelFormatType_422YpCbCr10:                   return "kCVPixelFormatType_422YpCbCr10"
        case kCVPixelFormatType_444YpCbCr10:                   return "kCVPixelFormatType_444YpCbCr10"
        case kCVPixelFormatType_420YpCbCr8Planar:              return "kCVPixelFormatType_420YpCbCr8Planar"
        case kCVPixelFormatType_420YpCbCr8PlanarFullRange:     return "kCVPixelFormatType_420YpCbCr8PlanarFullRange"
        case kCVPixelFormatType_422YpCbCr_4A_8BiPlanar:        return "kCVPixelFormatType_422YpCbCr_4A_8BiPlanar"
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:  return "kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange"
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:   return "kCVPixelFormatType_420YpCbCr8BiPlanarFullRange"
        case kCVPixelFormatType_422YpCbCr8_yuvs:               return "kCVPixelFormatType_422YpCbCr8_yuvs"
        case kCVPixelFormatType_422YpCbCr8FullRange:           return "kCVPixelFormatType_422YpCbCr8FullRange"
        case kCVPixelFormatType_OneComponent8:                 return "kCVPixelFormatType_OneComponent8"
        case kCVPixelFormatType_TwoComponent8:                 return "kCVPixelFormatType_TwoComponent8"
        case kCVPixelFormatType_30RGBLEPackedWideGamut:        return "kCVPixelFormatType_30RGBLEPackedWideGamut"
        case kCVPixelFormatType_OneComponent16Half:            return "kCVPixelFormatType_OneComponent16Half"
        case kCVPixelFormatType_OneComponent32Float:           return "kCVPixelFormatType_OneComponent32Float"
        case kCVPixelFormatType_TwoComponent16Half:            return "kCVPixelFormatType_TwoComponent16Half"
        case kCVPixelFormatType_TwoComponent32Float:           return "kCVPixelFormatType_TwoComponent32Float"
        case kCVPixelFormatType_64RGBAHalf:                    return "kCVPixelFormatType_64RGBAHalf"
        case kCVPixelFormatType_128RGBAFloat:                  return "kCVPixelFormatType_128RGBAFloat"
        case kCVPixelFormatType_14Bayer_GRBG:                  return "kCVPixelFormatType_14Bayer_GRBG"
        case kCVPixelFormatType_14Bayer_RGGB:                  return "kCVPixelFormatType_14Bayer_RGGB"
        case kCVPixelFormatType_14Bayer_BGGR:                  return "kCVPixelFormatType_14Bayer_BGGR"
        case kCVPixelFormatType_14Bayer_GBRG:                  return "kCVPixelFormatType_14Bayer_GBRG"
        default: return "UNKNOWN"
        }
    }
}

import VideoToolbox

extension UIImage {
    public convenience init?(pixelBuffer: CVPixelBuffer) {
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)

        guard let cgImage = cgImage else {
            return nil
        }

        self.init(cgImage: cgImage)
    }
}

extension MLMultiArray {
    /**
     Returns a new MLMultiArray with the specified dimensions.
     - Note: This does not copy the data but uses a pointer into the original
     multi-array's memory. The caller is responsible for keeping the original
     object alive, for example using `withExtendedLifetime(originalArray) {...}`
     */
    @nonobjc public func reshaped(to dimensions: [Int]) throws -> MLMultiArray {
        let newCount = dimensions.reduce(1, *)
        precondition(newCount == count, "Cannot reshape \(shape) to \(dimensions)")
        
        var newStrides = [Int](repeating: 0, count: dimensions.count)
        newStrides[dimensions.count - 1] = 1
        for i in stride(from: dimensions.count - 1, to: 0, by: -1) {
            newStrides[i - 1] = newStrides[i] * dimensions[i]
        }
        
        let newShape_ = dimensions.map { NSNumber(value: $0) }
        let newStrides_ = newStrides.map { NSNumber(value: $0) }
        
        return try MLMultiArray(dataPointer: self.dataPointer,
                                shape: newShape_,
                                dataType: self.dataType,
                                strides: newStrides_)
    }
}

extension CVPixelBuffer {

  func normalize() {

    let width = CVPixelBufferGetWidth(self)
    let height = CVPixelBufferGetHeight(self)

    CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
    let floatBuffer = unsafeBitCast(CVPixelBufferGetBaseAddress(self), to: UnsafeMutablePointer<Float>.self)

    var minPixel: Float = 1.0
    var maxPixel: Float = 0.0

    for y in 0 ..< height {
      for x in 0 ..< width {
        let pixel = floatBuffer[y * width + x]
        minPixel = min(pixel, minPixel)
        maxPixel = max(pixel, maxPixel)
      }
    }

    let range = maxPixel - minPixel

    for y in 0 ..< height {
      for x in 0 ..< width {
        let pixel = floatBuffer[y * width + x]
        floatBuffer[y * width + x] = (pixel - minPixel) / range
      }
    }

    CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
  }
}
