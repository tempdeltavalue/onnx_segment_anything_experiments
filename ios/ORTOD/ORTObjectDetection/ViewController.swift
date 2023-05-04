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

import UIKit
import CoreML
import Vision

extension UIImage {
    func pixelBuffer() -> CVPixelBuffer? {
        let width = self.size.width
        let height = self.size.height
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                         Int(width),
                                         Int(height),
                                         kCVPixelFormatType_32ARGB,
                                         attrs,
                                         &pixelBuffer)

        guard let resultPixelBuffer = pixelBuffer, status == kCVReturnSuccess else {
            return nil
        }

        CVPixelBufferLockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(resultPixelBuffer)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: pixelData,
                                      width: Int(width),
                                      height: Int(height),
                                      bitsPerComponent: 8,
                                      bytesPerRow: CVPixelBufferGetBytesPerRow(resultPixelBuffer),
                                      space: rgbColorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue) else {
                                        return nil
        }

        context.translateBy(x: 0, y: height)
        context.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(resultPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))

        return resultPixelBuffer
    }
}

extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        return UIGraphicsImageRenderer(size: size).image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
}

class ViewController: UIViewController {
    @IBOutlet var previewView: PreviewView!
    @IBOutlet var overlayView: OverlayView!
    @IBOutlet var bottomSheetView: UIView!
    
    private let edgeOffset: CGFloat = 2.0
    private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
    
    private var result: Result?
    private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
    private let delayBetweenInferencesMs: Double = 1000
    
    // Handle all the camera related functionality
//    private lazy var cameraCapture = CameraManager(previewView: previewView)
    
    // Handle the presenting of results on the screen
    private var inferenceViewController: InferenceViewController?
    
    // Handle all model and data preprocessing and run inference
    private var modelHandler: ModelHandler? = ModelHandler(
        modelFileInfo: (name: "mask_encoder_quant", extension: "onnx"),//modelFileInfo: (name: "ssd_mobilenet_v1", extension: "ort"),
        labelsFileInfo: (name: "labelmap", extension: "txt"))
    
    // MARK: View Controller Life Cycle

    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard modelHandler != nil else {
            fatalError("Model set up failed")
        }
        
        let tst_img = UIImage(named: "test_img3")!//.resized(to: CGSize(width: 224, height: 224))
        
        let imgPixBuff = tst_img.pixelBuffer()!
        
//        let mask_enc_result = try! self.modelHandler?.runModel(onFrame: imgPixBuff)
        
        let defaultConfig = MLModelConfiguration()

        // Create an instance of the image classifier's wrapper class.
        let imageClassifierWrapper = try? resnet_predictor_quant_16_preprocess_3(configuration: defaultConfig)

        guard let imageClassifier = imageClassifierWrapper else {
            fatalError("App failed to create an image classifier model instance.")
        }

        // Get the underlying model instance.
        let imageClassifierModel = imageClassifier.model

        let modelRequest = VNCoreMLRequest(model: try! VNCoreMLModel(for: imageClassifierModel)) { (finalisedReq, err) in

            if let results = finalisedReq.results![0] as? VNCoreMLFeatureValueObservation {
                let mask_enc_result = try! self.modelHandler?.runModel(multiArr: results.featureValue.multiArrayValue!, img_w: Float(tst_img.size.width), img_h: Float(tst_img.size.height))

            }

        }

        try? VNImageRequestHandler(cvPixelBuffer: imgPixBuff, options: [ : ]).perform([modelRequest])

        
        // Create model request, pass model and print finalisedReq

                    
         // results will be an array of VNClassificationObservation
                    

                    
        

        //cameraCapture.delegate = self
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
//        cameraCapture.checkCameraConfigurationAndStartSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
//        cameraCapture.stopSession()
    }
    
    // MARK: Storyboard Segue Handlers

    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        super.prepare(for: segue, sender: sender)
        
        if segue.identifier == "EMBED" {
            guard let tempModelHandler = modelHandler else {
                return
            }
            inferenceViewController = segue.destination as? InferenceViewController
            inferenceViewController?.wantedInputHeight = tempModelHandler.inputHeight
            inferenceViewController?.wantedInputWidth = tempModelHandler.inputWidth
            inferenceViewController?.threadCountLimit = tempModelHandler.threadCountLimit
            inferenceViewController?.currentThreadCount = tempModelHandler.threadCount
            inferenceViewController?.delegate = self
            
            guard let tempResult = result else {
                return
            }
            inferenceViewController?.inferenceTime = tempResult.processTimeMs
        }
    }
}

// MARK: InferenceViewControllerDelegate Methods

extension ViewController: InferenceViewControllerDelegate {
    func didChangeThreadCount(to count: Int32) {
        if modelHandler?.threadCount == count { return }
        modelHandler = ModelHandler(modelFileInfo: (name: "ssd_mobilenet_v1", extension: "ort"),
                                    labelsFileInfo: (name: "labelmap", extension: "txt"),
                                    threadCount: count)
    }
}

// MARK: CameraManagerDelegate Methods

extension ViewController {
    func didOutput(pixelBuffer: CVPixelBuffer) {
//        runModel(onPixelBuffer: pixelBuffer)
    }
    
    // MARK: Session Handling Alerts

    func presentCameraPermissionsDeniedAlert() {
        let alertController = UIAlertController(title: "Camera Permissions Denied",
                                                message: "Camera permissions have been denied for this app.",
                                                preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        let settingsAction = UIAlertAction(title: "Settings", style: .default) { _ in
            UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!,
                                      options: [:],
                                      completionHandler: nil)
        }
        alertController.addAction(cancelAction)
        alertController.addAction(settingsAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    func presentVideoConfigurationErrorAlert() {
        let alert = UIAlertController(title: "Camera Configuration Failed",
                                      message: "There was an error while configuring camera.",
                                      preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        
        present(alert, animated: true)
    }
    
    func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        
        let currentTimeMs = Date().timeIntervalSince1970 * 1000
        guard (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs
        else { return }
        previousInferenceTimeMs = currentTimeMs
        
        result = try! self.modelHandler?.runModel(onFrame: pixelBuffer)
        
        guard let displayResult = result else {
            return
        }
        
        // Display results by the `InferenceViewController`.
        DispatchQueue.main.async {
            let resolution = CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                                    height: CVPixelBufferGetHeight(pixelBuffer))
            self.inferenceViewController?.resolution = resolution
            
            var inferenceTime: Double = 0
            if let resultInferenceTime = self.result?.processTimeMs {
                inferenceTime = resultInferenceTime
            }
            self.inferenceViewController?.inferenceTime = inferenceTime
            self.inferenceViewController?.tableView.reloadData()
            
            // Draw bounding boxes and compute the inference score
            self.drawBoundingBoxesAndCalculate(onInferences: displayResult.inferences,
                                               withImageSize: CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                                                                     height: CVPixelBufferGetHeight(pixelBuffer)))
        }
    }
    
    func drawBoundingBoxesAndCalculate(onInferences inferences: [Inference], withImageSize imageSize: CGSize) {
        overlayView.objectOverlays = []
        overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        for inference in inferences {
            // Translate the bounding box rectangle to the current view
            var convertedRect = inference.rect.applying(
                CGAffineTransform(
                    scaleX: overlayView.bounds.size.width / imageSize.width,
                    y: overlayView.bounds.size.height / imageSize.height))
            
            if convertedRect.origin.x < 0 {
                convertedRect.origin.x = edgeOffset
            }
            
            if convertedRect.origin.y < 0 {
                convertedRect.origin.y = edgeOffset
            }
            
            if convertedRect.maxY > overlayView.bounds.maxY {
                convertedRect.size.height = overlayView.bounds.maxY - convertedRect.origin.y - edgeOffset
            }
            
            if convertedRect.maxX > overlayView.bounds.maxX {
                convertedRect.size.width = overlayView.bounds.maxX - convertedRect.origin.x - edgeOffset
            }
            
            let scoreValue = Int(inference.score * 100.0)
            let string = "\(inference.className)  (\(scoreValue)%)"
            
            let nameStringsize = string.size(usingFont: displayFont)
            
            let objectOverlay = ObjectOverlay(name: string,
                                              borderRect: convertedRect,
                                              nameStringSize: nameStringsize,
                                              color: inference.displayColor,
                                              font: displayFont)
            
            objectOverlays.append(objectOverlay)
        }
        
        // Update overlay view with detected bounding boxes and class names.
        draw(objectOverlays: objectOverlays)
    }
    
    func draw(objectOverlays: [ObjectOverlay]) {
        overlayView.objectOverlays = objectOverlays
        overlayView.setNeedsDisplay()
    }
}

extension String {
    /// This method gets size of a string with a particular font.
    func size(usingFont font: UIFont) -> CGSize {
        let attributedString = NSAttributedString(string: self, attributes: [NSAttributedString.Key.font: font])
        return attributedString.size()
    }
}
