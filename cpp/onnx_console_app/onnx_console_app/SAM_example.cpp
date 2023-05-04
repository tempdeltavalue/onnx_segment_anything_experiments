#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <random>
#include <map>


#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include "Helpers.cpp"

template<size_t numInputElements>
std::array<float, numInputElements>* generate_random_input(bool set_zero=false) {
    std::array<float, numInputElements>* input = new std::array<float, numInputElements>();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1, 2.0f);

    for (size_t i = 0; i < numInputElements; ++i) {
        if (set_zero) {
            (*input)[i] = 1;
        }
        else {
            (*input)[i] = dist(gen);
        }
    }

    return input;
}


int main()
{
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);

    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;


    auto modelPath = L"C:\\Users\\mykha\\source\\repos\\onnx_console_app\\onnx_console_app\\assets\\sam_onnx_quantized_example.onnx";
  

    // Use CUDA GPU
    Ort::SessionOptions ort_session_options;

    OrtCUDAProviderOptions options;
    options.device_id = 0;
    //options.arena_extend_strategy = 0;
    //options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
    //options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    //options.do_copy_in_default_stream = 1;

    //OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);

    // create session
    session = Ort::Session(env, modelPath, ort_session_options);

    // Use CPU
    //session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });

    // define shape
    const std::array<int64_t, 4> inputShape = { 1, 256, 64, 64 };
    // define array
    std::array<float, 256 * 64 * 64>* input = generate_random_input<256 * 64 * 64>();

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, 
        input->data(), 
        input->size(), 
        inputShape.data(), 
        inputShape.size());


    // define array
    const std::array<int64_t, 3> inputShape2 = { 1, 1, 2 };

    std::array<float, 2> input2 = {10, 10};

    auto memory_info2 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor2 = Ort::Value::CreateTensor<float>(memory_info,
        input2.data(),
        input2.size(),
        inputShape2.data(),
        inputShape2.size());


    const std::array<int64_t, 2> inputShape3 = { 1, 1 };

    std::array<float, 1> input3 = {1};

    auto inputTensor3 = Ort::Value::CreateTensor<float>(memory_info,
        input3.data(),
        input3.size(),
        inputShape3.data(),
        inputShape3.size());


    const std::array<int64_t, 4> inputShape4 = { 1, 1, 256, 256 };

    std::array<float, 256 * 256>* input4 = generate_random_input<256 * 256>(true);

    auto inputTensor4 = Ort::Value::CreateTensor<float>(memory_info,
        input4->data(),
        input4->size(),
        inputShape4.data(),
        inputShape4.size());


    const std::array<int64_t, 1> inputShape5 = { 1 };

    std::array<float, 1> input5 = {1};

    auto inputTensor5 = Ort::Value::CreateTensor<float>(memory_info,
        input5.data(),
        input5.size(),
        inputShape5.data(),
        inputShape5.size());


    const std::array<int64_t, 1> inputShape6 = { 2 };

    std::array<float, 512 * 512>* input6 = generate_random_input<512 * 512>();

    auto inputTensor6 = Ort::Value::CreateTensor<float>(memory_info,
        input6->data(),
        input6->size(),
        inputShape6.data(),
        inputShape6.size());

    // copy image data to input array
    //std::copy(imageVec.begin(), imageVec.end(), input.begin());



    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;

    //Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);

    const std::array<const char*, 6> inputNames = { "image_embeddings", 
                                                    "point_coords", 
                                                    "point_labels", 
                                                    "mask_input", 
                                                    "has_mask_input",
                                                    "orig_im_size"};

    const std::array<const char*, 3> outputNames = { "masks", "iou_predictions", "low_res_masks"};
    //outputName.release();

    std::array<float, 224 * 224> results;
    const std::array<int64_t, 4> outputShape = { 1, 1, 224, 224 };
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());


    //std::array<float, 224 * 224> *results2 = generate_random_input<224 * 224>(true);
    //const std::array<int64_t, 4> outputShape2 = {1, 1, 224, 244 };
    //auto outputTensor2 = Ort::Value::CreateTensor<float>(memory_info, results2->data(), results2->size(), outputShape2.data(), outputShape2.size());
    
    
    //
    //std::array<float, 256 * 256> results3;
    //const std::array<int64_t, 4> outputShape3 = { 1, 1, 256, 256 };
    //auto outputTensor3 = Ort::Value::CreateTensor<float>(memory_info, results3.data(), results3.size(), outputShape3.data(), outputShape3.size());

    //std::vector<Ort::Value> ort_outputs;
    //ort_outputs.push_back(std::move(outputTensor2));

    // run inference
    try {
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(inputTensor));
        ort_inputs.push_back(std::move(inputTensor2));
        ort_inputs.push_back(std::move(inputTensor3));
        ort_inputs.push_back(std::move(inputTensor4));
        ort_inputs.push_back(std::move(inputTensor5));
        ort_inputs.push_back(std::move(inputTensor6));

        


        //Ort::Value* input_tensors[2];
        //input_tensors[0] = &inputTensor;
        //input_tensors[1] = &inputTensor2;
        auto ort_outputs = session.Run(runOptions, 
            inputNames.data(), 
            ort_inputs.data(), 
            ort_inputs.size(), 
            outputNames.data(), 
            3);

        auto info = ort_outputs[0].GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();

        float* pred = ort_outputs[0].GetTensorMutableData<float>();

        cv::Mat img = cv::Mat::zeros(256, 256, CV_32FC1);
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img.at<float>(i, j) = pred[i * 256 + j];
            }
        }


        imshow("Array Image", img);
        cv::waitKey(0);


    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }
}