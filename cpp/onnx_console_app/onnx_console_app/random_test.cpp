//#include <array>
//#include <random>
//
//template<size_t numInputElements>
//std::array<float, numInputElements>* generate_random_input() {
//    std::array<float, numInputElements>* input = new std::array<float, numInputElements>();
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
//
//    for (size_t i = 0; i < numInputElements; ++i) {
//        (*input)[i] = dist(gen);
//    }
//
//    return input;
//}
//
//int main() {
//    const size_t numInputElements = 224*224*3;
//    std::array<float, numInputElements>* input = generate_random_input<numInputElements>();
//
//    // Do something with the input...
//
//    delete input; // Deallocate the memory
//
//    return 0;
//}