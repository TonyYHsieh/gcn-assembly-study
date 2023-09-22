#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>
#include <hip/math_functions.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include "../Utils/KernelArguments.hpp"
#include "../Utils/Math.hpp"
#include "../Utils/BufferUtils.hpp"


template<typename T>
T abs(T a)
{
    return (a > T(0)) ? a : -a;
}


template<typename T>
T max(T a, T b)
{
    return (a > b) ? a : b;
}


template<typename DType>
void cpuAMax(DType *out, DType *in, std::uint32_t length)
{
    // calculate amax
    out[0] = 0;
    for(int j=0; j<length; j++) {
        out[0] = max(out[0], abs(in[j]));
    }
}

template<typename T>
hipError_t launchASMAMax(hipFunction_t func, T *out, T* in, std::uint32_t length, std::size_t numRuns) {

    KernelArguments args;
    args.append(out);
    args.append(in);
    args.append(length);
    args.applyAlignment();
    std::size_t argsSize = args.size();
    void *launchArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        args.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argsSize,
        HIP_LAUNCH_PARAM_END
    };

    hipEvent_t beg, end;
    auto err = hipEventCreate(&beg);
    err = hipEventCreate(&end);
    err = hipEventRecord(beg);

    for (size_t i = 0; i < numRuns; ++i) {
        err = hipExtModuleLaunchKernel(func, 256, 1, 1, 256, 1, 1, 1000 * sizeof(float), nullptr, nullptr, launchArgs);
    }

    err = hipEventRecord(end);
    err = hipEventSynchronize(end);
    err = hipDeviceSynchronize();

    float dur{};
    err = hipEventElapsedTime(&dur, beg, end);
    std::cout << "ASM kernel time: " << std::to_string(dur / numRuns) << " ms\n";
    return err;
}

hipError_t prepareASMKernel(const std::string &funcName, const std::string &coPath, hipModule_t *module, hipFunction_t *func) {
    auto err = hipModuleLoad(module, coPath.c_str());
    if (err != hipSuccess)
        std::cout << "hipModuleLoad failed" << std::endl;
    err = hipModuleGetFunction(func, *module, funcName.c_str());
    if (err != hipSuccess)
        std::cout << "hipModuleGetFunction failed" << std::endl;
    return err;
}

template <typename T>
void dumpBuffer(const char* title, const std::vector<T>& data, int length)
{
    std::cout << "----- " << title << " -----" << std::endl;
    for(int j=0; j<length; j++)
    {
        std::cout << float(data[j]) << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void AMaxTest(const std::string& coPath, const std::uint32_t& length)
{
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);

    std::vector<T> cpuInput(length, 0);
    randomize(begin(cpuInput), end(cpuInput));

    T *gpuInput{};
    err = hipMalloc(&gpuInput, sizeof(T) * length);
    err = hipMemcpyHtoD(gpuInput, cpuInput.data(), cpuInput.size() * sizeof(T));

    T *gpuOutput{};
    err = hipMalloc(&gpuOutput, sizeof(T));
    err = hipMemset(gpuOutput, 0, sizeof(T));

    hipModule_t module{};
    hipFunction_t func{};
    err = prepareASMKernel("AMax", coPath, &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;
    err = launchASMAMax(func, gpuOutput, gpuInput, length, 2000);
    if (err)
        std::cout << "launchASMAMax error : " << err << std::endl;

    std::vector<T> cpuOutput(1, 0.0f);
    err = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(T));
    dumpBuffer("GPU result", cpuOutput, cpuOutput.size());

    std::vector<T> cpuRef(1, 0.f);
    cpuAMax<T>(cpuRef.data(), cpuInput.data(), length);
    dumpBuffer("CPU result", cpuRef, cpuRef.size());

    T error = 0.0;
    for (std::size_t i = 0; i < 1; ++i) {
        error = max(error, abs(cpuOutput[i]-cpuRef[i]));
    }

    std::cout << "Tony max error : " << float(error) << std::endl;

    err = hipFree(gpuOutput);
    err = hipFree(gpuInput);
    err = hipModuleUnload(module);
}


int main(int argc, char **argv) {

    assert(argc == 4);

    const std::string coPath(argv[1]);
    const std::string type(argv[2]);
    const std::uint32_t length(std::atoi(argv[3]));

    if (type == "S")
        AMaxTest<float>(coPath, length);
    else if (type == "H")
        AMaxTest<_Float16>(coPath, length);
    else
        std::cout << "unsupported type " << type << std::endl;

    return 0;
}
