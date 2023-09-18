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


template<typename DType>
void cpuAMax(DType *out, DType *in, std::uint32_t length)
{
    // calculate amax
    out[0] = 0;
    for(int j=0; j<length; j++) {
        out[0] = std::max(out[0], std::abs(in[j]));
    }
}

hipError_t launchASMAMax(hipFunction_t func, float *out, float* in, std::uint32_t length, std::size_t numRuns) {

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
//    std::cout << "Perf: " << numRuns * m * n * 2 * sizeof(float) * 1e3 / std::pow(1024.f, 3) / dur << " GB/s\n";
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
        std::cout << data[j] << " ";
        if (j%64 == 63)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);
    assert(argc == 3);

    const std::string coPath(argv[1]);
    const std::uint32_t length(std::atoi(argv[2]));

    std::vector<float> cpuInput(length, 0);
    randomize(begin(cpuInput), end(cpuInput));

    float *gpuInput{};
    err = hipMalloc(&gpuInput, sizeof(float) * length);
    err = hipMemcpyHtoD(gpuInput, cpuInput.data(), cpuInput.size() * sizeof(float));

    float *gpuOutput{};
    err = hipMalloc(&gpuOutput, sizeof(float));
    err = hipMemset(gpuOutput, 0, sizeof(float));

    hipModule_t module{};
    hipFunction_t func{};
    err = prepareASMKernel("AMax", coPath, &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;
    err = launchASMAMax(func, gpuOutput, gpuInput, length, 2000);
    if (err)
        std::cout << "launchASMAMax error : " << err << std::endl;

    std::vector<float> cpuOutput(1, 0.0f);
    err = hipMemcpyDtoH(cpuOutput.data(), gpuOutput, sizeof(float));
    dumpBuffer("GPU result", cpuOutput, cpuOutput.size());

    std::vector<float> cpuRef(1, 0.f);
    cpuAMax<float>(cpuRef.data(), cpuInput.data(), length);
    dumpBuffer("CPU result", cpuRef, cpuRef.size());

    float error = 0.0;
    int gpunan = 0;
    int cpunan = 0;
    int gpuinf = 0;
    int cpuinf = 0;
    for (std::size_t i = 0; i < 1; ++i) {
        error = std::max(error, std::abs(cpuOutput[i]-cpuRef[i]));
        if (std::isnan(cpuOutput[i])) {
            gpunan += 1;
        }
        if (std::isnan(cpuRef[i])) {
            cpunan += 1;
        }
        if (std::isinf(cpuOutput[i])) {
            gpuinf += 1;
        }
        if (std::isinf(cpuRef[i])) {
            cpuinf += 1;
        }
    }

    std::cout << "----- " << "data result" << " -----" << std::endl;
    if (gpunan)
        std::cout << "gpunan: " << gpunan << std::endl;
    if (cpunan)
        std::cout << "cpunan: " << cpunan << std::endl;
    if (gpuinf)
        std::cout << "gpuinf: " << gpuinf << std::endl;
    if (cpuinf)
        std::cout << "cpuinf: " << cpuinf << std::endl;
    std::cout << "Tony max error : " << error << std::endl;

    err = hipFree(gpuOutput);
    err = hipFree(gpuInput);
    err = hipModuleUnload(module);
    return 0;
}
