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
void cpuLayerNorm(DType *out, DType *in, std::uint32_t batch, std::uint32_t length, DType eps=1e-05)
{
    std::vector<DType> mean(batch, 0);
    std::vector<DType> var(batch, 0);

    // calculate mean
    for(int i=0; i<batch; i++)
    {
        DType* inC  = in  + i * length;
        DType* outC = out + i * length;

        for(int j=0; j<length; j++)
        {
            mean[i] += inC[j];
        }
        mean[i] = mean[i] / length;

        // calculate var
        for(int j=0; j<length; j++)
        {
            var[i] += (inC[j] - mean[i]) * (inC[j] - mean[i]);
        }
        var[i] = std::sqrt(var[i] / length + eps);

        // calculate var
        for(int j=0; j<length; j++) {
            outC[j] = (inC[j] - mean[i]) / var[i];
        }
    }
}

hipError_t launchASMLayerNorm(hipFunction_t func, float *src, float *dst, std::uint32_t m, std::uint32_t n, float eps, std::size_t numRuns, bool sync = true) {
    KernelArguments args;
    args.append(src);
    args.append(dst);
    args.append(m);
    args.append(n);
    args.append(eps);
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
    const auto numWorkgroups = m;

    for (size_t i = 0; i < numRuns; ++i) {
        err = hipExtModuleLaunchKernel(func, 256, numWorkgroups, 1, 256, 1, 1, 32 * sizeof(float), nullptr, nullptr, launchArgs);
    }

    err = hipEventRecord(end);

    if (sync) {
        err = hipEventSynchronize(end);
        err = hipDeviceSynchronize();
    }

    float dur{};
    err = hipEventElapsedTime(&dur, beg, end);
    std::cout << "ASM kernel time: " << std::to_string(dur / numRuns) << " ms\n";
    std::cout << "Perf: " << numRuns * m * n * 2 * sizeof(float) * 1e3 / std::pow(1024.f, 3) / dur << " GB/s\n";
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
void dumpBuffer(const char* title, const std::vector<T>& data, int m, int n)

{
    std::cout << std::endl <<  "----- " << title << " -----" << std::endl;
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            std::cout << data[j+i*n] << " ";
            if (j%64 == 63)
                std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);
    assert(argc == 4);

    const std::string coPath(argv[1]);
    const std::uint32_t m(std::atoi(argv[2]));
    const std::uint32_t n(std::atoi(argv[3]));
    const std::uint32_t numElements = m * n;

    std::vector<float> cpuMem(numElements, 0);
    randomize(begin(cpuMem), end(cpuMem));

    float *gpuMem{};
    err = hipMalloc(&gpuMem, sizeof(float) * numElements);
    err = hipMemcpyHtoD(gpuMem, cpuMem.data(), cpuMem.size() * sizeof(float));

    float *hipResult{};
    err = hipMalloc(&hipResult, sizeof(float) * numElements);
    err = hipMemset(hipResult, 0, sizeof(float) * numElements);
    std::cout << "hipResult address: " << hipResult << std::endl;

    hipModule_t module{};
    hipFunction_t func{};
    err = prepareASMKernel("LayerNorm", coPath, &module, &func);
    if (err)
        std::cout << "find asm kernel failed" << std::endl;
    err = launchASMLayerNorm(func, gpuMem, hipResult, m, n, 1e-05, 2000);
    if (err)
        std::cout << "launchASMLayerNorm error : " << err << std::endl;

    std::vector<float> asmResult(numElements, 0.0f);
    err = hipMemcpyDtoH(asmResult.data(), hipResult, numElements * sizeof(float));
    // dumpBuffer("GPU result", asmResult, m, n);

    std::vector<float> cpuRef(numElements, 0.f);
    cpuLayerNorm<float>(cpuRef.data(), cpuMem.data(), m, n, 1e-05);
    // dumpBuffer("CPU result", cpuRef, m, n);

    float error = 0.0;
    int gpunan = 0;
    int cpunan = 0;
    int gpuinf = 0;
    int cpuinf = 0;
    for (std::size_t i = 0; i < numElements; ++i) {
        error = std::max(error, std::abs(asmResult[i]-cpuRef[i]));
        if (std::isnan(asmResult[i])) {
            gpunan += 1;
        }
        if (std::isnan(cpuRef[i])) {
            cpunan += 1;
        }
        if (std::isinf(asmResult[i])) {
            gpuinf += 1;
        }
        if (std::isinf(cpuRef[i])) {
            cpuinf += 1;
        }
    }
    if (gpunan)
        std::cout << "gpunan: " << gpunan << std::endl;
    if (cpunan)
        std::cout << "cpunan: " << cpunan << std::endl;
    if (gpuinf)
        std::cout << "gpuinf: " << gpuinf << std::endl;
    if (cpuinf)
        std::cout << "cpuinf: " << cpuinf << std::endl;

    std::cout << "Tony max error : " << error << std::endl;

    err = hipFree(gpuMem);
    err = hipModuleUnload(module);
    return 0;
}
