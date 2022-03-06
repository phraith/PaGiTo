#include "gpu/core/gpu_information.h"

#include <iostream>
#include <map>

#include <cuda_runtime.h>
#include <cuda.h>

#include "gpu/core/gpu_helper.h"
#include "gpu/core/gpu_device_v2.h"

namespace GpuInfo
{
	std::map<cudaDeviceAttr, int&> GetAttribMapForGpuInfo(gpu_info_t& info)
	{
		std::map<cudaDeviceAttr, int&> attrib_map =
		{
			{cudaDevAttrMaxSharedMemoryPerBlock, info.sharedMemPerBlock},
			{cudaDevAttrMaxRegistersPerBlock, info.regsPerBlock},
			{cudaDevAttrWarpSize, info.warpSize},
			{cudaDevAttrMaxThreadsPerBlock, info.maxThreadsPerBlock},
			{cudaDevAttrMaxBlockDimX, info.maxThreadsDim[0]},
			{cudaDevAttrMaxBlockDimY, info.maxThreadsDim[1]},
			{cudaDevAttrMaxBlockDimZ, info.maxThreadsDim[2]},
			{cudaDevAttrMaxGridDimX, info.maxGridSize[0]},
			{cudaDevAttrMaxGridDimY, info.maxGridSize[1]},
			{cudaDevAttrMaxGridDimZ, info.maxGridSize[2]},
			{cudaDevAttrTotalConstantMemory, info.totalConstMem},
			{cudaDevAttrComputeCapabilityMajor, info.major},
			{cudaDevAttrComputeCapabilityMinor, info.minor},
			{cudaDevAttrMultiProcessorCount, info.multiProcessorCount},
			{cudaDevAttrComputeMode, info.computeMode},
			{cudaDevAttrConcurrentKernels, info.concurrentKernels},
			{cudaDevAttrAsyncEngineCount, info.asyncEngineCount},
			{cudaDevAttrL2CacheSize, info.l2CacheSize},
			{cudaDevAttrMaxThreadsPerMultiProcessor, info.maxThreadsPerMultiProcessor},
			{cudaDevAttrMaxSharedMemoryPerMultiprocessor, info.sharedMemPerMultiprocessor},
			{cudaDevAttrMaxRegistersPerMultiprocessor, info.regsPerMultiprocessor}
		};

		return attrib_map;
	}



	std::vector<std::shared_ptr<Device>> GetGpuInfo()
	{
		int deviceCount = 0;
		gpuErrchk(cudaGetDeviceCount(&deviceCount));

		std::vector<std::shared_ptr<Device>> deviceProperties;
		for (int i = 0; i < deviceCount; ++i)
		{
            gpu_info_t info;
			
			cuDeviceGetName(info.name, 256, i);

			gpuErrchk(cudaSetDevice(i));
			gpuErrchk(cudaMemGetInfo(&info.freeGlobalMem, &info.totalGlobalMem));

			auto attrib_map = GetAttribMapForGpuInfo(info);
			for (auto it = attrib_map.begin(); it != attrib_map.end(); ++it)
				gpuErrchk(cudaDeviceGetAttribute(&it->second, it->first, i));

			deviceProperties.emplace_back(std::make_shared<GpuDeviceV2::GpuDeviceV2>( info, i ));
		}

		return deviceProperties;
	}
}