#ifndef MODEL_SIMULATOR_UTIL_GPU_MEMORY_BLOCK_H
#define MODEL_SIMULATOR_UTIL_GPU_MEMORY_BLOCK_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <any>

#include "gpu/core/gpu_helper.h"

template<class T>
class GpuMemoryBlock
{
public:
	GpuMemoryBlock(int size, int device_id, bool locked = false, std::string uuid = "")
		:
		size_(size),
		device_id_(device_id),
		locked_(locked),
		ptr_(nullptr),
		uuid_(uuid)
	{
	}

	~GpuMemoryBlock()
	{
	}

	void Allocate()
	{
		if (ptr_ == nullptr)
			gpuErrchk(cudaMalloc(&ptr_, size_ * sizeof(T)));
	}

	void Delete()
	{
		if (ptr_ != nullptr)
			gpuErrchk(cudaFree(ptr_));
	}

	T* Get() const
	{
		return ptr_;
	}

	void Lock()
	{
		locked_ = true;
	}

	void Unlock()
	{
		locked_ = false;
	}

	bool IsLocked() const
	{
		return locked_;
	}

	int Size() const
	{
		return size_;
	}

	void InitializeHtD(const std::vector<T>& input)
	{
		if (input.size() > Size())
			throw std::out_of_range("input array is bigger than allocated device memory array");

		gpuErrchk(cudaMemcpy(ptr_, &input[0], input.size() * sizeof(T), cudaMemcpyHostToDevice));
	}

	void InitializeDtD(const T* dev_array, int size)
	{
		if (size > Size())
			throw std::out_of_range("input array is bigger than allocated device memory array");

		gpuErrchk(cudaMemcpy(ptr_, dev_array, size * sizeof(T), cudaMemcpyHostToDevice));
	}

	const std::string& Uuid()
	{
		return uuid_;
	}

private:
	int size_;
	int device_id_;
	bool locked_;

	T* ptr_;

	std::string uuid_;
};

#endif