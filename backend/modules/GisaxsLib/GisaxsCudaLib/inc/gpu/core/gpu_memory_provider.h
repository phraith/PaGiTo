#ifndef MODEL_SIMULATOR_CORE_GPU_MEMORY_PROVIDER_H
#define MODEL_SIMULATOR_CORE_GPU_MEMORY_PROVIDER_H

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <mutex>

#include "vector_types.h"

#include "gpu/core/event.h"
#include "gpu/core/gpu_memory_block.h"
#include <any>

template<typename AllocType>
class GpuMemoryProvider
{
public:
	GpuMemoryProvider(int device_id)
		:
		device_id_(device_id)
	{}

	~GpuMemoryProvider()
	{}

	std::shared_ptr<GpuMemoryBlock<AllocType>> ProvideMemory(int size)
	{
		std::lock_guard lock(mutex_);

		for (auto& block : memory_blocks_)
		{
			if (block->IsLocked())
				continue;

			if (block->Size() >= size)
			{
				block->Lock();
				return block;
			}
		}

		memory_blocks_.emplace_back(std::make_shared<GpuMemoryBlock<AllocType>>(size, device_id_, true));
		memory_blocks_.back()->Allocate();
		return memory_blocks_.back();
	}
	
	std::shared_ptr<const GpuMemoryBlock<AllocType>> ProvideConstantMemory(const std::string& uuid, const std::vector<AllocType>& values)
	{
		std::lock_guard lock(mutex_);

		for (auto block : constant_memory_blocks_)
		{
			if (block->Uuid() == uuid)
			{
				return block;
			}
		}

		constant_memory_blocks_.emplace_back(std::make_shared<GpuMemoryBlock<AllocType>>(values.size(), device_id_, false, uuid));
		constant_memory_blocks_.back()->Allocate();
		constant_memory_blocks_.back()->InitializeHtD(values);
		return constant_memory_blocks_.back();
	}


	void UnlockAll()
	{
		for (auto& block : memory_blocks_)
		{
			if (block->IsLocked())
				block->Unlock();
		}
	}

	void FreeAllMemory()
	{
		for (auto& block : memory_blocks_)
		{
			if (block->Get() != nullptr)
				block->Delete();
		}

		memory_blocks_.clear();

		for (auto& block : constant_memory_blocks_)
		{
			if (block->Get() != nullptr)
				block->Delete();
		}

		constant_memory_blocks_.clear();
	}

private:
	int device_id_;
	std::vector<std::shared_ptr<GpuMemoryBlock<AllocType>>> memory_blocks_;
	std::vector<std::shared_ptr<GpuMemoryBlock<AllocType>>> constant_memory_blocks_;


	std::mutex mutex_;
};

#endif