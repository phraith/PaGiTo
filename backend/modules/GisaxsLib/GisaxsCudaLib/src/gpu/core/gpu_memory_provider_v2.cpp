//
// Created by Phil on 10.02.2022.
//

#include <cuda_runtime.h>
#include <memory>
#include "gpu/core/gpu_memory_provider_v2.h"

InternalGpuMemoryBlockV2::InternalGpuMemoryBlockV2(size_t size_in_bytes, int device_id)
        :
        size_in_bytes_(size_in_bytes),
        address_(InternalGpuMemoryBlockV2::Allocate(size_in_bytes)),
        device_id_(device_id),
        locked_(false)
{

}

void InternalGpuMemoryBlockV2::Lock() const{
    if (locked_)
    {
        throw std::invalid_argument("cant lock memory block twice!");}
    locked_ = true;
}

void InternalGpuMemoryBlockV2::Unlock() const{
    locked_ = false;
}

bool InternalGpuMemoryBlockV2::IsLocked() const {
    return locked_;
}

void *InternalGpuMemoryBlockV2::Get() const {
    return address_.get();
}

size_t InternalGpuMemoryBlockV2::Size() const {
    return size_in_bytes_;
}

InternalGpuMemoryBlockV2::InternalGpuMemoryBlockV2(InternalGpuMemoryBlockV2 &&memory_block) noexcept
:
        size_in_bytes_(memory_block.size_in_bytes_),
        address_(std::move(memory_block.address_)),
        device_id_(memory_block.device_id_),
        locked_(memory_block.locked_)

{}

InternalGpuMemoryBlockV2 &InternalGpuMemoryBlockV2::operator=(InternalGpuMemoryBlockV2 &&memory_block) {
    address_ = std::move(memory_block.address_);
    size_in_bytes_ = memory_block.size_in_bytes_;
    locked_ = memory_block.locked_;
    device_id_ = memory_block.device_id_;
    return *this;
}

GpuMemoryProviderV2::GpuMemoryProviderV2(int device_id)
        :
        device_id_(device_id)
{

}

void GpuMemoryProviderV2::UnlockAll() const {
    for (auto & block : allocated_memory_) {
        block.Unlock();
    }

}
