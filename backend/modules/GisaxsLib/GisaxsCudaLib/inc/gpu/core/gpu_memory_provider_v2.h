//
// Created by Phil on 10.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_GPU_MEMORY_PROVIDER_V2_H
#define GISAXSMODELINGFRAMEWORK_GPU_MEMORY_PROVIDER_V2_H


#include <map>
#include <queue>
#include <memory>
#include "gpu_helper.h"
#include "common/standard_defs.h"

enum class MemoryType {
    CUDA, CPU
};

class InternalGpuMemoryBlockV2 {
public:
    explicit InternalGpuMemoryBlockV2(size_t size_in_bytes, int device_id);

    InternalGpuMemoryBlockV2(InternalGpuMemoryBlockV2 &&memory_block) noexcept;

    InternalGpuMemoryBlockV2 &operator=(InternalGpuMemoryBlockV2 &&memory_block);

    void Lock() const;

    void Unlock() const;

    [[nodiscard]] bool IsLocked() const;

    [[nodiscard]] void *Get() const;

    size_t Size() const;

    bool operator<(const InternalGpuMemoryBlockV2 &r) const {
        return (size_in_bytes_ < r.size_in_bytes_);
    }

private:
    [[nodiscard]] static std::unique_ptr<void, void (*)(void *)> Allocate(size_t size_in_bytes) {
        void *ptr;
        gpuErrchk(cudaMalloc(&ptr, size_in_bytes));
        return {ptr, InternalGpuMemoryBlockV2::Delete};
    }

    void static Delete(void *in) {
        gpuErrchk(cudaFree(in))
    }

private:
    std::unique_ptr<void, void (*)(void *)> address_;
    size_t size_in_bytes_;
    int device_id_;
    mutable bool locked_;
};

template<typename T>
class MemoryBlock {
public:
    explicit MemoryBlock(void *ptr, size_t size_in_bytes);

    T *Get() const;

    [[nodiscard]] size_t Size() const;

    void InitializeHtD(const std::vector<T> &input) const;

    [[maybe_unused]] std::vector<T> CopyToHost() const;

    void InitializeDtD(T *dev_array, int size) const;


private:
    T *ptr_;
    size_t count_;
};

template<typename T>
T *MemoryBlock<T>::Get() const {
    return ptr_;
}

template<typename T>
size_t MemoryBlock<T>::Size() const {
    return count_;
}

template<typename T>
void MemoryBlock<T>::InitializeHtD(const std::vector<T> &input) const {
    if (input.size() > count_)
        throw std::out_of_range("input array is bigger than allocated device memory array");
    gpuErrchk(cudaMemcpy(ptr_, &input[0], input.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void MemoryBlock<T>::InitializeDtD(T *dev_array, int size) const {
    if (size > count_)
        throw std::out_of_range("input array is bigger than allocated device memory array");

    gpuErrchk(cudaMemcpy(dev_array, ptr_, size * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
MemoryBlock<T>::MemoryBlock(void *ptr, size_t size_in_bytes)
        :
        ptr_((T *) ptr),
        count_(size_in_bytes / sizeof(T)) {

}

template<typename T>
std::vector<T> MemoryBlock<T>::CopyToHost() const {
    std::vector<T> target(count_);
    gpuErrchk(cudaMemcpy(&target[0], ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
    return target;
}

class GpuMemoryProviderV2 {
public:
    explicit GpuMemoryProviderV2(int device_id);

    ~GpuMemoryProviderV2() = default;

    template<typename T>
    MemoryBlock<T> RequestMemory(size_t count);

    template<typename T>
    MemoryBlock<T> RequestConstantMemory(ConstantMemoryId constant_memory_id, const std::vector<T> &initial_values);

    void UnlockAll() const;

private:
    std::vector<InternalGpuMemoryBlockV2> allocated_memory_;
    std::map<ConstantMemoryId, InternalGpuMemoryBlockV2> allocated_constant_memory_;

    int device_id_;
};

template<typename T>
MemoryBlock<T>
GpuMemoryProviderV2::RequestConstantMemory(ConstantMemoryId constant_memory_id, const std::vector<T> &initial_values) {
    for (const auto &entry: allocated_constant_memory_) {
        if (entry.first == constant_memory_id) {
            return MemoryBlock<T>(entry.second.Get(), entry.second.Size() * sizeof(T));
        }
    }

    allocated_constant_memory_.insert(std::make_pair(constant_memory_id,
                                                     InternalGpuMemoryBlockV2(initial_values.size() * sizeof(T),
                                                                              device_id_)));
    MemoryBlock<T> block(allocated_constant_memory_.at(constant_memory_id).Get(), initial_values.size() * sizeof(T));
    block.InitializeHtD(initial_values);
    return block;
}

template<typename T>
MemoryBlock<T> GpuMemoryProviderV2::RequestMemory(size_t count) {
    size_t size_in_bytes = count * sizeof(T);

    for (int i = 0; i < allocated_memory_.size(); ++i) {
        auto &block = allocated_memory_[i];
        if (!block.IsLocked() && block.Size() >= size_in_bytes) {
            block.Lock();
            return MemoryBlock<T>(block.Get(), block.Size());
        }

        if (block.Size() < size_in_bytes || i == allocated_memory_.size() - 1) {
            auto insertion_pos = allocated_memory_.begin() + i;
            allocated_memory_.insert(insertion_pos, InternalGpuMemoryBlockV2(size_in_bytes, device_id_));
            allocated_memory_[i].Lock();
            return MemoryBlock<T>(allocated_memory_[i].Get(), allocated_memory_[i].Size());
        }
    }

    //Relevant only when allocated_memory has no elements
    allocated_memory_.emplace_back(InternalGpuMemoryBlockV2(size_in_bytes, device_id_));
    allocated_memory_[0].Lock();
    return MemoryBlock<T>(allocated_memory_[0].Get(), allocated_memory_[0].Size());
}


#endif //GISAXSMODELINGFRAMEWORK_GPU_MEMORY_PROVIDER_V2_H
