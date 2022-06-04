#include "gpu/core/stream.h"

#include "gpu/core/gpu_helper.h"

Stream::Stream(int device_id, bool locked)
	:
	device_id_(device_id),
	locked_(locked),
	stream_(nullptr)
{
}

Stream::~Stream()
{
}

void Stream::Create()
{
	if (stream_ == nullptr)
		gpuErrchk(cudaStreamCreate(&stream_));
}

void Stream::Destroy()
{
	if (stream_ != nullptr)

		gpuErrchk(cudaStreamDestroy(stream_));
}

void Stream::Lock()
{
	locked_ = true;
}

void Stream::Unlock()
{
	locked_ = false;
}

bool Stream::IsLocked() const
{
	return locked_;
}

int Stream::DeviceID() const
{
	return device_id_;
}

cudaStream_t Stream::Get() const
{
	return stream_;
}
