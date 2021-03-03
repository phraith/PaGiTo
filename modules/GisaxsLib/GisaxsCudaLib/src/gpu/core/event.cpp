#include "gpu/core/event.h"

#include "gpu/core/gpu_helper.h"

Event::Event(int device_id, const cudaStream_t stream, bool locked)
	:
	locked_(locked),
	device_id_(device_id),
	stream_(stream),
	event_(nullptr)
{
}

Event::~Event()
{
}

void Event::Record() const
{
	gpuErrchk(cudaEventRecord(event_, stream_));
}

void Event::BlockStream(const cudaStream_t stream) const
{
	gpuErrchk(cudaStreamWaitEvent(stream, event_, 0));
}

cudaEvent_t Event::Get() const
{
	return event_;
}

void Event::Create()
{
	// if (event_ == nullptr)
	// 	gpuErrchk(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));

	if (event_ == nullptr)
		gpuErrchk(cudaEventCreate(&event_));
}

void Event::Destroy()
{
	if (event_ != nullptr)
		gpuErrchk(cudaEventDestroy(event_));
}

void Event::Lock()
{
	locked_ = true;
}

void Event::Unlock()
{
	locked_ = false;
}

bool Event::IsLocked() const
{
	return locked_;
}

int Event::DeviceID() const
{
	return device_id_;
}

const cudaStream_t Event::Stream() const
{
	return stream_;
}