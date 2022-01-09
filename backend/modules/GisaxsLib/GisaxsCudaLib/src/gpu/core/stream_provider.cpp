#include "gpu/core/stream_provider.h"
#include <iostream>

StreamProvider::StreamProvider(int device_id)
	:
	device_id_(device_id)
{
}

StreamProvider::~StreamProvider()
{
}

std::shared_ptr<Stream> StreamProvider::ProvideStream()
{
	std::lock_guard lock(mutex_);

	for (auto stream : streams_)
	{
		if (device_id_ != stream->DeviceID())
			continue;

		if (!stream->IsLocked())
		{
			stream->Lock();
			return stream;
		}
	}

	streams_.emplace_back(std::make_shared<Stream>(device_id_, true));
	streams_.back()->Create();
	return streams_.back();
}

void StreamProvider::UnlockAll()
{
	for (auto& stream : streams_)
	{
		if (stream->IsLocked())
			stream->Unlock();
	}
}

void StreamProvider::DestroyAllStreams()
{
	for (auto& stream : streams_)
		stream->Destroy();

	streams_.clear();
}