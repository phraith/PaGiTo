#ifndef MODEL_SIMULATOR_CORE_STREAM_PROVIDER_H
#define MODEL_SIMULATOR_CORE_STREAM_PROVIDER_H

#include <vector>
#include <memory>

#include "gpu/core/stream.h"
#include <mutex>

class StreamProvider
{
public:
	StreamProvider(int device_id);
	~StreamProvider();

	std::shared_ptr<Stream> ProvideStream();
	void UnlockAll();
	void DestroyAllStreams();

private:
	int device_id_;
	std::vector<std::shared_ptr<Stream>> streams_;

	std::mutex mutex_;
};

#endif