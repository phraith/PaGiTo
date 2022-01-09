#ifndef MODEL_SIMULATOR_CORE_EVENT_PROVIDER_H
#define MODEL_SIMULATOR_CORE_EVENT_PROVIDER_H

#include <vector>
#include <memory>
#include <mutex>

#include "gpu/core/event.h"

class EventProvider
{
public:
	EventProvider(int device_id);
	~EventProvider();

	std::shared_ptr<Event> ProvideEvent(const cudaStream_t stream);
	void UnlockAll();
	void DestroyAllEvents();

private:
	int device_id_;
	std::vector<std::shared_ptr<Event>> events_;

	std::mutex mutex_;
};

#endif