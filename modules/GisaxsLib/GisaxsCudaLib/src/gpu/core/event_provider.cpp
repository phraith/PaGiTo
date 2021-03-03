#include "gpu/core/event_provider.h"
#include <iostream>

EventProvider::EventProvider(int device_id)
	:
	device_id_(device_id)
{
}

EventProvider::~EventProvider()
{
	
}

std::shared_ptr<Event> EventProvider::ProvideEvent(const cudaStream_t stream)
{
	std::lock_guard lock(mutex_);

	for (auto ev : events_)
	{
		if ((stream != ev->Stream()) || (device_id_ != ev->DeviceID()))
			continue;

		if (!ev->IsLocked())
		{
			ev->Lock();
			return ev;
		}
	}
	events_.emplace_back(std::make_shared<Event>(device_id_, stream, true));
	events_.back()->Create();
	return events_.back();
}

void EventProvider::UnlockAll()
{
	for (auto& event : events_)
	{
		if (event->IsLocked())
			event->Unlock();
	}
}

void EventProvider::DestroyAllEvents()
{
	for (auto& ev : events_)
		ev->Destroy();

	events_.clear();
}
