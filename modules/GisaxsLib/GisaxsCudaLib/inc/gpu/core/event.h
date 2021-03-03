#ifndef MODEL_SIMULATOR_UTIL_EVENT_H
#define MODEL_SIMULATOR_UTIL_EVENT_H


#include <cuda_runtime.h>

class Event
{
public:
	Event(int device_id, const cudaStream_t stream, bool locked = false);
	~Event();

	void Record() const;
	void BlockStream(const cudaStream_t stream) const;
	cudaEvent_t Get() const;
	void Create();
	void Destroy();


	void Lock();
	void Unlock();

	bool IsLocked() const;

	int DeviceID() const;
	const cudaStream_t Stream() const;


private:
	cudaEvent_t event_;
	bool locked_;

	int device_id_;
	const cudaStream_t stream_;
};

#endif