#ifndef MODEL_SIMULATOR_UTIL_STREAM_H
#define MODEL_SIMULATOR_UTIL_STREAM_H


#include <cuda_runtime.h>

class Stream
{
public:
	Stream(int device_id, bool locked = false);
	~Stream();


	void Create();
	void Destroy();


	void Lock();
	void Unlock();

	bool IsLocked() const;

	int DeviceID() const;

	cudaStream_t Get() const;

private:
	cudaStream_t stream_;
	bool locked_;

	int device_id_;
};

#endif