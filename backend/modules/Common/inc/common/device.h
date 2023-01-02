#ifndef MODEL_SIMULATOR_CORE_DEVICE_H
#define MODEL_SIMULATOR_CORE_DEVICE_H

#include <common/simulation_description.h>

#include "standard_defs.h"
#include "image_data.h"

class Device {
public:
	Device();
	~Device();

	virtual SimData RunGISAXS(const SimJob &descr, std::shared_ptr<ImageData> real_img, bool copy_intensities) = 0;
	virtual void SetStatus(WorkStatus status) const = 0;
	virtual WorkStatus Status() const = 0;

	virtual void CleanUp() = 0;
	virtual void ResetTimers() = 0;

	virtual double AverageKernelTime() const = 0;
	virtual double AverageFullTime() const = 0;

	virtual double KernelTime() const = 0;
	virtual double FullTime() const = 0;

	virtual int Runs() const = 0;

	virtual std::string Name() const = 0;

private:
};

#endif