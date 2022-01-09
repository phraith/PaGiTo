#ifndef MODEL_SIMULATOR_CORE_MODEL_SIMULATOR_H
#define MODEL_SIMULATOR_CORE_MODEL_SIMULATOR_H

#include <vector>
#include <mutex>
#include <condition_variable>

#include "common/simulation_description.h"
#include "core/simulation_interval.h"

#include "util/hardware_information.h"
#include "util/distribution_functions.h"
#include "common/image_data.h"
#include "common/device.h"
#include <util/sim_connector.h>
#include <util/sim_publisher.h>

class ModelSimulator
{
public:
	ModelSimulator();
	ModelSimulator(const std::string& connector_port, const std::string& publisher_port);
	~ModelSimulator();

	SimData RunGISAXS(const SimJob &descr, const ImageData *real_img, bool copy_intensities = false) const;
	void Run();
	void Reset();
	void ResetTimers();

	std::vector<TimeMeasurement> GetDeviceTimings() const;
private:
	Device& LockAndReturnDevice() const;
	void UnlockDevice(Device &device) const;

	const std::vector<SimulationInterval> Distribute(std::shared_ptr<SimJob> simDescr, distribution_functions::DistributionType distType, GpuDevice& device, int blocksize) const;

	std::shared_ptr<HardwareInformation> hw_info_;

	

	mutable std::mutex mutex_;
	mutable std::condition_variable cv_;

	std::string connector_port_;
	std::string publisher_port_;

	SimConnector job_provider_;
	SimPublisher result_publisher_;

	bool quit_work_;
};

#endif