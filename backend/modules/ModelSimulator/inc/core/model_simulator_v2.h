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
#include "service.h"
#include <util/sim_connector.h>
#include <util/sim_publisher.h>

class ModelSimulatorV2 : public Service
{
public:
    ModelSimulatorV2();
	~ModelSimulatorV2() override;

    std::string ServiceName() const override;

    std::string HandleRequest(const std::string &request) const override;

	std::vector<TimeMeasurement> GetDeviceTimings() const;
private:
    SimData RunGISAXS(const SimJob &descr, const ImageData *real_img, bool copy_intensities = false) const;

    Device& LockAndReturnDevice() const;
	void UnlockDevice(Device &device) const;

	std::shared_ptr<HardwareInformation> hw_info_;

	mutable std::mutex mutex_;
	mutable std::condition_variable cv_;
};

#endif