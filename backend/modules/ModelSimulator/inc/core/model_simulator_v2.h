#ifndef MODEL_SIMULATOR_CORE_MODEL_SIMULATOR_H
#define MODEL_SIMULATOR_CORE_MODEL_SIMULATOR_H

#include <vector>
#include <mutex>
#include <condition_variable>

#include "common/simulation_description.h"
#include "core/simulation_interval.h"

#include "util/hardware_information.h"
#include "common/image_data.h"
#include "common/device.h"
#include "service.h"

class ModelSimulatorV2 : public Service {
public:
    ModelSimulatorV2(std::shared_ptr<HardwareInformation> hw_info);

    ~ModelSimulatorV2() override;

    std::string ServiceName() const override;

    std::vector<std::byte>
    HandleRequest(const std::string &request, std::vector<std::byte> image_data, const std::string &origin) override;

    std::vector<TimeMeasurement> GetDeviceTimings() const;

private:
    SimData RunGISAXS(const SimJob &descr, std::shared_ptr<ImageData> real_img, bool copy_intensities = false) const;


    static std::vector<std::byte>
    Serialize(const std::vector<GisaxsTransformationContainer::SimulationTargetDefinition> &target_definitions,
              const std::vector<unsigned char> &normalized_intensities,
              const std::vector<double> &intensities,
              IntensityFormat format);

    static std::vector<std::byte> SerializeResult(const SimData &sim_data, const DetectorConfiguration &detector_data,
                                                  const JobMetaInformation &job_meta_information,
                                                  bool copy_intensities);

//    Device &LockAndReturnDevice() const;

//    void UnlockDevice(Device &device) const;

    std::shared_ptr<HardwareInformation> hw_info_;


};

#endif