#ifndef MODEL_SIMULATOR_UTIL_SIMULATION_DESCRIPTION_H
#define MODEL_SIMULATOR_UTIL_SIMULATION_DESCRIPTION_H

#include <vector>
#include <map>
#include <memory>

#include "job_meta_information.h"
#include "experimental_data.h"

class SimJob
{
public:
    SimJob(const JobMetaInformation &meta_information, const ExperimentalData &experimental_information);
    [[nodiscard]] const JobMetaInformation &JobInfo() const;
    [[nodiscard]] const ExperimentalData &ExperimentInfo() const;

    [[nodiscard]] std::vector<Vector2<int>> DetectorPositions() const;

private:
    JobMetaInformation meta_information_;
    ExperimentalData experimental_information_;
};

#endif