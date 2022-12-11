//
// Created by Phil on 28.02.2022.
//

#include "common/job_meta_information.h"

#include <utility>

JobMetaInformation::JobMetaInformation(const GisaxsTransformationContainer::JobMetaInformationContainer &job_info)
        :
        client_id_(job_info.client_id),
        job_id_(job_info.job_id),
        intensity_format_(job_info.intensity_format),
        lineprofiles_(job_info.simulationTargets) {
}

const std::string &JobMetaInformation::ClientId() const {
    return std::to_string(client_id_);
}

const std::string &JobMetaInformation::JobId() const {
    return std::to_string(job_id_);
}

IntensityFormat JobMetaInformation::Format() const {
    return intensity_format_;
}

const std::vector<GisaxsTransformationContainer::SimulationTargetDefinition> &JobMetaInformation::SimulationTargets() const {
    return lineprofiles_;
}