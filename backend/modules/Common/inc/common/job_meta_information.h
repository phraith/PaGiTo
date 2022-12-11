//
// Created by Phil on 28.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_JOB_META_INFORMATION_H
#define GISAXSMODELINGFRAMEWORK_JOB_META_INFORMATION_H


#include <string>
#include "parameter_definitions/data_containers.h"

class JobMetaInformation {
public:
    explicit JobMetaInformation(const GisaxsTransformationContainer::JobMetaInformationContainer &job_info);

    [[nodiscard]] const std::string &ClientId() const;
    [[nodiscard]] const std::string &JobId() const;
    [[nodiscard]] IntensityFormat Format() const;
    [[nodiscard]] const std::vector<GisaxsTransformationContainer::SimulationTargetDefinition> &SimulationTargets() const;

private:
    long client_id_;
    long job_id_;
    IntensityFormat intensity_format_;
    std::vector<GisaxsTransformationContainer::SimulationTargetDefinition> lineprofiles_;
};


#endif //GISAXSMODELINGFRAMEWORK_JOB_META_INFORMATION_H
