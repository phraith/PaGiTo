#include "common/simulation_description.h"

SimJob::SimJob(const JobMetaInformation &meta_information, const ExperimentalData &experimental_information)
:
        meta_information_(meta_information),
        experimental_information_(experimental_information)
        {}

const JobMetaInformation &SimJob::JobInfo() const {
    return meta_information_;
}

const ExperimentalData &SimJob::ExperimentInfo() const {
    return experimental_information_;
}
