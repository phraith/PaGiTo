#ifndef MODEL_SIMULATOR_UTIL_SIMULATION_DESCRIPTION_H
#define MODEL_SIMULATOR_UTIL_SIMULATION_DESCRIPTION_H

#include <vector>
#include <map>
#include <memory>

#include "common/fitting_parameter.h"
#include "common/image_data.h"
#include <common/experimental_model.h>
//#include "common/unitcell.h"

#include "standard_vector_types.h"
#include "unitcell_v2.h"
#include "job_meta_information.h"
#include "experimental_data.h"

class SimJob
{
public:
    SimJob(const JobMetaInformation &meta_information, const ExperimentalData &experimental_information);
    [[nodiscard]] const JobMetaInformation &JobInfo() const;
    [[nodiscard]] const ExperimentalData &ExperimentInfo() const;
//    SimJob(const std::string &uuid, std::shared_ptr<UnitcellV2> h_unitcell, std::shared_ptr<ExperimentalModel> model, bool is_last);
//    ~SimJob();
//
//    const UnitcellV2& HUnitcell() const;
//    const std::vector<MyComplex> &GetPropagationCoefficients() const;
//
//    const std::string& Uuid() const;
//    bool IsLast() const;
//    std::shared_ptr<ExperimentalModel> model_;

private:
    JobMetaInformation meta_information_;
    ExperimentalData experimental_information_;
//    std::shared_ptr<UnitcellV2> h_unitcell_;
//
//    std::string uuid_;
//    bool is_last_;
};

#endif