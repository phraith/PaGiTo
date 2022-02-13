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

class SimJob
{
public:
    SimJob(const std::string &uuid, std::shared_ptr<UnitcellV2> h_unitcell, std::shared_ptr<ExperimentalModel> model, bool is_last);
    ~SimJob();

    const UnitcellV2& HUnitcell() const;
    const QGrid& GetQGrid() const;
    const std::vector<MyComplex> &GetPropagationCoefficients() const;

    const std::string& Uuid() const;
    bool IsLast() const;
private:
    std::shared_ptr<UnitcellV2> h_unitcell_;
    std::shared_ptr<ExperimentalModel> model_;

    std::string uuid_;
    bool is_last_;
};

#endif