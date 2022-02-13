#include "common/simulation_description.h"
#include <algorithm>
#include <iostream>
#include "assert.h"

SimJob::SimJob(const std::string& uuid, std::shared_ptr<UnitcellV2> unitcell, std::shared_ptr<ExperimentalModel> model, bool is_last)
    :
    uuid_(uuid),
    h_unitcell_(unitcell),
    model_(model),
    is_last_(is_last)
{}

SimJob::~SimJob()
{}

const UnitcellV2& SimJob::HUnitcell() const
{
    assert(h_unitcell_ != nullptr);
    return *h_unitcell_;
}

const QGrid& SimJob::GetQGrid() const
{
    return model_->GetQGrid();
}

const std::vector<MyComplex> &SimJob::GetPropagationCoefficients() const
{
    return model_->GetPropagationCoefficients();
}

const std::string& SimJob::Uuid() const
{
    return uuid_;
}

bool SimJob::IsLast() const
{
    return is_last_;
}
