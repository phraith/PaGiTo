//#include "core/fitting_description.h"
//
//#include <vector>
//
//#include "common/fitting_parameter.h"
//#include <algorithm>
//#include <stdexcept>
//#include <cassert>
//
//
//
//FitJob::FitJob(std::shared_ptr<ExperimentalModel> model, std::unique_ptr<const ImageData> real_img, std::shared_ptr<Unitcell> h_unitcell, std::string uuid, int evolutions, int individuals, int populations, bool is_last)
//	:
//	model_(model),
//	real_img_(std::move(real_img)),
//    h_unitcell_(h_unitcell),
//    uuid_(uuid),
//    evolutions_(evolutions),
//    individuals_(individuals),
//    populations_(populations),
//    is_last_(is_last)
//{
//}
//
//std::shared_ptr<ExperimentalModel> FitJob::GetModel() const
//{
//	return model_;
//}
//
//const ImageData* FitJob::RealImg() const
//{
//    return real_img_.get();
//}
//
//std::shared_ptr<Unitcell> FitJob::HUnitcell() const
//{
//    return h_unitcell_;
//}
//
//const std::string& FitJob::Uuid()
//{
//    return uuid_;
//}
//
//size_t FitJob::Evolutions() const
//{
//    return evolutions_;
//}
//
//size_t FitJob::Individuals() const
//{
//    return individuals_;
//}
//
//size_t FitJob::Populations() const
//{
//    return populations_;
//}
//
//bool FitJob::IsLast()
//{
//    return is_last_;
//}
