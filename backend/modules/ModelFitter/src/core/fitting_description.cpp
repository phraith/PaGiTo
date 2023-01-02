#include "core/fitting_description.h"

FitJob::FitJob(json sim_data, ImageData real_image, int evolutions, int individuals, int populations)
        :
        sim_data_(sim_data),
        base_shapes_(GisaxsTransformationContainer::ConvertToFlatShapes(sim_data.at("config").at("shapes"))),
        real_img_(real_image),
        evolutions_(evolutions),
        individuals_(individuals),
        populations_(populations) {}

const ImageData &FitJob::RealImg() const {
    return real_img_;
}

size_t FitJob::Evolutions() const {
    return evolutions_;
}

size_t FitJob::Individuals() const {
    return individuals_;
}

size_t FitJob::Populations() const {
    return populations_;
}

const json &FitJob::SimulationData() const {
    return sim_data_;
}

const GisaxsTransformationContainer::FlatShapeContainer &FitJob::BaseShapes() const {
    return base_shapes_;
}
