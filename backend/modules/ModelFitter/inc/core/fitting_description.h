#ifndef MODEL_FITTER_CORE_FITTING_DESCRIPTION_H
#define MODEL_FITTER_CORE_FITTING_DESCRIPTION_H

#include <memory>
#include "common/image_data.h"
#include "common/simulation_description.h"


class FitJob {
public:

    FitJob(json sim_data, ImageData real_image, int evolutions, int individuals, int populations);

    [[nodiscard]] const ImageData &RealImg() const;

    [[nodiscard]] size_t Evolutions() const;

    [[nodiscard]] size_t Individuals() const;

    [[nodiscard]] size_t Populations() const;

    [[nodiscard]] const json &SimulationData() const;

    [[nodiscard]] const GisaxsTransformationContainer::FlatShapeContainer &BaseShapes() const;
private:
    ImageData real_img_;
    json sim_data_;
    GisaxsTransformationContainer::FlatShapeContainer base_shapes_;

    int evolutions_;
    int individuals_;
    int populations_;
};

#endif