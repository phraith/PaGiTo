//
// Created by Phil on 06.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H
#define GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H

#include <nlohmann/json.hpp>
#include "standard_vector_types.h"
#include "common/layer.h"

using json = nlohmann::json;

namespace GisaxsTransformationContainer {
    struct DetectorContainer {
        MyType2I beamImpact;
        MyType2I resolution;
        MyType pixelsize;
        MyType sampleDistance;
    };

    struct FlatShapeContainer {
        std::vector<MyType2> parameters;
        std::vector<int> parameter_indices;
        std::vector<MyType2> upper_bounds;
        std::vector<MyType2> lower_bounds;
        std::vector<MyType3> positions;
        std::vector<int> position_indices;
        std::vector<ShapeTypeV2> shape_types;
    };

    struct SampleContainer {
        std::vector<Layer> layers;

        MyType substrate_delta;
        MyType substrate_beta;
    };

    struct BeamContainer {
        MyType alphai;
        MyType photonEv;
    };

    struct UnitcellMetaContainer {
        MyType3I repetitions;
        MyType3 translation;
    };

    FlatShapeContainer ConvertToFlatShapes(const json &json);
    void from_json(const json &j, FlatShapeContainer &shapes);
    SampleContainer ConvertToSample(const json &json);
    void from_json(const json &j, SampleContainer &sample);
    BeamContainer ConvertToBeam(const json &json);
    void from_json(const json &j, BeamContainer &beam);
    UnitcellMetaContainer ConvertToUnitcellMeta(const json &json);
    void from_json(const json &j, UnitcellMetaContainer &unitcellMeta);
    DetectorContainer ConvertToDetector(const json &json);
    void from_json(const json &j, DetectorContainer &detector);



}
#endif //GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H
