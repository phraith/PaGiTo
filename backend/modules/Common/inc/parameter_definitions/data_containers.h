//
// Created by Phil on 26.03.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_DATA_CONTAINERS_H
#define GISAXSMODELINGFRAMEWORK_DATA_CONTAINERS_H

#include "common/standard_defs.h"
#include "common/layer.h"

namespace GisaxsTransformationContainer {

    struct LineProfileContainer {
        std::vector<MyType> intensities;
        std::vector<int> offsets;
    };

    struct SimulationTargetDefinition {
        Vector2<int> start;
        Vector2<int> end;
    };

    struct SimulationTargetData {
        std::vector<MyType> intensities;
        SimulationTargetDefinition simulation_target_definition;
    };

    struct DetectorContainer {
        Vector2<int> beamImpact;
        Vector2<int> resolution;
        MyType pixelsize;
        MyType sampleDistance;
    };

    struct FlatShapeContainer {
        std::vector <Vector2<MyType>> parameters;
        std::vector<int> parameter_indices;
        std::vector <Vector2<MyType>> upper_bounds;
        std::vector <Vector2<MyType>> lower_bounds;
        std::vector <Vector3<MyType>> positions;
        std::vector<int> position_indices;
        std::vector <ShapeTypeV2> shape_types;
    };

    struct SampleContainer {
        std::vector <Layer> layers;

        MyType substrate_delta;
        MyType substrate_beta;
    };

    struct BeamContainer {
        MyType alphai;
        MyType photonEv;
    };

    struct UnitcellMetaContainer {
        Vector3<int> repetitions;
        Vector3 <MyType> translation;
    };
}

#endif //GISAXSMODELINGFRAMEWORK_DATA_CONTAINERS_H
