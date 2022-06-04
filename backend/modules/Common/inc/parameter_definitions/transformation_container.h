//
// Created by Phil on 06.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H
#define GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H

#include <nlohmann/json.hpp>
#include "common/layer.h"
#include "common/simulation_description.h"
#include "data_containers.h"

using json = nlohmann::json;

void from_json (const json &j, Vector3<MyType> &vector);
void to_json (json &j, const Vector3<MyType> &vector);

namespace GisaxsTransformationContainer {

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
    json UpdateShapes(const json &input, const FlatShapeContainer &shape_container);
    SimJob CreateSimJobFromRequest(const std::string &request);
    SimJob CreateSimJobFromRequest(json request);


}
#endif //GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H
