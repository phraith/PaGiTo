//
// Created by Phil on 06.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H
#define GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H

#include <nlohmann/json.hpp>
#include "standard_vector_types.h"

using json = nlohmann::json;

struct DetectorSetupContainer {
    MyType2I beamImpact;
    MyType2I resolution;
    MyType pixelsize;
    MyType sampleDistance;
};

void from_json(const json &j, DetectorSetupContainer &detector) {
    j.at("beamImpact").at("x").get_to(detector.beamImpact.x);
    j.at("beamImpact").at("y").get_to(detector.beamImpact.y);
    j.at("resolution").at("width").get_to(detector.resolution.x);
    j.at("resolution").at("height").get_to(detector.resolution.y);
    j.at("sampleDistance").get_to(detector.sampleDistance);
    j.at("pixelsize").get_to(detector.pixelsize);
}

struct ShapeContainer {
    std::vector<std::unique_ptr<Shape>> shapes;
};

NLOHMANN_JSON_SERIALIZE_ENUM(ShapeTypeV2, {
    { ShapeTypeV2::sphere, "sphere" },
    { ShapeTypeV2::cylinder, "cylinder" }
})

void from_json(const json &j, ShapeContainer &shapes) {
    for (json shape: j) {
        auto shapeType = shape.at("type").get<ShapeTypeV2>();
        switch (shapeType) {
            case ShapeTypeV2::sphere: {
                auto radiusMean = shape.at("radius").at("mean");
                auto radiusStddev = shape.at("radius").at("stddev");
                BoundedDistribution radius(radiusMean, Bounds(radiusMean, radiusMean), radiusStddev,
                                           Bounds(radiusStddev, radiusStddev));

                std::vector<MyType3> positions;
                for (json position: shape.at("locations")) {
                    positions.emplace_back(MyType3{position.at("x"), position.at("y"), position.at("z")});
                }

                shapes.shapes.emplace_back(std::make_unique<Sphere>(radius, positions));
                break;
            }
            case ShapeTypeV2::cylinder: {
                auto radiusMean = shape.at("radius").at("mean");
                auto radiusStddev = shape.at("radius").at("stddev");
                BoundedDistribution radius(radiusMean, Bounds(radiusMean, radiusMean), radiusStddev,
                                           Bounds(radiusStddev, radiusStddev));

                auto heightMean = shape.at("radius").at("mean");
                auto heightStddev = shape.at("radius").at("stddev");
                BoundedDistribution height(heightMean, Bounds(heightMean, heightMean), heightStddev,
                                           Bounds(heightStddev, heightStddev));

                std::vector<MyType3> positions;
                for (json position: shape.at("locations")) {
                    positions.emplace_back(MyType3{position.at("x"), position.at("y"), position.at("z")});
                }

                shapes.shapes.emplace_back(std::make_unique<Cylinder>(radius, height, positions));
                break;
            }
        }
    }
}

struct SampleContainer {
    std::vector<Layer> layers;

     MyType substrate_delta;
     MyType substrate_beta;
};

void from_json(const json &j, SampleContainer &sample) {
    auto substrate = j.at("substrate");
    sample.substrate_beta = substrate.at("refraction").at("beta");
    sample.substrate_delta = substrate.at("refraction").at("delta");

    if (j.contains("layers"))
    {
        for (json layer: j.at("layers")) {
            sample.layers.emplace_back(
                    Layer(layer.at("refraction").at("delta"), layer.at("refraction").at("beta"), layer.at("order"), layer.at("thickness")));
        }
    }
}

struct BeamContainer {
    MyType alphai;
    MyType photonEv;
};

void from_json(const json &j, BeamContainer &beam) {
    j.at("alphai").get_to(beam.alphai);
    j.at("photonEv").get_to(beam.photonEv);
}

struct UnitcellMetaContainer {
    MyType3I repetitions;
    MyType3 translation;
};

void from_json(const json &j, UnitcellMetaContainer &unitcellMeta) {
    j.at("repetitions").at("x").get_to(unitcellMeta.repetitions.x);
    j.at("repetitions").at("y").get_to(unitcellMeta.repetitions.y);
    j.at("repetitions").at("z").get_to(unitcellMeta.repetitions.z);

    j.at("translation").at("x").get_to(unitcellMeta.translation.x);
    j.at("translation").at("y").get_to(unitcellMeta.translation.y);
    j.at("translation").at("z").get_to(unitcellMeta.translation.z);
}

#endif //GISAXSMODELINGFRAMEWORK_TRANSFORMATION_CONTAINER_H
