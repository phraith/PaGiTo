#include "parameter_definitions/transformation_container.h"

namespace GisaxsTransformationContainer {
    void
    from_json(const json &j, DetectorContainer &detector) {
        j.at("beamImpact").at("x").get_to(detector.beamImpact.x);
        j.at("beamImpact").at("y").get_to(detector.beamImpact.y);
        j.at("resolution").at("width").get_to(detector.resolution.x);
        j.at("resolution").at("height").get_to(detector.resolution.y);
        j.at("sampleDistance").get_to(detector.sampleDistance);
        j.at("pixelsize").get_to(detector.pixelsize);
    }

    void from_json(const json &j, FlatShapeContainer &shapes) {
        for (json shape: j) {
            auto shapeType = shape.at("type").get<ShapeTypeV2>();
            switch (shapeType) {
                case ShapeTypeV2::sphere: {
                    shapes.shape_types.emplace_back(ShapeTypeV2::sphere);
                    shapes.parameter_indices.emplace_back(shapes.parameters.size());
                    auto radiusMean = shape.at("radius").at("mean");
                    auto radiusStddev = shape.at("radius").at("stddev");
                    shapes.parameters.emplace_back(Vector2<MyType>{radiusMean, radiusStddev});

                    std::vector<Vector3<MyType>> positions;
                    shapes.position_indices.emplace_back(shapes.positions.size());
                    for (json position: shape.at("locations")) {
                        positions.emplace_back(Vector3<MyType>{position.at("x"), position.at("y"), position.at("z")});
                    }
                    shapes.positions.insert(shapes.positions.end(), positions.begin(), positions.end());
                    break;
                }
                case ShapeTypeV2::cylinder: {
                    shapes.shape_types.emplace_back(ShapeTypeV2::cylinder);
                    shapes.parameter_indices.emplace_back(shapes.parameters.size());
                    auto radiusMean = shape.at("radius").at("mean");
                    auto radiusStddev = shape.at("radius").at("stddev");
                    shapes.parameters.emplace_back(Vector2<MyType>{radiusMean, radiusStddev});

                    shapes.parameter_indices.emplace_back(shapes.parameters.size());
                    auto heightMean = shape.at("height").at("mean");
                    auto heightStddev = shape.at("height").at("stddev");
                    shapes.parameters.emplace_back(Vector2<MyType>{heightMean, heightStddev});

                    std::vector<Vector3<MyType>> positions;
                    shapes.position_indices.emplace_back(shapes.positions.size());
                    for (json position: shape.at("locations")) {
                        positions.emplace_back(Vector3<MyType>{position.at("x"), position.at("y"), position.at("z")});
                    }
                    shapes.positions.insert(shapes.positions.end(), positions.begin(), positions.end());
                    break;
                }
            }
        }
        shapes.position_indices.emplace_back(shapes.positions.size());
    }

    void from_json(const json &j, SampleContainer &sample) {
        auto substrate = j.at("substrate");
        sample.substrate_beta = substrate.at("refraction").at("beta");
        sample.substrate_delta = substrate.at("refraction").at("delta");

        if (j.contains("layers")) {
            for (json layer: j.at("layers")) {
                sample.layers.emplace_back(
                        Layer(layer.at("refraction").at("delta"), layer.at("refraction").at("beta"), layer.at("order"),
                              layer.at("thickness")));
            }
        }
    }

    void from_json(const json &j, UnitcellMetaContainer &unitcellMeta) {
        j.at("repetitions").at("x").get_to(unitcellMeta.repetitions.x);
        j.at("repetitions").at("y").get_to(unitcellMeta.repetitions.y);
        j.at("repetitions").at("z").get_to(unitcellMeta.repetitions.z);

        j.at("translation").at("x").get_to(unitcellMeta.translation.x);
        j.at("translation").at("y").get_to(unitcellMeta.translation.y);
        j.at("translation").at("z").get_to(unitcellMeta.translation.z);
    }

    void from_json(const json &j, BeamContainer &beam) {
        j.at("alphai").get_to(beam.alphai);
        j.at("photonEv").get_to(beam.photonEv);
    }

    FlatShapeContainer ConvertToFlatShapes(const json &json) {
        return json.get<FlatShapeContainer>();
    }

    SampleContainer ConvertToSample(const json &json) {
        return json.get<SampleContainer>();
    }

    BeamContainer ConvertToBeam(const json &json) {
        return json.get<BeamContainer>();
    }

    UnitcellMetaContainer ConvertToUnitcellMeta(const json &json) {
        return json.get<UnitcellMetaContainer>();
    }

    DetectorContainer ConvertToDetector(const json &json) {
        return json.get<DetectorContainer>();
    }
}