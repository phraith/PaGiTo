#include <spdlog/spdlog.h>
#include "parameter_definitions/transformation_container.h"

void from_json(const json &j, Vector3<MyType> &vector) {
    vector = {j.at("x"), j.at("y"), j.at("z")};
}

void to_json(json &j, const Vector3<MyType> &vector) {
    j["x"] = vector.x;
    j["y"] = vector.y;
    j["z"] = vector.z;
}

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
                    auto radiusMeanUpper = shape.at("radius").at("meanUpper");
                    auto radiusMeanLower = shape.at("radius").at("meanLower");

                    auto radiusStddevUpper = shape.at("radius").at("stddevUpper");
                    auto radiusStddevLower = shape.at("radius").at("stddevLower");

                    shapes.upper_bounds.emplace_back(Vector2<MyType>{radiusMeanUpper, radiusStddevUpper});
                    shapes.lower_bounds.emplace_back(Vector2<MyType>{radiusMeanLower, radiusStddevLower});
                    shapes.parameters.emplace_back(Vector2<MyType>{radiusMeanUpper, radiusStddevUpper});

                    std::vector<Vector3<MyType>> positions;
                    shapes.position_indices.emplace_back(shapes.positions.size());
                    for (json position: shape.at("locations")) {
                        positions.emplace_back(position.get<Vector3<MyType>>());
                    }
                    shapes.positions.insert(shapes.positions.end(), positions.begin(), positions.end());
                    break;
                }
                case ShapeTypeV2::cylinder: {
                    shapes.shape_types.emplace_back(ShapeTypeV2::cylinder);
                    shapes.parameter_indices.emplace_back(shapes.parameters.size());

                    auto radiusMeanUpper = shape.at("radius").at("meanUpper");
                    auto radiusMeanLower = shape.at("radius").at("meanLower");

                    auto radiusStddevUpper = shape.at("radius").at("stddevUpper");
                    auto radiusStddevLower = shape.at("radius").at("stddevLower");

                    shapes.upper_bounds.emplace_back(Vector2<MyType>{radiusMeanUpper, radiusStddevUpper});
                    shapes.lower_bounds.emplace_back(Vector2<MyType>{radiusMeanLower, radiusStddevLower});
                    shapes.parameters.emplace_back(Vector2<MyType>{radiusMeanUpper, radiusStddevUpper});

                    shapes.parameter_indices.emplace_back(shapes.parameters.size());
                    auto heightMeanUpper = shape.at("height").at("meanUpper");
                    auto heightMeanLower = shape.at("height").at("meanLower");

                    auto heightStddevUpper = shape.at("height").at("stddevUpper");
                    auto heightStddevLower = shape.at("height").at("stddevLower");

                    shapes.upper_bounds.emplace_back(Vector2<MyType>{heightMeanUpper, heightStddevUpper});
                    shapes.lower_bounds.emplace_back(Vector2<MyType>{heightMeanLower, heightStddevLower});
                    shapes.parameters.emplace_back(Vector2<MyType>{heightMeanUpper, heightStddevUpper});

                    std::vector<Vector3<MyType>> positions;
                    shapes.position_indices.emplace_back(shapes.positions.size());
                    for (json position: shape.at("locations")) {
                        positions.emplace_back(position.get<Vector3<MyType>>());
                    }
                    shapes.positions.insert(shapes.positions.end(), positions.begin(), positions.end());
                    break;
                }
            }
        }
        shapes.position_indices.emplace_back(shapes.positions.size());
    }

    void to_json(json &j, const FlatShapeContainer &shapes) {
        j = json::array();
        for (int i = 0; i < shapes.shape_types.size(); ++i) {
            auto shape_type = shapes.shape_types.at(i);
            int first_parameter = shapes.parameter_indices[i];

            int first_position = shapes.position_indices[i];
            int last_position = shapes.position_indices[i + 1];

            json shape;
            shape["type"] = shape_type;

            std::vector<Vector3<MyType>> positions;
            for (int j = first_position; j < last_position; ++j) {
                positions.emplace_back(shapes.positions[j]);
            }
            shape["locations"] = positions;
            switch (shape_type) {
                case ShapeTypeV2::sphere: {
                    auto radius = shapes.parameters[first_parameter];
                    shape["radius"]["meanUpper"] = radius.x;
                    shape["radius"]["meanLower"] = radius.x;
                    shape["radius"]["stddevUpper"] = radius.y;
                    shape["radius"]["stddevLower"] = radius.y;
                    j.push_back(shape);
                    break;
                }
                case ShapeTypeV2::cylinder: {
                    auto radius = shapes.parameters[first_parameter];
                    auto height = shapes.parameters[first_parameter + 1];

                    shape["radius"]["meanUpper"] = radius.x;
                    shape["radius"]["meanLower"] = radius.x;
                    shape["radius"]["stddevUpper"] = radius.y;
                    shape["radius"]["stddevLower"] = radius.y;

                    shape["height"]["meanUpper"] = height.x;
                    shape["height"]["meanLower"] = height.x;
                    shape["height"]["stddevUpper"] = height.y;
                    shape["height"]["stddevLower"] = height.y;
                    j.push_back(shape);
                    break;
                }
            }
        }
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

    void from_json(const json &j, SimulationTargetDefinition &lineprofile) {
        j.at("start").at("x").get_to(lineprofile.start.x);
        j.at("start").at("y").get_to(lineprofile.start.y);

        j.at("end").at("x").get_to(lineprofile.end.x);
        j.at("end").at("y").get_to(lineprofile.end.y);
    }

    void from_json(const json &j, JobMetaInformationContainer &job_meta_information) {
        auto intensityFormat = j.at("intensityFormat").get<IntensityFormat>();
        job_meta_information.intensity_format = intensityFormat;

        auto simulationTargets = j.at("simulationTargets");
        for (const json &item: simulationTargets) {
            job_meta_information.simulationTargets.emplace_back(ConvertToSimulationTargetDefinition(item));
        }

//        j.at("clientId").get_to(job_meta_information.client_id);
//        j.at("jobId").get_to(job_meta_information.job_id);
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

    SimulationTargetDefinition ConvertToSimulationTargetDefinition(const json &json) {
        return json.get<SimulationTargetDefinition>();
    }

    JobMetaInformationContainer ConvertToJobMetaInformation(const json &json) {
        return json.get<JobMetaInformationContainer>();
    }

    SimJob CreateSimJobFromRequest(const std::string &request) {
        json data = json::parse(request);
        return CreateSimJobFromRequest(data);
    }

    json UpdateShapes(const json &input, const FlatShapeContainer &shape_container) {
        json j(input);
        j["config"]["shapes"] = shape_container;
        return j;
    }

    SimJob CreateSimJobFromRequest(json request) {
        json detector = request.at("config").at("instrumentation").at("detector");
        json shapes = request.at("config").at("shapes");

        json sample = request.at("config").at("sample");
        json beam = request.at("config").at("instrumentation").at("beam");
        json unitcellMeta = request.at("config").at("unitcellMeta");

        json job_info = request.at("properties");
        auto job_info_container = ConvertToJobMetaInformation(job_info);

        auto detector_container = ConvertToDetector(detector);
        auto shapes_container = ConvertToFlatShapes(shapes);
        auto sample_container = ConvertToSample(sample);
        auto beam_container = ConvertToBeam(beam);
        auto unitcell_meta_container = ConvertToUnitcellMeta(unitcellMeta);
        auto flat_unitcell = FlatUnitcellV2(shapes_container, unitcell_meta_container.repetitions,
                                            unitcell_meta_container.translation);

        DetectorConfiguration detector_config(detector_container);
        BeamConfiguration beam_config(beam_container.alphai, detector_config.Directbeam(), beam_container.photonEv,
                                      0.1);
        SampleConfiguration sample_config(
                Layer(sample_container.substrate_delta, sample_container.substrate_beta, -1, 0),
                sample_container.layers);

        return {JobMetaInformation{job_info_container},
                ExperimentalData{detector_config, beam_config, sample_config, flat_unitcell}};
    }
}