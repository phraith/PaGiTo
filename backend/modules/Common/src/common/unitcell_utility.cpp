//
// Created by Phil on 13.01.2022.
//

#include "common/unitcell_utility.h"

using json = nlohmann::json;

namespace GisaxsModeling {

//    UnitcellV2 CreateFromJson(json unitcell) {
//        std::vector<std::unique_ptr<Shape>> shapes;
//        for (const json &shape_json: unitcell.at("components")) {
//            ShapeTypeV2 shape_type = shape_json.at("type").get<ShapeTypeV2>();
//            std::vector<MyType3> locations;
//            for (const auto &location: shape_json.at("locations")) {
//                locations.emplace_back(MyType3{location[0], location[1], location[2]});
//                std::unique_ptr<Shape> shape = CreateShape(shape_type, shape_json);
//            }
//
//        }
//    }


    UnitcellV2 Convert(FlatUnitcellV2 flat_unitcell) {
        std::vector<std::unique_ptr<Shape>> shapes;

        int shape_count = flat_unitcell.ShapeTypes().size();
        for (int i = 0; i < shape_count; ++i) {
            ShapeTypeV2 shape_type = flat_unitcell.ShapeTypes()[i];
            int parameter_idx = flat_unitcell.ParameterIndices()[i];
            switch (shape_type) {
                case ShapeTypeV2::sphere: {
                    MyType2 radius = flat_unitcell.Parameters()[parameter_idx];
                    MyType2 upper_bounds = flat_unitcell.UpperBounds()[parameter_idx];
                    MyType2 lower_bounds = flat_unitcell.LowerBounds()[parameter_idx];

                    const std::vector<MyType3> &positions = flat_unitcell.Positions();

                    int positions_first_idx = flat_unitcell.PositionIndices()[i];
                    int positions_last_idx = (i < shape_count - 1) ? flat_unitcell.PositionIndices()[i + 1] :
                                             positions.size() - 1;

                    std::vector<MyType3> shape_positions = std::vector<MyType3>(positions.begin() + positions_first_idx,
                                                                                positions.begin() + positions_last_idx);

                    BoundedDistribution bounded_radius = BoundedDistribution(
                            radius.x,
                            Bounds(upper_bounds.x, lower_bounds.x),
                            radius.y,
                            Bounds(upper_bounds.y, lower_bounds.y));
                    shapes.push_back(std::make_unique<Sphere>(bounded_radius, shape_positions));
                    break;
                }
                case ShapeTypeV2::cylinder:
                    break;
            }
        }
        return UnitcellV2(std::move(shapes), flat_unitcell.Repetitons(), flat_unitcell.Translation());
    }

//    std::unique_ptr<Shape> CreateShape(ShapeTypeV2 shape_type, const json &json) {
//        std::vector<MyType3> locations;
//        for (const auto &location: json.at("locations")) {
//            locations.emplace_back(MyType3{location[0], location[1], location[2]});
//        }
//
//        switch (shape_type) {
//            case ShapeTypeV2::sphere: {
//                MyType2 radius = {json.at("radius")[0], json.at("radius")[1]};
//                return std::make_unique<Sphere>(
//                        BoundedDistribution(
//                                radius.x,
//                                Bounds(radius.x, radius.x),
//                                radius.y,
//                                Bounds(radius.y, radius.y)),
//                        locations);
//            }
//            case ShapeTypeV2::cylinder:
//                return nullptr;
//        }
//    }
}