//
// Created by Phil on 13.01.2022.
//

#include "common/unitcell_utility.h"

using json = nlohmann::json;

namespace GisaxsModeling {

    UnitcellV2 Convert(FlatUnitcellV2 flat_unitcell) {
        std::vector<std::unique_ptr<Shape>> shapes;

        int shape_count = flat_unitcell.ShapeTypes().size();
        for (int i = 0; i < shape_count; ++i) {
            ShapeTypeV2 shape_type = flat_unitcell.ShapeTypes()[i];
            int parameter_idx = flat_unitcell.ParameterIndices()[i];
            switch (shape_type) {
                case ShapeTypeV2::sphere: {
                    Vector2<MyType> radius = flat_unitcell.Parameters()[parameter_idx];
                    Vector2<MyType> upper_bounds = flat_unitcell.UpperBounds()[parameter_idx];
                    Vector2<MyType> lower_bounds = flat_unitcell.LowerBounds()[parameter_idx];

                    const std::vector<Vector3<MyType>> &positions = flat_unitcell.Positions();

                    int positions_first_idx = flat_unitcell.PositionIndices()[i];
                    int positions_last_idx = (i < shape_count - 1) ? flat_unitcell.PositionIndices()[i + 1] :
                                             positions.size() - 1;

                    std::vector<Vector3<MyType>> shape_positions = std::vector<Vector3<MyType>>(positions.begin() + positions_first_idx,
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
        return UnitcellV2(std::move(shapes), flat_unitcell.Repetitions(), flat_unitcell.Translation());
    }
}