//
// Created by Phil on 13.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_FLAT_UNITCELL_H
#define GISAXSMODELINGFRAMEWORK_FLAT_UNITCELL_H

#include <vector>
#include "standard_defs.h"
#include "parameter_definitions/transformation_container.h"


class FlatUnitcellV2 {
public:
    FlatUnitcellV2(GisaxsTransformationContainer::FlatShapeContainer shape_container, Vector3<int> repetitions, Vector3<MyType> translation);

    [[nodiscard]] const std::vector<Vector2<MyType>> &Parameters() const;

    [[nodiscard]] const std::vector<int> &ParameterIndices() const;

    [[nodiscard]] const std::vector<Vector2<MyType>> &UpperBounds() const;

    [[nodiscard]] const std::vector<Vector2<MyType>> &LowerBounds() const;

    [[nodiscard]] const std::vector<ShapeTypeV2> &ShapeTypes() const;

    [[nodiscard]] const std::vector<Vector3<MyType>> &Positions() const;

    [[nodiscard]] const std::vector<int> &PositionIndices() const;

    [[nodiscard]] const Vector3<MyType> &Translation() const;

    [[nodiscard]] const Vector3<int> &Repetitions() const;

    [[nodiscard]] std::vector<int> LocationCounts() const;

private:
    const std::vector<Vector2<MyType>> parameters_;
    const std::vector<int> parameter_indices_;
    const std::vector<Vector2<MyType>> upper_bounds_;
    const std::vector<Vector2<MyType>> lower_bounds_;
    const std::vector<ShapeTypeV2> shape_types_;
    const std::vector<Vector3<MyType>> positions_;
    const std::vector<int> position_indices_;

    Vector3<int> repetitions_;
    Vector3<MyType> translation_;
};


#endif
