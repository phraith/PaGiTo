//
// Created by Phil on 13.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_UNITCELL_UTILITY_H
#define GISAXSMODELINGFRAMEWORK_UNITCELL_UTILITY_H

#include <nlohmann/json.hpp>
#include "unitcell_v2.h"
#include "flat_unitcell.h"

namespace GisaxsModeling
{
    inline UnitcellV2 CreateFromJson(nlohmann::json unitcell);
    inline UnitcellV2 Convert(FlatUnitcellV2 flat_unitcell);
    inline FlatUnitcellV2 Convert(UnitcellV2 unitcell);

    inline std::unique_ptr<Shape> CreateShape(ShapeTypeV2 shape_type, const nlohmann::json &json);
}

#endif //GISAXSMODELINGFRAMEWORK_UNITCELL_UTILITY_H
