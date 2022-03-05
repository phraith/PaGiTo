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
//    UnitcellV2 CreateFromJson(nlohmann::json unitcell);
//    UnitcellV2 Convert(FlatUnitcellV2 flat_unitcell);
    FlatUnitcellV2 Convert(const UnitcellV2 &unitcell);

    //std::unique_ptr<Shape> CreateShape(ShapeTypeV2 shape_type, const nlohmann::json &json);
}

#endif //GISAXSMODELINGFRAMEWORK_UNITCELL_UTILITY_H
