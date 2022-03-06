#include "util/utility.h"
#include <algorithm>

#include <stdexcept>

ShapeTypeV2 Utility::ConvertStringToShapeType(std::string input)
{
	std::transform(input.begin(), input.end(), input.begin(), ::tolower);
	if (input == "cylinder")
		return ShapeTypeV2::cylinder;
	else if (input == "sphere")
		return ShapeTypeV2::sphere;
	else
		throw std::runtime_error("not supported shape type!");
}
