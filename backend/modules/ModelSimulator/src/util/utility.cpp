#include "util/utility.h"
#include <algorithm>

#include <stdexcept>

ShapeType Utility::ConvertStringToShapeType(std::string input)
{
	std::transform(input.begin(), input.end(), input.begin(), ::tolower);
	if (input == "cylinder")
		return ShapeType::kCylinder;
	else if (input == "sphere")
		return ShapeType::kSphere;
	else if (input == "trapezoid")
		return ShapeType::kTrapezoid;
	else
		throw std::runtime_error("not supported shape type!");
}
