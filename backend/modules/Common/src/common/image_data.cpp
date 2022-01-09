#include "common/image_data.h"

#include <iostream>
#include <algorithm>
ImageData::ImageData(std::vector<float> intensities, std::vector<int> offsets)
	:
	intensities_(std::move(intensities)),
	max_intensity_(intensities_[std::distance(intensities_.begin(), std::max_element(intensities_.begin(), intensities_.end()))]),
	min_intensity_(intensities_[std::distance(intensities_.begin(), std::min_element(intensities_.begin(), intensities_.end()))]),
	offsets_(std::move(offsets)),
	uuid_(UuidGenerator::generate_uuid())
{
}

ImageData::~ImageData()
{}

const std::vector<float> &ImageData::Intensities() const
{
	return intensities_;
}

const std::vector<int>& ImageData::Offsets() const
{
	return offsets_;
}

const std::string& ImageData::Id() const
{
	return uuid_;
}

int ImageData::Size() const
{
	return intensities_.size();
}

float ImageData::MaxIntensity() const
{
	return max_intensity_;
}
