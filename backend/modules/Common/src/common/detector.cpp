//#include "common/detector.h"
//
//Detector::Detector(double pixel_size, MyType2I resolution, MyType2 direct_beam_location)
//	:
//	pixel_size_(pixel_size),
//	resolution_(resolution),
//	direct_beam_location_(direct_beam_location)
//{
//}
//
//double Detector::PixelSize() const
//{
//	return pixel_size_;
//}
//
//const MyType2I&Detector::Resolution() const
//{
//	return resolution_;
//}
//
//const MyType2& Detector::DirectBeamLocation() const
//{
//	return direct_beam_location_;
//}
//
//std::string Detector::InfoStr() const
//{
//	std::string info = "";
//	info += "Detector info:\n";
//	info += "	-pixel_size in mm: " + std::to_string(pixel_size_) + "\n";
//	info += "	-resolution (x, y): " + std::to_string(resolution_.x) + ", " + std::to_string(resolution_.y) + "\n";
//	info += "	-direct beam location in mm (x, y): " + std::to_string(direct_beam_location_.x * pixel_size_) + ", " + std::to_string(direct_beam_location_.y * pixel_size_) + "\n";
//	return info;
//}
//
////Detector::Detector(DetectorParameters setup)
////:
////pixel_size_(setup.Pixelsize()),
////resolution_(setup.Resolution()),
////direct_beam_location_(setup.Directbeam())
////{
////
////}
