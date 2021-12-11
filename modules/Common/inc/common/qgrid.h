#ifndef MODEL_SIMULATOR_UTIL_QGRID_H
#define MODEL_SIMULATOR_UTIL_QGRID_H

#include <vector>
#include <complex>
#include <string>

#include <common/beam_configuration.h>
#include <common/detector.h>

#include "vector_types.h"

class QGrid
{
public:
	QGrid(const Detector& detector, const std::vector<int>& position_offsets, const BeamConfiguration& beam_config, MyType sample_detector_dist, std::complex<MyType> refractive_index);

	const std::vector<MyType2> &QPointsXY() const;
	const std::vector<MyType2> &QPointsZCoeffs() const;
	const std::vector<MyType>& AlphaFs() const;
	const std::vector<MyType>& ThetaFs() const;

	int QCount() const;

	std::string InfoStr() const;

	const std::vector<MyType> &Qx() const;
	const std::vector<MyType> &Qy() const;
	const std::vector<MyType> &Qz() const;

	const std::vector<MyType2> &QPar() const;
	const std::vector<MyType2> &Q() const;

	MyType2I Resolution() const;

private:
	void InitializeQGrid();
	
	void RealSpaceToQ(int x, int y, int i);

	const MyType2I &resolution_;
	const MyType2&direct_beam_location_;

	MyType pixel_size_;
	const std::vector<int>& position_offsets_;

	int qcount_;
	MyType alpha_i_;
	MyType k0_;

	MyType sample_detector_dist_;
	std::complex<MyType> refractive_index_;

	std::vector<MyType> alpha_fs_;
	std::vector<MyType> theta_fs_;

	std::vector<MyComplex> qpoints_xy_;
	std::vector<MyComplex> qpoints_z_coeffs_;
	std::vector<MyComplex> qpar_;
	std::vector<MyComplex> q_;

	std::vector<MyType> qx_;
	std::vector<MyType> qy_;
	std::vector<MyType> qz_;
};
#endif