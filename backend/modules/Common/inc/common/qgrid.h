#ifndef MODEL_SIMULATOR_UTIL_QGRID_H
#define MODEL_SIMULATOR_UTIL_QGRID_H

#include <vector>
#include <complex>
#include <string>

#include <common/beam_configuration.h>

#include "parameter_definitions/detector_setup.h"
#include "parameter_definitions/experimental_setup.h"
#include "standard_defs.h"
class QGrid
{
public:
	QGrid(const DetectorConfiguration& detector, const std::vector<Vector2<int>>& position_offsets, const BeamConfiguration& beam_config, std::complex<MyType> refractive_index);

	[[nodiscard]] const std::vector<std::complex<MyType>> &QPointsXY() const;
	[[nodiscard]] const std::vector<std::complex<MyType>> &QPointsZCoeffs() const;
	[[nodiscard]] const std::vector<MyType>& AlphaFs() const;
	[[nodiscard]] const std::vector<MyType>& ThetaFs() const;

	[[nodiscard]] int QCount() const;

	[[nodiscard]] std::string InfoStr() const;

	[[nodiscard]] const std::vector<MyType> &Qx() const;
	[[nodiscard]] const std::vector<MyType> &Qy() const;
	[[nodiscard]] const std::vector<MyType> &Qz() const;

	[[nodiscard]] const std::vector<std::complex<MyType>> &QPar() const;
	[[nodiscard]] const std::vector<std::complex<MyType>> &Q() const;

	[[nodiscard]] Vector2<int> Resolution() const;

private:
	void InitializeQGrid();
	
	void RealSpaceToQ(int x, int y, int i);

	const Vector2<int> &resolution_;
	const Vector2<int>&direct_beam_location_;

	MyType pixel_size_{};
	const std::vector<Vector2<int>>& position_offsets_;

	int qcount_;
	MyType alpha_i_{};
	MyType k0_{};

	MyType sample_detector_dist_{};
	std::complex<MyType> refractive_index_;

	std::vector<MyType> alpha_fs_;
	std::vector<MyType> theta_fs_;

	std::vector<std::complex<MyType>> qpoints_xy_;
	std::vector<std::complex<MyType>> qpoints_z_coeffs_;
	std::vector<std::complex<MyType>> qpar_;
	std::vector<std::complex<MyType>> q_;

	std::vector<MyType> qx_;
	std::vector<MyType> qy_;
	std::vector<MyType> qz_;
};
#endif