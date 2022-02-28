#include "common/qgrid.h"

#include <cmath>
#include <algorithm>
#include <iostream>

#include "standard_vector_types.h"
#include "common/standard_defs.h"
#include "common/timer.h"

QGrid::QGrid(const DetectorConfiguration& detector, const std::vector<int>& position_offsets, const BeamConfiguration& beam_config, std::complex<MyType> refractive_index)
	:
	resolution_(detector.Resolution()),
	direct_beam_location_(detector.Directbeam()),
	pixel_size_(detector.Pixelsize()),
	position_offsets_(position_offsets),
	qcount_(position_offsets_.size() != 0 ? position_offsets_.size() : resolution_.x * resolution_.y),
	alpha_i_(beam_config.AlphaI()),
	k0_(beam_config.K0()),
	sample_detector_dist_(detector.SampleDistance()),
	refractive_index_(refractive_index),
	alpha_fs_(qcount_),
	theta_fs_(qcount_),
	qpoints_xy_(2 * qcount_),
	qpoints_z_coeffs_(4 * qcount_),
	qx_(qcount_),
	qy_(qcount_),
	qz_(qcount_),
	qpar_(qcount_),
	q_(4 * qcount_)
{
	InitializeQGrid();
}


const std::vector<std::complex<MyType>>& QGrid::QPointsXY() const
{
	return qpoints_xy_;
}

const std::vector<std::complex<MyType>>& QGrid::QPointsZCoeffs() const
{
	return qpoints_z_coeffs_;
}

const std::vector<MyType>& QGrid::AlphaFs() const
{
	return alpha_fs_;
}

const std::vector<MyType>& QGrid::ThetaFs() const
{
	return theta_fs_;
}

int QGrid::QCount() const
{
	return qcount_;
}

std::string QGrid::InfoStr() const
{
	std::string info = "";
	info += "QGrid:\n";
	info += "	-pixel_size in mm: " + std::to_string(pixel_size_) + "\n";
	info += "	-resolution (x, y): " + std::to_string(resolution_.x) + ", " + std::to_string(resolution_.y) + "\n";
	info += "	-direct beam location in mm (x, y): " + std::to_string(direct_beam_location_.x * pixel_size_) + ", " + std::to_string(direct_beam_location_.y * pixel_size_) + "\n";
	info += "	-alpha_i (in rad): " + std::to_string(alpha_i_) + "\n";
	info += "	-k0 (in 1/nm): " + std::to_string(k0_) + "\n";
	info += "	-qcount: " + std::to_string(qcount_) + "\n";
	info += "	-sample_to_detector distance (in mm): " + std::to_string(sample_detector_dist_) + "\n";
	info += "	-n^2 - 1 (2 * delta, 2 * beta): " + std::to_string(refractive_index_.real()) + ", " + std::to_string(refractive_index_.imag()) + "\n";
	return info;
}

const std::vector<MyType> &QGrid::Qx() const
{
	return qx_;
}

const std::vector<MyType> &QGrid::Qy() const
{
	return qy_;
}

const std::vector<MyType> &QGrid::Qz() const
{
	return qz_;
}

const std::vector<std::complex<MyType>> &QGrid::QPar() const
{
	return qpar_;
}

const std::vector<std::complex<MyType>> &QGrid::Q() const
{
	return q_;
}

MyType2I QGrid::Resolution() const
{
	return resolution_;
}

void QGrid::InitializeQGrid()
{
	if (!position_offsets_.empty())
	{
		for (int i = 0; i < position_offsets_.size(); ++i)
		{
			int position_offset = position_offsets_.at(i);

			int pixel_y = position_offset / resolution_.x;
			int pixel_x = position_offset % resolution_.x;

			RealSpaceToQ(pixel_x, pixel_y, i);
		}
	}
	else
	{
        Timer t;
        t.Start();
		for (int i = 0; i < resolution_.y; ++i)
		{
			for (int j = 0; j < resolution_.x; ++j)
			{
				int idx = i * resolution_.x + j;
				RealSpaceToQ(j, i, idx);
			}
		}
        t.End();
        std::cout << "QGrid: " << t.Duration() << " s" << std::endl;
	}
}

void QGrid::RealSpaceToQ(int x, int y, int i)
{
	int pixel_x = x + 1;
	int pixel_y = y + 1;

	MyType pixel_dist_x = pixel_size_ * (pixel_x - direct_beam_location_.x);
	MyType pixel_dist_y = pixel_size_ * (pixel_y - direct_beam_location_.y);

	MyType theta_f = std::atan2(pixel_dist_x, sample_detector_dist_);

	MyType quad_dist_x = std::sqrt(sample_detector_dist_ * sample_detector_dist_ + pixel_dist_x * pixel_dist_x);
	MyType alpha_f = std::atan2(pixel_dist_y, quad_dist_x);

	MyType qx = k0_ * (std::cos(alpha_f) * std::cos(theta_f) - std::cos(alpha_i_));
	MyType qy = k0_ * std::cos(alpha_f) * std::sin(theta_f);
	MyType qz = k0_ * ( std::sin(alpha_f) + std::sin(alpha_i_) );

	MyType qx2 = qx * qx;
	MyType qy2 = qy * qy;

	MyType qpar = std::sqrt(qx2 + qy2);
	std::complex<MyType> cqpar =  { qpar, 0 };
	qpar_.at(i) = { cqpar.real(), cqpar.imag() };

	qx_.at(i) = qx;
	qy_.at(i) = qy;
	qz_.at(i) = qz;

	std::complex<MyType> cqx = { qx, 0 };
	std::complex<MyType> cqy = { qy, 0 };
	
	std::complex<MyType> refm1 = refractive_index_;
	refm1 = 0;
	//std::complex<MyType> refm1 = 2.f * std::complex<MyType>{ 6e-06,  2e-08 };
	//std::complex<MyType> refm1 = 2.f * std::complex<MyType>{ 0.001, 1e-05 };
	std::complex<MyType> kz_af = k0_ * std::sqrt(std::sin(alpha_f) * std::sin(alpha_f) + refm1);
	std::complex<MyType> kz_ai = -k0_ * std::sqrt(std::sin(alpha_i_) * std::sin(alpha_i_) + refm1);

	alpha_fs_.at(i) = alpha_f;
	theta_fs_.at(i) = theta_f;

	/*x0,y0,x1,y1,...*/
	qpoints_xy_.at(i) = { cqx.real(), cqx.imag() };
	qpoints_xy_.at(qcount_ + i) = { cqy.real(), cqy.imag() };

	std::complex<MyType> qz0 = kz_af - kz_ai;
	std::complex<MyType> qz1 = -kz_af - kz_ai;
	std::complex<MyType> qz2 = kz_af + kz_ai;
	std::complex<MyType> qz3 = -kz_af + kz_ai;

	/*z00,z01,z02,z03,z10,...*/
	qpoints_z_coeffs_.at(i) = { qz0.real(), qz0.imag() };
	qpoints_z_coeffs_.at(qcount_ + i) = { qz1.real(), qz1.imag() };
	qpoints_z_coeffs_.at(2 * qcount_ + i) = { qz2.real(), qz2.imag() };
	qpoints_z_coeffs_.at(3 * qcount_ + i) = { qz3.real(), qz3.imag() };

	std::complex<MyType> qz02 = qz0 * qz0;
	std::complex<MyType> qz12 = qz1 * qz1;
	std::complex<MyType> qz22 = qz2 * qz2;
	std::complex<MyType> qz32 = qz3 * qz3;

	std::complex<MyType> q0 = std::sqrt(qx2 + qy2 + qz02);
	std::complex<MyType> q1 = std::sqrt(qx2 + qy2 + qz12);
	std::complex<MyType> q2 = std::sqrt(qx2 + qy2 + qz22);
	std::complex<MyType> q3 = std::sqrt(qx2 + qy2 + qz32);

	q_.at(i) = { q0.real(), q0.imag() };
	q_.at(qcount_ + i) = { q1.real(), q1.imag() };
	q_.at(2 * qcount_ + i) = { q2.real(), q2.imag() };
	q_.at(3 * qcount_ + i) = { q3.real(), q3.imag() };
}
