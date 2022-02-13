#include "common/experimental_model.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>


ExperimentalModel::ExperimentalModel(DetectorSetup detector, const std::vector<int> position_offsets,
                                     BeamConfiguration beam_config, Sample sample, double sample_detector_dist,
                                     int level)
        :
        detector_(detector),
        position_offsets_(position_offsets),
        beam_config_(beam_config),
        sample_(sample),
        sample_detector_dist_(sample_detector_dist),
        qgrid_(detector_, position_offsets_, beam_config_, sample_detector_dist_, sample.N2M1OfLevel(level)),
        level_(level),
        //prop_coeffs_(sample_.PropagationCoeffs(beam_config_.AlphaI(), qgrid_.AlphaFs(), beam_config_.K0(), level))
        prop_coeffs_(sample_.PropagationCoeffsTopBuried(beam_config_.AlphaI(), qgrid_.AlphaFs(), beam_config_.K0())) {
}

const QGrid &ExperimentalModel::GetQGrid() const {
    return qgrid_;
}

const std::vector<MyComplex> &ExperimentalModel::GetPropagationCoefficients() const {
    return prop_coeffs_;
}

void ExperimentalModel::PrintInfo() const {
    std::cout << detector_.InfoStr() << std::endl << std::endl;
    std::cout << beam_config_.InfoStr() << std::endl << std::endl;
    std::cout << qgrid_.InfoStr() << std::endl << std::endl;

    int qcount = qgrid_.QCount();


    int print_count = 2;
    bool print_all = (qcount <= 2 * print_count) ? true : false;

    std::string dwba_info = "Dwba info:\n";
    if (print_all) {
        for (int i = 0; i < qcount; ++i) {
            dwba_info += DwbaInfo(i);
        }
    } else {
        dwba_info += "	-First " + std::to_string(print_count) + " elements:\n";
        for (int i = 0; i < print_count; ++i) {
            dwba_info += DwbaInfo(i);
        }

        dwba_info += "	-Last " + std::to_string(print_count) + " elements:\n";
        for (int i = qcount - print_count; i < qcount; ++i) {
            dwba_info += DwbaInfo(i);
        }
    }

    std::cout << dwba_info << std::endl;
}

std::string ExperimentalModel::DwbaInfo(int idx) const {
    std::string dwba_info = "";
    MyType alpha_f = qgrid_.AlphaFs().at(idx);
    MyType theta_f = qgrid_.ThetaFs().at(idx);

    int qcount = qgrid_.QCount();

    const MyType2 &qx = qgrid_.QPointsXY().at(idx);
    const MyType2 &qy = qgrid_.QPointsXY().at(qcount + idx);

    const MyType2 &qz1 = qgrid_.QPointsZCoeffs().at(idx);
    const MyType2 &qz2 = qgrid_.QPointsZCoeffs().at(qcount + idx);
    const MyType2 &qz3 = qgrid_.QPointsZCoeffs().at(2 * qcount + idx);
    const MyType2 &qz4 = qgrid_.QPointsZCoeffs().at(3 * qcount + idx);

    /*const MyComplex& trans_ref = prop_coeffs_.at(idx);

    dwba_info += "	exit_angle in rad (a_f, t_f): (" + std::to_string(alpha_f) + ", " + std::to_string(theta_f) + ")\n";
    dwba_info += "	qpar in 1/nm (qx, qy): (" + std::to_string(qx.x) + " " + std::to_string(qx.y) + ", " + std::to_string(qy.x) + " " + std::to_string(qy.y) + ")\n";
    dwba_info += "	qz1, qz2 in 1/nm : (" + std::to_string(qz1.x) + " " + std::to_string(qz1.y) + ", " + std::to_string(qz2.x) + " " + std::to_string(qz2.y) + ")\n";
    dwba_info += "	rf1, rf2: (" + std::to_string(trans_ref.x.x) + " " + std::to_string(trans_ref.x.y) + ", " + std::to_string(trans_ref.y.x) + " " + std::to_string(trans_ref.y.y) + ")\n";
    dwba_info += "	qz3, qz4 in 1/nm : (" + std::to_string(qz3.x) + " " + std::to_string(qz3.y) + ", " + std::to_string(qz4.x) + " " + std::to_string(qz4.y) + ")\n";
    dwba_info += "	rf3, rf4: (" + std::to_string(trans_ref.z.x) + " " + std::to_string(trans_ref.z.y) + ", " + std::to_string(trans_ref.w.x) + " " + std::to_string(trans_ref.w.y) + ")\n\n";*/

    return dwba_info;
}


