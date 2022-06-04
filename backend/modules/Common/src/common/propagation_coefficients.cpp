//
// Created by Phil on 28.02.2022.
//

#include <complex>
#include "common/propagation_coefficients.h"
#include "common/sample.h"

std::vector<std::complex<MyType>>
PropagationCoefficientsCpu::PropagationCoeffsTopBuried(const SampleConfiguration &sample_config,
                                                       const DetectorConfiguration &detector,
                                                       const BeamConfiguration &beam_config) {
    int qcount = detector.Resolution().x * detector.Resolution().y;

    std::vector<std::complex<MyType>> trans_refs(4 * qcount);
    auto &substrate = sample_config.Layers().at(sample_config.Layers().size() - 1);
    const auto &ns2m1 = substrate.N2MinusOne();

    MyType sin_ai = std::sin(beam_config.AlphaI());
    MyType kzi = -1.f * beam_config.K0() * sin_ai;
    std::complex<MyType> tmp = std::sqrt(sin_ai * sin_ai - ns2m1);
    std::complex<MyType> rki = (sin_ai - tmp) / (sin_ai + tmp);

    MyType quad_dist_x = std::sqrt(
            detector.SampleDistance() * detector.SampleDistance() + detector.Pixelsize() * detector.Pixelsize());


#pragma omp parallel for default(none) shared(qcount, detector, beam_config, quad_dist_x, kzi, trans_refs, ns2m1, rki)
    for (int i = 0; i < qcount; ++i) {
        int y = (i / detector.Resolution().x) + 1;

        MyType pixel_dist_y = detector.Pixelsize() * (y - beam_config.BeamDirection().y);

        const auto alpha_f = std::atan2(pixel_dist_y, quad_dist_x);

        MyType qz = beam_config.K0() * (std::sin(alpha_f) + std::sin(beam_config.AlphaI()));
        MyType kzf = qz + kzi;

        if (kzf < 0) {
            trans_refs[i] = {0, 0};
            trans_refs[qcount + i] = {0, 0};
            trans_refs[2 * qcount + i] = {0, 0};
            trans_refs[3 * qcount + i] = {0, 0};
        } else {
            MyType sin_af = kzf / beam_config.K0();
            auto t = std::sqrt(sin_af * sin_af - ns2m1);
            std::complex<MyType> rkf = (sin_af - t) / (sin_af + t);

            std::complex<MyType> t4 = rki * rkf;

            trans_refs[i] = {1, 0};
            trans_refs[qcount + i] = {rkf.real(), rkf.imag()};
            trans_refs[2 * qcount + i] = {rki.real(), rki.imag()};
            trans_refs[3 * qcount + i] = {t4.real(), t4.imag()};
        }
    }

    return trans_refs;
}
