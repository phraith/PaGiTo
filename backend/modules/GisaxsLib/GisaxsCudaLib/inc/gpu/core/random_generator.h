//
// Created by Phil on 06.03.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_RANDOM_GENERATOR_H
#define GISAXSMODELINGFRAMEWORK_RANDOM_GENERATOR_H


#include <curand.h>

class RandomGenerator {
public:
    RandomGenerator();
    ~RandomGenerator();

    void GenerateRandoms(float *rands, int size, float mean, float stddev) const;
private:
    curandGenerator_t gen_{};
};

#endif //GISAXSMODELINGFRAMEWORK_RANDOM_GENERATOR_H
