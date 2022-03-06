//
// Created by Phil on 06.03.2022.
//

#include <iostream>
#include "gpu/core/random_generator.h"

RandomGenerator::RandomGenerator()
        :
        gen_() {
    auto status = curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        std::cout << "Error encountered in generating handle" << std::endl;
    }
}

RandomGenerator::~RandomGenerator() {
    curandDestroyGenerator(gen_);
}

void RandomGenerator::GenerateRandoms(float *rands, int size, float mean, float stddev) const {
    curandGenerateNormal(gen_, rands, size, mean, stddev);
}
