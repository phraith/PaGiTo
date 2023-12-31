//
// Created by Phil on 06.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SERVICE_H
#define GISAXSMODELINGFRAMEWORK_SERVICE_H


#include <string>
#include <vector>
#include "standard_defs.h"

class Service {
public:
    virtual ~Service() = default;

    [[nodiscard]] virtual std::string ServiceName() const = 0;

    [[nodiscard]] virtual RequestResult HandleRequest(const std::string &request) = 0;
};


#endif //GISAXSMODELINGFRAMEWORK_SERVICE_H
