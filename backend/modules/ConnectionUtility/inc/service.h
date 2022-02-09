//
// Created by Phil on 06.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SERVICE_H
#define GISAXSMODELINGFRAMEWORK_SERVICE_H


#include <string>

class Service {
public:
    virtual ~Service() = default;

    [[nodiscard]] virtual std::string ServiceName() const = 0;

    [[nodiscard]] virtual std::string HandleRequest(const std::string &request) const = 0;
};


#endif //GISAXSMODELINGFRAMEWORK_SERVICE_H
