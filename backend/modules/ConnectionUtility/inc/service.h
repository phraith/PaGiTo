//
// Created by Phil on 06.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_SERVICE_H
#define GISAXSMODELINGFRAMEWORK_SERVICE_H


#include <string>
#include <vector>

class Service {
public:
    virtual ~Service() = default;

    [[nodiscard]] virtual std::string ServiceName() const = 0;

    [[nodiscard]] virtual std::vector<std::byte> HandleRequest(const std::string &request, std::vector<std::byte> image_data, const std::string &origin) = 0;
};


#endif //GISAXSMODELINGFRAMEWORK_SERVICE_H
