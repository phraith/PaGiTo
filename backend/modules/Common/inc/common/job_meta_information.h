//
// Created by Phil on 28.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_JOB_META_INFORMATION_H
#define GISAXSMODELINGFRAMEWORK_JOB_META_INFORMATION_H


#include <string>

class JobMetaInformation {
public:
    explicit JobMetaInformation(std::string id);

    [[nodiscard]] const std::string &ID() const;

private:
    std::string id_;
};


#endif //GISAXSMODELINGFRAMEWORK_JOB_META_INFORMATION_H
