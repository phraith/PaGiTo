//
// Created by Phil on 28.02.2022.
//

#include "common/job_meta_information.h"

#include <utility>

JobMetaInformation::JobMetaInformation(std::string id)
:
id_(std::move(id))
{
}

const std::string &JobMetaInformation::ID() const{
    return id_;
}
