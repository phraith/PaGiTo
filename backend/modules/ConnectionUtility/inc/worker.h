//
// Created by Phil on 05.02.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_WORKER_H
#define GISAXSMODELINGFRAMEWORK_WORKER_H


#include <string>
#include <memory>

#include "service.h"

class Worker {
public:
    Worker(std::unique_ptr<Service> &service, const std::string& worker_address, const std::string& broker_address);

    void Start();
private:
    std::unique_ptr<Service> service_;
    const std::string &broker_address_;
    const std::string &worker_address_;
};


#endif //GISAXSMODELINGFRAMEWORK_WORKER_H
