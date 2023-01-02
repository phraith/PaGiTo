//
// Created by Phil on 26.03.2022.
//

#include "client.h"
#include "majordomo_utility.h"

#include <utility>
#include <spdlog/spdlog.h>

majordomo::Client::Client(std::string  broker_address)
:
broker_address_(broker_address),
socket_(context_, zmq::socket_type::dealer)
{
    socket_.connect(broker_address_);
}

void majordomo::Client::Send(const std::string &service_name, const std::string& job_payload, const std::vector<std::byte> &image_data){
    zmq::multipart_t message;
    message.pushmem(&image_data[0], sizeof(std::byte) * image_data.size());
    message.pushstr(job_payload);
    message.pushstr(service_name);
    message.pushstr("MDPC01");

    majordomo::SendToDealer(socket_, message);
}

zmq::multipart_t majordomo::Client::Recv(const std::string &service_name) {

    zmq::multipart_t response;
    majordomo::ReceiveFromDealer(socket_, response);

    if (response.size() >= 3)
    {
        auto header = response.pop();
        if (header.to_string() == "MDPC01")
        {
            auto service = response.pop();
            if (service.to_string() == service_name)
            {
                spdlog::info("Received valid response for subtask");
                return response;
            }
        }
    }

    throw std::invalid_argument( "Received message is invalid!" );
}
