#include "../inc/connector_v2.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <memory>

#include "zmq.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

ConnectorV2::ConnectorV2(const std::string ip)
    :
    ip_(ip),
    connection_handler_(&ConnectorV2::Listen, this)
{
}

ConnectorV2::~ConnectorV2()
{
}

void ConnectorV2::Listen()
{
    zmq::context_t context;
    zmq::socket_t socket(context, zmq::socket_type::rep);
    socket.bind("tcp://" + ip_);
    std::cout << "tcp://" + ip_ + "\n" << std::endl;

    /* Assign the in-process name "#1" */
    zmq::context_t local_job_context;
    zmq::socket_t local_job_socket(local_job_context, zmq::socket_type::pub);
    local_job_socket.bind("inproc://local-job-endpoint");

    while (true) {
        zmq::message_t reply;
        zmq::recv_result_t result = socket.recv(reply, zmq::recv_flags::none);
        std::string json_data = std::string(static_cast<char*>(reply.data()), reply.size());
        json data = json::parse(json_data);

        if (!data.contains("job_type"))
        { 
            socket.send(zmq::buffer("Server received invalid job..."));
            continue; 
        } 

        std::string job_type_id = data["job_type"];
        zmq::send_result_t send_result = local_job_socket.send(zmq::buffer(job_type_id), zmq::send_flags::sndmore);
        zmq::send_result_t send_result2 = local_job_socket.send(zmq::buffer(json_data));

        socket.send(zmq::buffer("Server received valid job..."));
    }
    socket.unbind("tcp://" + ip_);
}
