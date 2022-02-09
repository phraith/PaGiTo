//
// Created by Phil on 23.01.2022.
//

#ifndef GISAXSMODELINGFRAMEWORK_MAJORDOMO_UTILITY_H
#define GISAXSMODELINGFRAMEWORK_MAJORDOMO_UTILITY_H

#include <chrono>
#include <string>
#include <zmq.hpp>
#include <zmq_addon.hpp>

namespace majordomo {

    namespace client {
        inline const char *ident = "MDPC01";
    }

    namespace worker {
        inline const char* ident = "MDPW01";

        /// WorkerInformation commands as strings
        inline const char* ready = "\001";
        inline const char* request = "\002";
        inline const char* reply = "\003";
        inline const char* heartbeat = "\004";
        inline const char* disconnect = "\005";
    }

    using ms_time_t = std::chrono::milliseconds;
    using remote_id_t = std::string;

    const int HEARTBEAT_LIVENESS = 3;
    const ms_time_t HEARTBEAT_INTERVAL{2500};
    const ms_time_t HEARTBEAT_EXPIRY{HEARTBEAT_INTERVAL * HEARTBEAT_LIVENESS};

    ms_time_t now();

    bool
    SendToRouter(zmq::socket_t &socket, zmq::multipart_t &mulitpart_msg, const remote_id_t &client_id);

    bool
    ReceiveFromFromRouter(zmq::socket_t &socket, zmq::multipart_t &multipart_msg, remote_id_t &client_id);

    zmq::send_result_t SendToClient(zmq::socket_t &socket, const zmq::multipart_t &mulitpart_msg,
                                    zmq::send_flags flags = zmq::send_flags::none);

    zmq::send_result_t SendToDealer(zmq::socket_t &socket, zmq::multipart_t &mulitpart_msg,
                                    zmq::send_flags flags = zmq::send_flags::none);

    zmq::recv_result_t ReceiveFromDealer(zmq::socket_t &socket, zmq::multipart_t &multipart_msg,
                                         zmq::recv_flags flags = zmq::recv_flags::none);

    zmq::recv_result_t ReceiveFromClient(zmq::socket_t &socket, zmq::multipart_t &multipart_msg,
                                         zmq::recv_flags flags = zmq::recv_flags::none);

    void BrokerActor(const std::string &address);

    void EchoWorker(const std::string& broker_address, const std::string& worker_address);

    ms_time_t ms_now();

    void ms_sleep(ms_time_t timeout);
}


#endif //GISAXSMODELINGFRAMEWORK_MAJORDOMO_UTILITY_H
