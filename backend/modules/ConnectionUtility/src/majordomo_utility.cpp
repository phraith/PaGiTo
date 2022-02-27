//
// Created by Phil on 23.01.2022.
//

#include "majordomo_utility.h"
#include "broker.h"
#include "worker_request_handler.h"
#include <zmq.hpp>
#include <thread>

namespace majordomo {
    zmq::send_result_t
    SendToClient(zmq::socket_t &socket, const zmq::multipart_t &mulitpart_msg, zmq::send_flags flags) {
        zmq::message_t msg = mulitpart_msg.encode();
        return socket.send(msg, flags);
    }

    bool SendToRouter(zmq::socket_t &socket, zmq::multipart_t &multipart_msg,
                                 const majordomo::remote_id_t &client_id) {
        multipart_msg.pushmem(nullptr, 0);
        multipart_msg.pushstr(client_id);
        return multipart_msg.send(socket);
    }

    bool ReceiveFromFromRouter(zmq::socket_t &socket, zmq::multipart_t &multipart_msg,
                                          majordomo::remote_id_t &client_id) {
        auto res = multipart_msg.recv(socket);
        if (!res) { return res; }
        client_id = multipart_msg.popstr();
        multipart_msg.pop();
        return res;
    }

    zmq::recv_result_t
    ReceiveFromClient(zmq::socket_t &socket, zmq::multipart_t &multipart_msg, zmq::recv_flags flags) {
        zmq::message_t msg;
        auto res = socket.recv(msg, flags);
        if (!res) { return res; }
        multipart_msg.decode_append(msg);
        return res;
    }

    ms_time_t ms_now() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch());
    }

    void BrokerActor(const std::string &address) {
        Broker broker(address);

        ms_time_t now = ms_now();
        ms_time_t heartbeat_interval = HEARTBEAT_INTERVAL;
        ms_time_t heartbeat_at = now  + heartbeat_interval;

        zmq::poller_t poller;
        poller.add(broker.Socket(), zmq::event_flags::pollin);

        bool interrupted = false;
        while(!interrupted)
        {
            ms_time_t  timeout = ms_time_t::zero();
            //if (heartbeat_at > now) { timeout = heartbeat_at - now; }

            std::vector<zmq::poller_event<>> events(2);
            int events_count = poller.wait_all(events, timeout);
            for (int i = 0; i < events_count; ++i) {
                if(events[i].socket == broker.Socket() )
                {
                    broker.ProcessInputSocket();
                }
            }

            //broker.ProcessHeartbeat(heartbeat_at);
            //heartbeat_at += heartbeat_interval;
            //now = majordomo::ms_now();
        }
    }

    void ms_sleep(ms_time_t timeout) {
        std::this_thread::sleep_for(timeout);
    }

    void EchoWorker(const std::string& broker_address, const std::string& worker_address) {
        WorkerRequestHandler worker(worker_address, broker_address, "simm");
        ms_time_t now = ms_now();
        ms_time_t heartbeat_interval = HEARTBEAT_INTERVAL;
        ms_time_t  heartbeat_at = now + heartbeat_interval;

        zmq::poller_t poller;
        poller.add(worker.Socket(), zmq::event_flags::pollin);

        bool interrupted = false;
        zmq::multipart_t reply;
        while(!interrupted)
        {
            ms_time_t  timeout = ms_time_t::zero();
            if (heartbeat_at > now)
            {
                timeout = heartbeat_at - now;
                std::vector<zmq::poller_event<>> events(2);
                int events_count = poller.wait_all(events, timeout);
                for (int i = 0; i < events_count; ++i) {
                    if(events[i].socket == worker.Socket() )
                    {
                        zmq::multipart_t request;
                        worker.Receive(request);
                        if(request.empty())
                        {
                            break;
                        }
                        reply = std::move(request);
                        worker.Send(reply);
                    }
                }
            }
        }
    }

    zmq::send_result_t
    SendToDealer(zmq::socket_t &socket, zmq::multipart_t &mulitpart_msg, zmq::send_flags flags) {
        mulitpart_msg.pushmem(nullptr, 0);
        return mulitpart_msg.send(socket);
    }

    zmq::recv_result_t
    ReceiveFromDealer(zmq::socket_t &socket, zmq::multipart_t &multipart_msg, zmq::recv_flags flags) {
        auto res = multipart_msg.recv(socket);
        if (!res) {return res;}
        multipart_msg.pop();
        return res;
    }
}
