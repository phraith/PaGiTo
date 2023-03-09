//
// Created by phili on 29.01.2023.
//

#include <sys/time.h>
#include "rabbitmq_client.h"
#include "amqp.h"
#include "amqp_tcp_socket.h"
#include "amqp_framing.h"
#include <iostream>
#include <spdlog/spdlog.h>

#define SUMMARY_EVERY_US 1000000

uint64_t now_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t) tv.tv_sec * 1000000 + (uint64_t) tv.tv_usec;
}


RabbitMqClient::RabbitMqClient(const std::string &host_name, int port, const std::string &queue_name)
        :
        host_name_(host_name),
        queue_name_(queue_name),
        port_(port),
        connection_(nullptr) {

    SetUpConnection();
}

void RabbitMqClient::SetUpConnection() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (connection_ != nullptr) {
            int ret_connection = amqp_destroy_connection(connection_);
            if (ret_connection < 0) {
                spdlog::error("Closing connection did not work...");
                continue;
            }
        }

        connection_ = amqp_new_connection();
        auto socket = amqp_tcp_socket_new(connection_);
        if (!socket) {
            spdlog::error("Creating TCP socket failed...");
            continue;
        }

        auto status = amqp_socket_open(socket, host_name_.c_str(), port_);
        if (status) {
            spdlog::error("Opening TCP socket failed...");
            continue;
        }

        auto user = std::getenv("RABBITMQ_USER");
        auto password = std::getenv("RABBITMQ_PASSWORD");

        if (user == nullptr || password == nullptr) {
            spdlog::error("RABBITMQ_USER or RABBITMQ_PASSWORD are not set...");
            continue;
        }

        auto ret = amqp_login(connection_, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN, user, password,
                              std::getenv("RABBITMQ_PASSWORD"));
        bool success = EvaluateRpcOperation(ret);
        if (!success) {
            spdlog::error("Login failed...");
            continue;
        }

        amqp_channel_open(connection_, 1);
        amqp_channel_open(connection_, 2);

        if (!EvaluateRpcOperation(amqp_get_rpc_reply(connection_)))
        {
            continue;
        }

        amqp_bytes_t amq_queue_name{queue_name_.size(), (void *) queue_name_.c_str()};
        amqp_basic_consume(connection_, 1, amq_queue_name, amqp_empty_bytes, 0, 1, 0,
                           amqp_empty_table);

        return;
    }
}

bool RabbitMqClient::EvaluateRpcOperation(amqp_rpc_reply_t reply) {
    if (AMQP_RESPONSE_NORMAL == reply.reply_type) {
        return true;
    }

    if (AMQP_RESPONSE_LIBRARY_EXCEPTION == reply.reply_type &&
        AMQP_STATUS_CONNECTION_CLOSED == reply.library_error) {
        return false;
    }

    if (AMQP_RESPONSE_LIBRARY_EXCEPTION == reply.reply_type) {

        switch (reply.library_error) {
            case AMQP_STATUS_CONNECTION_CLOSED:
                break;
            case AMQP_STATUS_UNEXPECTED_STATE:
                amqp_frame_t frame;
                if (AMQP_STATUS_OK != amqp_simple_wait_frame(connection_, &frame)) {

                }

                if (AMQP_FRAME_METHOD == frame.frame_type) {
                    switch (frame.payload.method.id) {
                        case AMQP_BASIC_ACK_METHOD:
                            /* if we've turned publisher confirms on, and we've published a
                             * message here is a message being confirmed.
                             */
                            break;
                        case AMQP_BASIC_RETURN_METHOD:
                            /* if a published message couldn't be routed and the mandatory
                             * flag was set this is what would be returned. The message then
                             * needs to be read.
                             */
                        {
                            amqp_message_t message;
                            reply = amqp_read_message(connection_, frame.channel, &message, 0);
                            if (AMQP_RESPONSE_NORMAL != reply.reply_type) {
                                break;
                            }

                            amqp_destroy_message(&message);
                        }
                            break;

                        case AMQP_CHANNEL_CLOSE_METHOD:
                            /* a channel.close method happens when a channel exception occurs,
                             * this can happen by publishing to an exchange that doesn't exist
                             * for example.
                             *
                             * In this case you would need to open another channel redeclare
                             * any queues that were declared auto-delete, and restart any
                             * consumers that were attached to the previous channel.
                             */
                            break;

                        case AMQP_CONNECTION_CLOSE_METHOD:
                            /* a connection.close method happens when a connection exception
                             * occurs, this can happen by trying to use a channel that isn't
                             * open for example.
                             *
                             * In this case the whole connection must be restarted.
                             */
                            break;

                        default:
                            spdlog::error("An unexpected method was received {}", frame.payload.method.id);
                            break;
                    }
                }
                break;
        }
    }
    return false;
}