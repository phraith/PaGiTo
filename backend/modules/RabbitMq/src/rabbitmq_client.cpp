//
// Created by phili on 29.01.2023.
//

#include <sys/time.h>
#include "rabbitmq_client.h"
#include "amqp.h"
#include "amqp_tcp_socket.h"
#include "amqp_framing.h"
#include <stdarg.h>
#define SUMMARY_EVERY_US 1000000

uint64_t now_microseconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + (uint64_t)tv.tv_usec;
}

static void run(amqp_connection_state_t conn) {
    uint64_t start_time = now_microseconds();
    int received = 0;
    int previous_received = 0;
    uint64_t previous_report_time = start_time;
    uint64_t next_summary_time = start_time + SUMMARY_EVERY_US;

    amqp_frame_t frame;

    uint64_t now;

    for (;;) {
        amqp_rpc_reply_t ret;
        amqp_envelope_t envelope;

        now = now_microseconds();
        if (now > next_summary_time) {
            int countOverInterval = received - previous_received;
            double intervalRate =
                    countOverInterval / ((now - previous_report_time) / 1000000.0);
            printf("%d ms: Received %d - %d since last report (%d Hz)\n",
                   (int)(now - start_time) / 1000, received, countOverInterval,
                   (int)intervalRate);

            previous_received = received;
            previous_report_time = now;
            next_summary_time += SUMMARY_EVERY_US;
        }

        amqp_maybe_release_buffers(conn);
        ret = amqp_consume_message(conn, &envelope, NULL, 0);

        if (AMQP_RESPONSE_NORMAL != ret.reply_type) {
            if (AMQP_RESPONSE_LIBRARY_EXCEPTION == ret.reply_type &&
                AMQP_STATUS_UNEXPECTED_STATE == ret.library_error) {
                if (AMQP_STATUS_OK != amqp_simple_wait_frame(conn, &frame)) {
                    return;
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
                            ret = amqp_read_message(conn, frame.channel, &message, 0);
                            if (AMQP_RESPONSE_NORMAL != ret.reply_type) {
                                return;
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
                            return;

                        case AMQP_CONNECTION_CLOSE_METHOD:
                            /* a connection.close method happens when a connection exception
                             * occurs, this can happen by trying to use a channel that isn't
                             * open for example.
                             *
                             * In this case the whole connection must be restarted.
                             */
                            return;

                        default:
                            fprintf(stderr, "An unexpected method was received %u\n",
                                    frame.payload.method.id);
                            return;
                    }
                }
            }

        } else {
            amqp_destroy_envelope(&envelope);
        }

        received++;
    }
}

void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

void die_on_error(int x, char const *context) {
    if (x < 0) {
        fprintf(stderr, "%s: %s\n", context, amqp_error_string2(x));
        exit(1);
    }
}

void die_on_amqp_error(amqp_rpc_reply_t x, char const *context) {
    switch (x.reply_type) {
        case AMQP_RESPONSE_NORMAL:
            return;

        case AMQP_RESPONSE_NONE:
            fprintf(stderr, "%s: missing RPC reply type!\n", context);
            break;

        case AMQP_RESPONSE_LIBRARY_EXCEPTION:
            fprintf(stderr, "%s: %s\n", context, amqp_error_string2(x.library_error));
            break;

        case AMQP_RESPONSE_SERVER_EXCEPTION:
            switch (x.reply.id) {
                case AMQP_CONNECTION_CLOSE_METHOD: {
                    amqp_connection_close_t *m =
                            (amqp_connection_close_t *)x.reply.decoded;
                    fprintf(stderr, "%s: server connection error %uh, message: %.*s\n",
                            context, m->reply_code, (int)m->reply_text.len,
                            (char *)m->reply_text.bytes);
                    break;
                }
                case AMQP_CHANNEL_CLOSE_METHOD: {
                    amqp_channel_close_t *m = (amqp_channel_close_t *)x.reply.decoded;
                    fprintf(stderr, "%s: server channel error %uh, message: %.*s\n",
                            context, m->reply_code, (int)m->reply_text.len,
                            (char *)m->reply_text.bytes);
                    break;
                }
                default:
                    fprintf(stderr, "%s: unknown server error, method id 0x%08X\n",
                            context, x.reply.id);
                    break;
            }
            break;
    }

    exit(1);
}

static void dump_row(long count, int numinrow, int *chs) {
    int i;

    printf("%08lX:", count - numinrow);

    if (numinrow > 0) {
        for (i = 0; i < numinrow; i++) {
            if (i == 8) {
                printf(" :");
            }
            printf(" %02X", chs[i]);
        }
        for (i = numinrow; i < 16; i++) {
            if (i == 8) {
                printf(" :");
            }
            printf("   ");
        }
        printf("  ");
        for (i = 0; i < numinrow; i++) {
            if (isprint(chs[i])) {
                printf("%c", chs[i]);
            } else {
                printf(".");
            }
        }
    }
    printf("\n");
}

static int rows_eq(int *a, int *b) {
    int i;

    for (i = 0; i < 16; i++)
        if (a[i] != b[i]) {
            return 0;
        }

    return 1;
}

void amqp_dump(void const *buffer, size_t len) {
    unsigned char *buf = (unsigned char *)buffer;
    long count = 0;
    int numinrow = 0;
    int chs[16];
    int oldchs[16] = {0};
    int showed_dots = 0;
    size_t i;

    for (i = 0; i < len; i++) {
        int ch = buf[i];

        if (numinrow == 16) {
            int j;

            if (rows_eq(oldchs, chs)) {
                if (!showed_dots) {
                    showed_dots = 1;
                    printf(
                            "          .. .. .. .. .. .. .. .. : .. .. .. .. .. .. .. ..\n");
                }
            } else {
                showed_dots = 0;
                dump_row(count, numinrow, chs);
            }

            for (j = 0; j < 16; j++) {
                oldchs[j] = chs[j];
            }

            numinrow = 0;
        }

        count++;
        chs[numinrow++] = ch;
    }

    dump_row(count, numinrow, chs);

    if (numinrow != 0) {
        printf("%08lX:\n", count);
    }
}

RabbitMqClient::RabbitMqClient(const std::string &host_name, int port, const std::string &queue_name)
        :
        host_name_(host_name),
        queue_name_(queue_name),
        port_(port) {

    auto conn = amqp_new_connection();
    auto socket = amqp_tcp_socket_new(conn);
    if (!socket) {
        die("creating TCP socket");
    }

    auto status = amqp_socket_open(socket, host_name_.c_str(), port_);
    if (status) {
        die("opening TCP socket");
    }

    die_on_amqp_error(amqp_login(conn, "/", 0, 131072, 0, AMQP_SASL_METHOD_PLAIN,
                                 "guest", "guest"),
                      "Logging in");
    amqp_channel_open(conn, 1);
    amqp_get_rpc_reply(conn);

    amqp_bytes_t amq_queue_name {queue_name_.size(), (void *)queue_name_.c_str()};

    amqp_basic_consume(conn, 1, amq_queue_name, amqp_empty_bytes, 0, 1, 0,
                       amqp_empty_table);

    run(conn);
}
