using NetMQ;
using NetMQ.Sockets;
using System;

namespace RedisTest
{
    public class MajordomoClient : IDisposable
    {
        private readonly string ip;
        private DealerSocket client;
        private readonly TimeSpan timeout;

        public MajordomoClient(string connectionString)
        {
            this.ip = connectionString;
            this.client = new DealerSocket(connectionString);
            this.timeout = TimeSpan.FromMilliseconds(5000);
        }

        public void Dispose()
        {
            client.Dispose();
        }

        public NetMQMessage Send(string serviceName, NetMQMessage message)
        {
            message.Push(serviceName);
            message.Push("MDPC01");
            message.PushEmptyFrame();
            client.SendMultipartMessage(message);

            NetMQMessage? reply = null;
            if (client.TryReceiveMultipartMessage(timeout, ref reply))
            {
                if (reply.FrameCount < 4)
                    throw new TransientException("[CLIENT ERROR] received a malformed reply");

                var emptyFrame = reply.Pop();
                if (emptyFrame != NetMQFrame.Empty)
                {
                    throw new TransientException($"[CLIENT ERROR] received a malformed reply expected empty frame instead of: { emptyFrame } ");
                }
                var header = reply.Pop(); // [MDPHeader] <- [service name][reply] OR ['mmi.service'][return code]

                if (header.ConvertToString() != "MDPC01")
                    throw new TransientException($"[CLIENT INFO] MDP Version mismatch: {header}");

                var service = reply.Pop(); // [service name or 'mmi.service'] <- [reply] OR [return code]

                if (service.ConvertToString() != serviceName)
                    throw new TransientException($"[CLIENT INFO] answered by wrong service: {service.ConvertToString()}");

                return reply;
            }

            client = new DealerSocket(ip);

            throw new TransientException();
        }
    }
}