using NetMQ;
using NetMQ.Sockets;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;

namespace ParallelGisaxsToolkit.Gisaxs.Core.Connection
{
    public class MajordomoClient : IDisposable
    {
        private readonly DealerSocket _client;
        private readonly TimeSpan _timeout;

        public MajordomoClient(string connectionString)
        {
            _client = new DealerSocket(connectionString);
            _timeout = TimeSpan.FromMilliseconds(200_000);
        }

        public void Dispose()
        {
            _client.Dispose();
        }

        public void Send(string serviceName, NetMQMessage message)
        {
            message.Push(serviceName);
            message.Push("MDPC01");
            message.PushEmptyFrame();
            _client.SendMultipartMessage(message);
        }

        public NetMQMessage? Receive(string serviceName)
        {
            NetMQMessage? reply = null;
            if (!_client.TryReceiveMultipartMessage(_timeout, ref reply))
            {
                return reply;
            }

            if (reply.FrameCount < 4)
            {
                throw new TransientException("[CLIENT ERROR] received a malformed reply");
            }

            NetMQFrame emptyFrame = reply.Pop();
            if (emptyFrame != NetMQFrame.Empty)
            {
                throw new TransientException(
                    $"[CLIENT ERROR] received a malformed reply expected empty frame instead of: {emptyFrame} ");
            }

            NetMQFrame header = reply.Pop(); // [MDPHeader] <- [service name][reply] OR ['mmi.service'][return code]

            if (header.ConvertToString() != "MDPC01")
            {
                throw new TransientException($"[CLIENT INFO] MDP Version mismatch: {header}");
            }

            NetMQFrame service = reply.Pop(); // [service name or 'mmi.service'] <- [reply] OR [return code]

            if (service.ConvertToString() != serviceName)
            {
                throw new TransientException($"[CLIENT INFO] answered by wrong service: {service.ConvertToString()}");
            }

            return reply;
        }
    }
}