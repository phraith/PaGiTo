#nullable enable

using NetMQ;
using NetMQ.Sockets;
using Vraith.Gisaxs.Core.RequestHandling;

namespace Vraith.Gisaxs.Core.Connection
{
    public class MajordomoClient : IDisposable
    {
        private readonly string _ip;
        private DealerSocket _client;
        private readonly TimeSpan _timeout;

        public MajordomoClient(string connectionString)
        {
            _ip = connectionString;
            _client = new DealerSocket(connectionString);
            _timeout = TimeSpan.FromMilliseconds(50000);
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
    }
}