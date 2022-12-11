#nullable enable

using Microsoft.Extensions.Options;
using NetMQ;
using Polly;
using Polly.Retry;
using StackExchange.Redis;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.Connection;

namespace Vraith.Gisaxs.Core.RequestHandling
{
    internal class MajordomoRequestHandler : IRequestHandler
    {
        private readonly IDatabase _db;
        private readonly RetryPolicy _retryPolicy;
        private readonly IOptionsMonitor<ConnectionStrings> _connectionStrings;

        public MajordomoRequestHandler(IOptionsMonitor<ConnectionStrings> connectionStrings)
        {
            _db = ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis).GetDatabase();
            _retryPolicy = Policy.Handle<TransientException>()
                .WaitAndRetry(retryCount: 3, sleepDurationProvider: i => TimeSpan.FromSeconds(50000));
            _connectionStrings = connectionStrings;
        }

        public RequestResult? HandleRequest(Request request)
        {
            if (_db.KeyExists(request.JobHash))
            {
                return new RequestResult(request.JobHash, request.DataAccessor);
            }

            RequestResult? response = null;

            var attempt = 0;
            _retryPolicy.Execute(() =>
            {
                using var client = new MajordomoClient(_connectionStrings.CurrentValue.GisaxsBackend);
                NetMQMessage msg = new();
                msg.Append(request.RawRequest);
                msg.Append(CreateTestImageData());
                Console.WriteLine($"Attempt {++attempt}");
                client.Send(request.RequestInformation.ClientInformation.JobType, msg);

                NetMQMessage? currentMessage = client.Receive(request.RequestInformation.ClientInformation.JobType);
                if (currentMessage == null || currentMessage.IsEmpty)
                {
                    throw new TransientException();
                }

                NetMQFrame frame = currentMessage.First();

                if (frame == NetMQFrame.Empty)
                {
                    throw new TransientException();
                }

                response = request.RequestInformation.ClientInformation.JobType switch
                {
                    "simulation" => HandleSimulationResult(frame.ToByteArray(), request.JobHash, request.DataAccessor),
                    "fitting" => null,
                    _ => throw new TransientException()
                };
            });

            return response;
        }

        private RequestResult? HandleSimulationResult(byte[] contentFrameData, string hash, string dataAccessor)
        {
            if (contentFrameData.Length <= 0)
            {
                return null;
            }

            _db.StringSet($"{hash}", contentFrameData);
            return new RequestResult(hash, dataAccessor);
        }

        private static byte[] CreateTestImageData()
        {
            var lp0 = new double[] { 1.0, 2.0, 1.0, 2.0, 1.0 }.SelectMany(value => BitConverter.GetBytes(value))
                .ToArray();
            var offsetLp0 = new int[] { 0, 1, 2, 3, 4 }.SelectMany(value => BitConverter.GetBytes(value)).ToArray();
            var lp0PxCount = BitConverter.GetBytes(5);
            var lpCount = BitConverter.GetBytes(1);
            return lpCount.Concat(lp0PxCount).Concat(lp0).Concat(offsetLp0).ToArray();
        }
    }
}