#nullable enable

using Microsoft.Extensions.Options;
using NetMQ;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.Connection;
using Polly;
using Polly.Retry;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling
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
                msg.Append(request.ImageDataForFitting);
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
                    "fitting" => HandleFittingResult(frame.ToByteArray(), request.JobHash, request.DataAccessor),
                    _ => throw new TransientException()
                };
            });

            return response;
        }

        private RequestResult HandleFittingResult(byte[] toByteArray, string requestJobHash, string requestDataAccessor)
        {
            return null;
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
    }
}