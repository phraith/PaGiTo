using System.Text.Json;
using Microsoft.Extensions.Options;
using NetMQ;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.Connection;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling
{
    public class MajordomoRequestHandler : IRequestHandler
    {
        private readonly IDatabase _db;
        private readonly IOptionsMonitor<ConnectionStrings> _connectionStrings;
        private readonly IRequestResultDeserializer _requestResultDeserializer;

        public MajordomoRequestHandler(IOptionsMonitor<ConnectionStrings> connectionStrings, IDatabase redisClient)
        {
            _db = redisClient;
            _connectionStrings = connectionStrings;
            _requestResultDeserializer = new RequestResultDeserializer();
        }

        public RequestResult? HandleRequest(Request request)
        {
            IReadOnlyList<byte> data = ProcessRequest(request).GetAwaiter().GetResult();
            RequestResult? response = HandleResult(data.ToArray(), request);
            return response;
        }

        private async Task<IReadOnlyList<byte>> ProcessRequest(Request request)
        {
            if (_db.KeyExists(request.JobHash))
            {
                byte[]? bytes = await _db.StringGetAsync(request.JobHash);
                return bytes!;
            }

            string serviceName = request.RequestInformation.MetaInformation.Type.ToString().ToLowerInvariant();
            using MajordomoClient client = new MajordomoClient(_connectionStrings.CurrentValue.GisaxsBackend);
            NetMQMessage msg = new();
            msg.Append(request.RawRequest);
            msg.Append(request.ImageDataForFitting);
            client.Send(serviceName, msg);

            NetMQMessage? currentMessage = client.Receive(serviceName);
            if (currentMessage == null || currentMessage.IsEmpty)
            {
                throw new TransientException();
            }

            NetMQFrame frame = currentMessage.First();

            if (frame == NetMQFrame.Empty)
            {
                throw new TransientException();
            }

            byte[] data = frame.ToByteArray();
            _db.StringSet($"{request.JobHash}", data);

            return data;
        }

        private RequestResult? HandleResult(byte[] contentFrameData, Request request)
        {
            if (contentFrameData.Length <= 0)
            {
                return null;
            }

            string? colormap = request.RequestInformation.MetaInformation.Colormap;

            var resultData = _requestResultDeserializer.Deserialize(contentFrameData, colormap);
            var serialized = JsonSerializer.Serialize(
                resultData, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });

            _db.StringSet(request.JobId, serialized);
            return new RequestResult(request.JobId, request.RequestInformation.MetaInformation.Notification);
        }
    }
}