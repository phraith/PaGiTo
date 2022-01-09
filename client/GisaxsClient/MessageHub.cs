using GisaxsClient.Utility;
using Microsoft.AspNetCore.SignalR;
using NetMQ;
using NetMQ.Sockets;
using RedisTest.Controllers;
using StackExchange.Redis;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace RedisTest
{
    public class MessageHub : Hub
    {
        private readonly IConnectionMultiplexer redis;

        public RequestSocket Socket { get; }
        public SubscriberSocket SubSocket { get; }

        public MessageHub(IConnectionMultiplexer redis)
        {
            this.redis = redis;
            Socket = new RequestSocket("tcp://127.0.0.1:5558");
            SubSocket = new SubscriberSocket("tcp://127.0.0.1:5559");
            SubSocket.SubscribeToAnyTopic();
        }

        ~MessageHub()
        {
            Socket.Close();
            SubSocket.Close();
        }

        public async Task IssueJob(string stringRequest)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            var request = JsonSerializer.Deserialize<GisaxsConfigWithMetaInformation>(stringRequest, options);
            if (request.Config == null || request.Config.shapes == null) { return; }


            var gisaxsConfig = GisaxsConfigCreator.CreateValidConfigFromFormDataConfig(request.Config);
            var instrumentationConfig = GisaxsConfigCreator.CreateValidInstrumentationConfigFromFormData(request.Config.instrumentation);
            IGisaxsMessage message = new CapnpGisaxsMessage(gisaxsConfig, instrumentationConfig);

            var db = redis.GetDatabase();
            if (db.KeyExists(message.ID))
            {
                await Clients.All.SendAsync("ReceiveJobId", $"hash={message.ID}&colorMapName={request.Info.ColormapName}");
                return;
            }

            Socket.SendFrame(message.Message);
            var _ = Socket.ReceiveFrameBytes();

            var id = SubSocket.ReceiveFrameBytes();
            var data = SubSocket.ReceiveFrameBytes();
            string jobHash = Encoding.UTF8.GetString(id, 0, id.Length);

            DataEntry entry = new() { Data = data, Height = instrumentationConfig.detector.resolution[1], Width = instrumentationConfig.detector.resolution[0] };

            db.StringSet(jobHash, JsonSerializer.Serialize(entry));

            await Clients.All.SendAsync("ReceiveJobId", $"hash={message.ID}&colorMapName={request.Info.ColormapName}");
            return;
        }
    }
}
