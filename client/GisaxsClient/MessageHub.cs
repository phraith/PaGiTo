using Capnp;
using CapnpGen;
using Microsoft.AspNetCore.SignalR;
using NetMQ;
using NetMQ.Sockets;
using RedisTest.Controllers;
using StackExchange.Redis;
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace RedisTest
{
    public class MessageHub : Hub
    {
        private IConnectionMultiplexer redis;
        private SHA256 sha256Hash;

        public RequestSocket Socket { get; }
        public SubscriberSocket SubSocket { get; }

        public MessageHub(IConnectionMultiplexer redis)
        {
            this.redis = redis;
            sha256Hash = SHA256.Create();
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
            string instrumentation = JsonSerializer.Serialize(instrumentationConfig);
            string config = JsonSerializer.Serialize(gisaxsConfig);
            var hash = sha256Hash.ComputeHash(Encoding.UTF8.GetBytes(config).Concat(Encoding.UTF8.GetBytes(instrumentation)).ToArray());
            var hashStr = BitConverter.ToString(hash);
            var db = redis.GetDatabase();

            if (db.KeyExists(hashStr))
            {
                await Clients.All.SendAsync("ReceiveJobId", $"hash={hashStr}&colorMapName={request.Info.ColormapName}");
                return;
            }

            SerializedSimulationDescription descr = new SerializedSimulationDescription();
            descr.IsLast = false;
            descr.Timestamp = 1001;
            descr.InstrumentationData = instrumentation;
            descr.ConfigData = config;
            descr.ClientId = hashStr;

            Socket.SendFrame(CreateCapnpMessage(descr));
            var _ = Socket.ReceiveFrameBytes();

            var id = SubSocket.ReceiveFrameBytes();
            var data = SubSocket.ReceiveFrameBytes();
            string jobHash = Encoding.UTF8.GetString(id, 0, id.Length);

            DataEntry entry = new DataEntry() { Data = data, Height = instrumentationConfig.detector.resolution[1], Width = instrumentationConfig.detector.resolution[0] };

            db.StringSet(jobHash, JsonSerializer.Serialize(entry));

            await Clients.All.SendAsync("ReceiveJobId", $"hash={hashStr}&colorMapName={request.Info.ColormapName}");
            return;
        }



        private static byte[] CreateCapnpMessage(SerializedSimulationDescription descr)
        {
            var msg = MessageBuilder.Create();
            var root = msg.BuildRoot<SerializedSimulationDescription.WRITER>();
            descr.serialize(root);
            var mems = new MemoryStream();
            var pump = new FramePump(mems);
            pump.Send(msg.Frame);
            mems.Seek(0, SeekOrigin.Begin);
            return mems.ToArray();
        }
    }
}
