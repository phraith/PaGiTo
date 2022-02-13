using ConnectioniUtility.ConnectionUtility.Majordomo;
using GisaxsClient.Utility;
using Microsoft.AspNetCore.SignalR;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json.Linq;
using RedisTest.Controllers;
using StackExchange.Redis;
using System;
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
        }

        ~MessageHub()
        {
        }

        public async Task IssueJob(string stringRequest)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            JObject o = JObject.Parse(stringRequest);
            JToken meta = o["info"];
            var metaInf = JsonSerializer.Deserialize<MetaInformation>(meta.ToString(), options);
            var hash = BitConverter.ToString(SHA256.HashData(Encoding.UTF8.GetBytes(stringRequest)));

            using (var g = new MajordomoClient("tcp://127.0.0.1:5555", true))
            {
                NetMQMessage msg = new NetMQMessage();
                msg.Append(stringRequest);
                NetMQMessage result = g.Send("sim", msg);

                string data = result.First.ConvertToString();

                var db = redis.GetDatabase();
                if (db.KeyExists(hash))
                {
                    await Clients.All.SendAsync("ReceiveJobId", $"hash={hash}&colorMapName={metaInf.ColormapName}");
                    return;
                }


                db.StringSet(hash, data);

                await Clients.All.SendAsync("ReceiveJobId", $"hash={hash}&colorMapName={metaInf.ColormapName}");
            }

            await Clients.All.SendAsync("ReceiveJobId", $"hash={hash}&colorMapName={metaInf.ColormapName}");
        }
    }
}
