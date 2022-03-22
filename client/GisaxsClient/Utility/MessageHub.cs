using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using NetMQ;
using Newtonsoft.Json.Linq;
using Polly;
using Polly.Retry;
using RedisTest.Controllers;
using StackExchange.Redis;
using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace RedisTest
{
    [Authorize]
    public class MessageHub : Hub
    {
        private readonly IConnectionMultiplexer redis;
        private readonly RetryPolicy retryPolicy;

        public MessageHub(IConnectionMultiplexer redis)
        {
            this.redis = redis;
            this.retryPolicy = Policy.Handle<TransientException>()
                .WaitAndRetry(retryCount: 3, sleepDurationProvider: i => TimeSpan.FromSeconds(5));
        }

        public async Task IssueJob(string stringRequest)
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            JObject o = JObject.Parse(stringRequest);
            JToken meta = o["info"];
            JToken config = o["config"];
            var configJson = config.ToString();
            var metaInf = JsonSerializer.Deserialize<MetaInformation>(meta.ToString(), options);
            var hash = BitConverter.ToString(SHA256.HashData(Encoding.UTF8.GetBytes(configJson)));

            var db = redis.GetDatabase();
            if (db.KeyExists(hash))
            {
                await Clients.All.SendAsync("ReceiveJobId", $"hash={hash}&colorMapName={metaInf.ColormapName}");
                return;
            }

            NetMQMessage? result = null;
            

            using (var client = new MajordomoClient("tcp://127.0.0.1:5555"))
            {
                var attempt = 0;
                retryPolicy.Execute(() =>
                {
                    NetMQMessage msg = new();
                    msg.Append(configJson);
                    Console.WriteLine($"Attempt {++attempt}");
                    result = client.Send("sim", msg);
                });
            };

            if (result != null)
            {
                string data = result.First.ConvertToString();
                db.StringSet(hash, data);
                await Clients.All.SendAsync("ReceiveJobId", $"hash={hash}&colorMapName={metaInf.ColormapName}");
            }
            return;
        }
    }
}
