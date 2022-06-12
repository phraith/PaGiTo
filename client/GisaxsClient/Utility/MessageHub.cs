using GisaxsClient;
using GisaxsClient.Controllers;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using NetMQ;
using StackExchange.Redis;
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

#nullable enable

namespace GisaxsClient.Utility
{
    //[Authorize]
    public class MessageHub : Hub
    {
        private readonly IRequestHandler requestHandler;
        private readonly IHashComputer hashComputer;

        public MessageHub()
        {
            requestHandler = new MajordomoRequestHandler();
            hashComputer = new Sha256HashComputer();
        }

        public async Task IssueJob(string stringRequest)
        {
            IRequestFactory factory = new RequestFactory(hashComputer);
            Request? request = factory.CreateRequest(stringRequest);

            if (request == null) { return; }

            await Clients.All.SendAsync("ReceiveJobInfos", $"hash={request.InfoHash}");

            RequestResult? result = requestHandler.HandleRequest(request);
            if (result == null) { return; }

            await Clients.All.SendAsync(result.Command, result.Body);
        }

        public async Task GetProfiles(string stringRequest)
        {
            //var profilesT = new List<LineProfile>();
            //Random r = new Random();
            //profilesT.Add(new LineProfile() { Data = Enumerable.Repeat(r.NextDouble(), 1679).ToArray() });
            //await Clients.All.SendAsync("ProcessLineprofiles", $"{{\"profiles\": {JsonSerializer.Serialize(profilesT)}}}");

            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            JsonNode? jsonNode = JsonNode.Parse(stringRequest);
            if (jsonNode == null) { return; }

            var info = jsonNode["profiles"];

            var config = jsonNode["config"];
            if (config == null || info == null) { return; }

            var profileInfos = new List<LineProfileInfo>();
            foreach (KeyValuePair<string, JsonNode?> node in info.AsObject())
            {
                var profile = node.Value.Deserialize<LineProfileInfo>(options);
                profileInfos.Add(profile);
            }



            var hash = hashComputer.Hash(config.ToString());
            IDatabase db = RedisConnectorHelper.Connection.GetDatabase();

            var keyWidth = $"{hash}-width";
            var keyHeight = $"{hash}-height";


            if (!db.KeyExists(keyWidth) || !db.KeyExists(keyHeight)) { return; }

            string heightAsString = await db.StringGetAsync(keyHeight);
            string widthAsString = await db.StringGetAsync(keyWidth);

            int height = int.Parse(heightAsString);
            int width = int.Parse(widthAsString);

            //byte[] data = db.StringGet(hash);

            //int x = BitConverter.ToInt32(data, 0);
            //int y = BitConverter.ToInt32(data, sizeof(int));

            //int dataStart = 2 * sizeof(int) + x * y;
            //byte[] relevantData = data[dataStart..];

            var profiles = new List<LineProfile>();
            foreach (var profileInfo in profileInfos)
            {
                var start = profileInfo.AbsoluteStart(width, height);
                var end = profileInfo.AbsoluteEnd(width, height);

                if ((int)start.X == (int)end.X)
                {
                    var profileData = new double[height];
                    byte[] data = await db.StringGetAsync($"{hash}-v-{(int)start.X}");
                    for (int i = 0; i < height; ++i)
                    {
                        int startIndex = i * sizeof(double);
                        profileData[i] = Math.Log(BitConverter.ToDouble(data, startIndex));
                    }
                    profiles.Add(new LineProfile { Data = profileData });
                }
                else if ((int)start.Y == (int)end.Y)
                {
                    var profileData = new double[width];
                    byte[] data = await db.StringGetAsync($"{hash}-h-{(int)start.Y}");
                    for (int i = 0; i < width; ++i)
                    {
                        int startIndex = i * sizeof(double);
                        profileData[i] = Math.Log(BitConverter.ToDouble(data, startIndex));
                    }
                    profiles.Add(new LineProfile { Data = profileData });
                }
                else
                {
                    return;
                }
            }
            await Clients.All.SendAsync("ProcessLineprofiles", $"{{\"profiles\": {JsonSerializer.Serialize(profiles)}}}");
        }

        internal class LineProfileInfo
        {
            public Coordinate StartRel { get; set; }
            public Coordinate EndRel { get; set; }


            public Coordinate AbsoluteStart(int width, int height)
            {
                return new Coordinate { X = StartRel.X * width, Y = StartRel.Y * height };
            }

            public Coordinate AbsoluteEnd(int width, int height)
            {
                return new Coordinate { X = EndRel.X * width, Y = EndRel.Y * height };
            }
        }

        internal class LineProfile
        {
            public double[] Data { get; set; }
        }

        internal class Coordinate
        {
            public double X { get; set; }
            public double Y { get; set; }
        }
    }
}
