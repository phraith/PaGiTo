#nullable enable

using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Options;
using StackExchange.Redis;
using Vraith.GisaxsClient.Controllers;
using Vraith.GisaxsClient.Core.RequestHandling;
using Vraith.GisaxsClient.Utility.HashComputer;
using Vraith.GisaxsClient.Utility.LineProfile;

namespace Vraith.GisaxsClient.Hubs
{

    [Authorize]
    public class MessageHub : Hub
    {
        private readonly IRequestHandler requestHandler;
        private readonly IHashComputer hashComputer;
        private readonly ConnectionMultiplexer connection;

        public MessageHub(IOptionsMonitor<ConnectionStrings> connectionStrings)
        {
            requestHandler = new MajordomoRequestHandler(connectionStrings);
            hashComputer = new Sha256HashComputer();
            this.connection = ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis);
        }

        public async Task IssueJob(string stringRequest)
        {
            IRequestFactory factory = new RequestFactory(hashComputer);
            Request? request = factory.CreateRequest(stringRequest);

            if (request == null) { return; }

            await Clients.All.SendAsync("ReceiveJobInfos", $"hash={request.InfoHash}");

            RequestResult? result = requestHandler.HandleRequest(request);
            if (result == null) { return; }

            await Clients.Caller.SendAsync(result.Command, result.Body);
        }

        public async Task GetProfiles(string stringRequest)
        {
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
                LineProfileInfo? profile = node.Value.Deserialize<LineProfileInfo>(options);
                if (profile != null) { profileInfos.Add(profile); }
            }

            var hash = hashComputer.Hash(config.ToString());
            IDatabase db = connection.GetDatabase();

            var keyWidth = $"{hash}-width";
            var keyHeight = $"{hash}-height";


            if (!db.KeyExists(keyWidth) || !db.KeyExists(keyHeight)) { return; }

            string heightAsString = await db.StringGetAsync(keyHeight);
            string widthAsString = await db.StringGetAsync(keyWidth);

            int height = int.Parse(heightAsString);
            int width = int.Parse(widthAsString);

            var profiles = new List<LineProfile>();
            foreach (var profileInfo in profileInfos)
            {
                var start = profileInfo.AbsoluteStart(width, height);
                var end = profileInfo.AbsoluteEnd(width, height);

                if ((int)start.X == (int)end.X)
                {
                    var verticalLp = await GetLineprofile(db, LineprofileType.Vertical, hash, (int)start.X, height);
                    if (verticalLp != null)
                    {
                        profiles.Add(verticalLp);
                    }
                }
                else if ((int)start.Y == (int)end.Y)
                {
                    var horizontalLp = await GetLineprofile(db, LineprofileType.Horizontal, hash, (int)start.Y, width);
                    if (horizontalLp != null)
                    {
                        profiles.Add(horizontalLp);
                    }
                }
            }

            await Clients.Caller.SendAsync("ProcessLineprofiles", $"{{\"profiles\": {JsonSerializer.Serialize(profiles)}}}");
        }

        private static async Task<LineProfile?> GetLineprofile(IDatabase db, LineprofileType lpType, string basicDataHash, int invariantPixelPosition, int axisDimension)
        {
            string accessor = DataAccessor(lpType, basicDataHash, invariantPixelPosition);
            if (!db.KeyExists(accessor)) { return null; }
            byte[] data = await db.StringGetAsync(accessor);

            if (data.Length != sizeof(double) * axisDimension) { return null; }
            var profileData = new double[axisDimension];
            for (int i = 0; i < axisDimension; ++i)
            {
                int startIndex = i * sizeof(double);
                profileData[i] = Math.Log(BitConverter.ToDouble(data, startIndex));
            }
            return new LineProfile(profileData);
        }

        private static string DataAccessor(LineprofileType lpType, string basicDataHash, int invariantPixelPosition)
        {
            return lpType switch
            {
                LineprofileType.Vertical => $"{basicDataHash}-v-{invariantPixelPosition}",
                LineprofileType.Horizontal => $"{basicDataHash}-h-{invariantPixelPosition}",
                _ => throw new NotImplementedException(nameof(lpType)),
            };
        }
    }
}
