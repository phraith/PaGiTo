#nullable enable

using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Options;
using StackExchange.Redis;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.RequestHandling;
using Vraith.Gisaxs.Utility.HashComputer;
using Vraith.Gisaxs.Utility.LineProfile;
using Vraith.GisaxsClient.Controllers;

namespace Vraith.GisaxsClient.Hubs
{
    public class MessageHub : Hub
    {
        private readonly IRequestHandler requestHandler;
        private readonly IHashComputer hashComputer;
        private readonly ConnectionMultiplexer connection;

        public MessageHub(IOptionsMonitor<ConnectionStrings> connectionStrings)
        {
            requestHandler = RequestHandlerFactory.CreateMajordomoRequestHandler(connectionStrings);
            hashComputer = HashComputerFactory.CreateSha256HashComputer();
            this.connection = ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis);
        }

        [Authorize]
        public async Task IssueJob(string stringRequest, string colormap)
        {
            IRequestFactory factory = new RequestFactory(hashComputer);
            Request? request = factory.CreateRequest(stringRequest, "ReceiveJobId");

            if (request == null)
            {
                return;
            }

            await Clients.All.SendAsync("ReceiveJobInfos", $"hash={request.InfoHash}");

            RequestResult? result = requestHandler.HandleRequest(request);
            if (result == null)
            {
                return;
            }

            await Clients.Caller.SendAsync(result.SignalREndpoint, result.DataAccessor, colormap);
        }

        [Authorize]
        public async Task GetProfiles(string stringRequest)
        {
            IRequestFactory factory = new RequestFactory(hashComputer);
            Request? request = factory.CreateRequest(stringRequest, "ProcessLineprofiles");

            if (request == null)
            {
                return;
            }

            await Clients.All.SendAsync("ReceiveJobInfos", $"hash={request.InfoHash}");

            RequestResult? result = requestHandler.HandleRequest(request);

            if (result == null)
            {
                return;
            }

            await Clients.Caller.SendAsync(result.SignalREndpoint, result.DataAccessor);
        }

        private static async Task<LineProfile?> GetLineprofile(IDatabase db, LineprofileType lpType,
            string basicDataHash, int invariantPixelPosition, int axisDimension)
        {
            string accessor = DataAccessor(lpType, basicDataHash, invariantPixelPosition);
            if (!db.KeyExists(accessor))
            {
                return null;
            }

            byte[] data = await db.StringGetAsync(accessor);

            if (data.Length != sizeof(double) * axisDimension)
            {
                return null;
            }

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