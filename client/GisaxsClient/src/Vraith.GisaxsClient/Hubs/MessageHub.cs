using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Options;
using StackExchange.Redis;
using Vraith.Gisaxs.Configuration;
using Vraith.Gisaxs.Core.RequestHandling;
using Vraith.Gisaxs.Utility.HashComputer;

namespace Vraith.GisaxsClient.Hubs
{
    public class MessageHub : Hub
    {
        private readonly IRequestHandler _requestHandler;
        private readonly IHashComputer _hashComputer;

        public MessageHub(IOptionsMonitor<ConnectionStrings> connectionStrings)
        {
            _requestHandler = RequestHandlerFactory.CreateMajordomoRequestHandler(connectionStrings);
            _hashComputer = HashComputerFactory.CreateSha256HashComputer();
            ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis);
        }

        [Authorize]
        public async Task IssueJob(string stringRequest, string colormap)
        {
            IRequestFactory factory = new RequestFactory(_hashComputer);
            Request? request = factory.CreateRequest(stringRequest, "ReceiveJobId");

            if (request == null)
            {
                return;
            }

            await Clients.All.SendAsync("ReceiveJobInfos", $"hash={request.InfoHash}");

            RequestResult? result = _requestHandler.HandleRequest(request);
            if (result == null)
            {
                return;
            }

            await Clients.Caller.SendAsync(result.SignalREndpoint, result.DataAccessor, colormap);
        }

        [Authorize]
        public async Task GetProfiles(string stringRequest)
        {
            IRequestFactory factory = new RequestFactory(_hashComputer);
            Request? request = factory.CreateRequest(stringRequest, "ProcessLineprofiles");

            if (request == null)
            {
                return;
            }

            await Clients.All.SendAsync("ReceiveJobInfos", $"hash={request.InfoHash}");

            RequestResult? result = _requestHandler.HandleRequest(request);

            if (result == null)
            {
                return;
            }

            await Clients.Caller.SendAsync(result.SignalREndpoint, result.DataAccessor);
        }
    }
}