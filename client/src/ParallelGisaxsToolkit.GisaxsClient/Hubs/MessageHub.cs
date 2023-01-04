#nullable enable

using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Options;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer;
using ParallelGisaxsToolkit.Gisaxs.Utility.LineProfile;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient.Hubs
{

    [Authorize]
    public class MessageHub : Hub
    {
        private readonly IImageStore _imageStore;
        private readonly IRequestHandler _requestHandler;
        private readonly IHashComputer _hashComputer;

        public MessageHub(IOptionsMonitor<ConnectionStrings> connectionStrings, IImageStore imageStore)
        {
            _imageStore = imageStore;
            _requestHandler = RequestHandlerFactory.CreateMajordomoRequestHandler(connectionStrings);
            _hashComputer = HashComputerFactory.CreateSha256HashComputer();
            ConnectionMultiplexer.Connect(connectionStrings.CurrentValue.Redis);
        }

        public async Task IssueJob(string stringRequest, string? colormapName)
        {
            IRequestFactory factory = new RequestFactory(_hashComputer, _imageStore);
            Request? request = factory.CreateRequest(stringRequest, "ReceiveJobId");

            if (request == null) { return; }

            RequestResult? result = _requestHandler.HandleRequest(request);
            if (result == null)
            {
                return;
            }

            if (colormapName != null)
            {
                await Clients.Caller.SendAsync(result.SignalREndpoint, result.DataAccessor, colormapName);
                return;
            }
            
            await Clients.Caller.SendAsync(result.SignalREndpoint, result.DataAccessor);
        }
    }
}
