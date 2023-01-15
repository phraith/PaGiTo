using System.Security.Claims;
using Microsoft.AspNetCore.SignalR;

namespace ParallelGisaxsToolkit.GisaxsClient.Hubs
{
    // [Authorize]
    public class MessageHub : Hub
    {
        public override async Task OnConnectedAsync()
        {
            string? identifier = Context.User?.FindFirstValue(ClaimTypes.NameIdentifier);
            if (identifier != null)
            {
                await Groups.AddToGroupAsync(Context.ConnectionId, identifier);
            }

            await base.OnConnectedAsync();
        }
    }
}