using Microsoft.AspNetCore.SignalR;
using ParallelGisaxsToolkit.Gisaxs.Core.Hubs;

namespace ParallelGisaxsToolkit.GisaxsClient;

public class MessageHubNotifier : INotifier
{
    private readonly IHubContext<MessageHub> _hubContext;

    public MessageHubNotifier(IHubContext<MessageHub> hubContext)
    {
        _hubContext = hubContext;
    }

    public async Task Notify(string target, string notificationType, string notification)
    {
        await _hubContext.Clients.Group(target).SendAsync(notificationType, notification);
    }
}