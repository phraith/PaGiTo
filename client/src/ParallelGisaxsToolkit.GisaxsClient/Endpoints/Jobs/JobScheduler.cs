using System.Collections.Concurrent;
using Microsoft.AspNetCore.SignalR;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.GisaxsClient.Hubs;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

public class JobScheduler : IJobScheduler, IDisposable
{
    private readonly IRequestHandler _requestHandler;
    private readonly IHubContext<MessageHub> _notificationHub;
    private readonly ConcurrentDictionary<string, Task> _activeTasks;


    public JobScheduler(IRequestHandler requestHandler, IHubContext<MessageHub> notificationHub)
    {
        _requestHandler = requestHandler;
        _notificationHub = notificationHub;
        _activeTasks = new ConcurrentDictionary<string, Task>();
    }

    public void ScheduleJob(Request request, CancellationToken cancellationToken)
    {
        Task task = Task.Factory.StartNew(async () => await ProcessJob(request), cancellationToken);
        if (!_activeTasks.TryAdd(request.JobId, task))
        {
            throw new InvalidOperationException("Task could not be activated");
        }
        task.ContinueWith(_ => _activeTasks.Remove(request.JobId, out var _), cancellationToken);
    }
    
    private async Task ProcessJob(Request request)
    {
        RequestResult? result = _requestHandler.HandleRequest(request);
        if (result == null)
        {
            throw new InvalidOperationException("Result is null!");
        }

        await _notificationHub.Clients.Group(request.ClientId).SendAsync(result.Notification, result.JobId);
    }

    public void Dispose()
    {
        Task.WaitAll(_activeTasks.Values.ToArray());
    }
}