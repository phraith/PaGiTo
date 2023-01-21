using System.Collections.Concurrent;
using Microsoft.AspNetCore.SignalR;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.GisaxsClient.Hubs;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

public class JobScheduler : IJobScheduler
{
    private readonly IRequestHandler _requestHandler;
    private readonly IHubContext<MessageHub> _notificationHub;
    private readonly ILogger<JobScheduler> _logger;
    private readonly ConcurrentDictionary<string, Task> _activeTasks;


    public JobScheduler(IRequestHandler requestHandler, IHubContext<MessageHub> notificationHub,
        ILogger<JobScheduler> logger)
    {
        _requestHandler = requestHandler;
        _notificationHub = notificationHub;
        _logger = logger;
        _activeTasks = new ConcurrentDictionary<string, Task>();
    }

    public void ScheduleJob(Request request, CancellationToken cancellationToken)
    {
        Task task = Task.Run(async () => await ProcessJob(request), cancellationToken);
        if (!_activeTasks.TryAdd(request.JobId, task))
        {
            throw new ArgumentException("Job registering failed!");
        }
    }

    private async Task ProcessJob(Request request)
    {
        _logger.LogInformation("Processing request {Request}!", request);
        RequestResult? result = _requestHandler.HandleRequest(request);
        _logger.LogInformation("Finished job with id {JobId}!", request.JobId);

        if (result == null)
        {
            _logger.LogCritical("Job with id {JobId} had an empty result!", request.JobId);
            throw new InvalidOperationException("Result is null!");
        }

        _logger.LogInformation("Sending notification {Notification} to client {ClientId}!", result.Notification,
            request.ClientId);
        
        if (!_activeTasks.TryRemove(request.JobId, out _))
        {
            throw new ArgumentException("Job removal failed!");
        }
        
        await _notificationHub.Clients.Group(request.ClientId).SendAsync(result.Notification, result.JobId);
    }
}