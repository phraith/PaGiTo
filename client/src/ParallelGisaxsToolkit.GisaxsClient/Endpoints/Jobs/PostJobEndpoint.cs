using System.ComponentModel.DataAnnotations;
using System.Security.Claims;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpPost("/api/job")]
public class PostJobEndpoint : Endpoint<PostJobRequest, PostJobResponse>
{
    private readonly IImageStore _imageStore;
    private readonly IJobStore _jobStore;
    private readonly IRabbitMqPublisher _publisher;
    private readonly IDatabase _redisClient;
    private readonly IRequestResultDeserializer _requestResultDeserializer;
    private readonly INotifier _notifier;
    private readonly IHashComputer _hashComputer;

    public PostJobEndpoint(IImageStore imageStore, IHashComputer hashComputer, IJobStore jobStore,
        IRabbitMqPublisher publisher, IDatabase redisClient, IRequestResultDeserializer requestResultDeserializer,
        INotifier notifier)
    {
        _imageStore = imageStore;
        _jobStore = jobStore;
        _publisher = publisher;
        _redisClient = redisClient;
        _requestResultDeserializer = requestResultDeserializer;
        _notifier = notifier;
        _hashComputer = hashComputer;
    }

    public override async Task HandleAsync(PostJobRequest req, CancellationToken ct)
    {
        IRequestFactory factory = new RequestFactory(_hashComputer, _imageStore);
        string? clientId = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (clientId == null)
        {
            throw new InvalidOperationException("User connection does not exist!");
        }

        Request? request = factory.CreateRequest(req.JsonConfig, clientId);
        if (request == null)
        {
            throw new InvalidOperationException("Request creation failed!");
        }

        string? colormap = request.RequestInformation.MetaInformation.Colormap;
        byte[]? cachedResponse = _redisClient.StringGet(request.JobHash);
        if (cachedResponse is { Length: > 0 })
        {
            IRequestData resultData = _requestResultDeserializer.Deserialize(cachedResponse,
                request.RequestInformation.MetaInformation.Type, colormap);
            string serializedResult = resultData.Serialize();
            var job = new Job(request.JobId, clientId, DateTime.Now, DateTime.Now, serializedResult,
                req.JsonConfig);
            await _jobStore.Insert(job);
            await _notifier.Notify(request!.ClientId, request.RequestInformation.MetaInformation.Notification,
                request.JobId);
            await SendAsync(new PostJobResponse(request.JobId), 200, ct);
            return;
        }

        await _jobStore.Insert(new Job(request.JobId, clientId, DateTime.Now, null, null, req.JsonConfig));
        await _publisher.Publish(request);
        await SendAsync(new PostJobResponse(request.JobId), 201, ct);
    }
}

public sealed record PostJobResponse(string JobId);

public sealed record PostJobRequest
{
    [Required] public string JsonConfig { get; init; } = string.Empty;
}