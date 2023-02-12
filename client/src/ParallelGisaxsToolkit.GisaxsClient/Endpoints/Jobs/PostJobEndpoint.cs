using System.Security.Claims;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling;
using ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpPost("/api/job")]
public class PostJobEndpoint : Endpoint<PostJobRequest, PostJobResponse>
{
    private readonly IImageStore _imageStore;
    private readonly IJobStore _jobStore;
    private readonly IRabbitMqPublisher _publisher;
    private readonly IHashComputer _hashComputer;

    public PostJobEndpoint(IImageStore imageStore, IHashComputer hashComputer, IJobStore jobStore,
        IRabbitMqPublisher publisher)
    {
        _imageStore = imageStore;
        _jobStore = jobStore;
        _publisher = publisher;
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

        await _jobStore.Insert(new Job(req.JsonConfig, request.JobId, clientId));
        await _publisher.Publish(request);
        await SendAsync(new PostJobResponse(request.JobId), 201, ct);
    }
}

public record PostJobResponse(string JobId);

public record PostJobRequest(string JsonConfig)
{
    public PostJobRequest() : this(string.Empty)
    {
    }
}