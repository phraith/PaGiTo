using System.Security.Claims;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core;
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
    private readonly IProducer _producer;
    private readonly IHashComputer _hashComputer;

    public PostJobEndpoint(IImageStore imageStore, IHashComputer hashComputer, IJobStore jobStore,
        IProducer producer)
    {
        _imageStore = imageStore;
        _jobStore = jobStore;
        _producer = producer;
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

        await _jobStore.Insert(new Job(new JobInfo(req.JsonConfig), null, clientId));
        _producer.Produce(request);
        await SendAsync(new PostJobResponse(request.JobHash), 201, ct);
    }
}

public record PostJobResponse(string JobHash);

public record PostJobRequest(string JsonConfig)
{
    public PostJobRequest() : this(string.Empty)
    {
    }
}