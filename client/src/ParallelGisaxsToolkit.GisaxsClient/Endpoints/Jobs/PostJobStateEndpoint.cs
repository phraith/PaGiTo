using System.ComponentModel.DataAnnotations;
using System.Security.Claims;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpPost("/api/job/state")]
public class PostJobStateEndpoint : Endpoint<PostJobStateRequest, PostJobStateResponse>
{
    private readonly IJobStore _jobStore;
    private readonly ILogger<JobsEndpoint> _logger;

    public PostJobStateEndpoint(IJobStore jobStore, ILoggerFactory loggerFactory)
    {
        _jobStore = jobStore;
        _logger = loggerFactory.CreateLogger<JobsEndpoint>();
    }

    public override async Task HandleAsync(PostJobStateRequest req, CancellationToken ct)
    {
        string? userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (userId == null)
        {
            throw new ArgumentNullException(nameof(userId), "User id is null!");
        }

        var job = await _jobStore.Get(req.JobId, userId, req.IncludeConfig, req.IncludeResult);
        await SendAsync(new PostJobStateResponse(job), 200, ct);
    }
}

public sealed record PostJobStateResponse(Job? Job);

public sealed record PostJobStateRequest
{
    [Required] public string JobId { get; init; } = string.Empty;
    public bool IncludeConfig { get; init; } = false;
    public bool IncludeResult { get; init; } = false;
}