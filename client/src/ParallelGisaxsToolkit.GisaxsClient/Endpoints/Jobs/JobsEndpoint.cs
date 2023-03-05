using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using Serilog;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpGet("/api/jobs")]
public class JobsEndpoint : EndpointWithoutRequest<JobsResponse>
{
    private readonly IJobStore _jobStore;
    private readonly ILogger<JobsEndpoint> _logger;

    public JobsEndpoint(IJobStore jobStore, ILoggerFactory loggerFactory)
    {
        _jobStore = jobStore;
        _logger = loggerFactory.CreateLogger<JobsEndpoint>();
    }

    public override async Task HandleAsync(CancellationToken ct)
    {
        IEnumerable<Job> jobs = await _jobStore.Get();
        await SendAsync(new JobsResponse(jobs), cancellation: ct);
    }
}

public record JobsResponse(IEnumerable<Job> Jobs);