using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;
using Serilog;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpGet("/api/jobs/count")]
public class JobsCountEndpoint : EndpointWithoutRequest<JobsCountResponse>
{
    private readonly IJobStore _jobStore;
    private readonly ILogger<JobsEndpoint> _logger;

    public JobsCountEndpoint(IJobStore jobStore, ILoggerFactory loggerFactory)
    {
        _jobStore = jobStore;
        _logger = loggerFactory.CreateLogger<JobsEndpoint>();
    }

    public override async Task HandleAsync(CancellationToken ct)
    {
        IEnumerable<long> counts = await _jobStore.Count();
        await SendAsync(new JobsCountResponse(counts.First()), cancellation: ct);
    }
}

public sealed record JobsCountResponse(long JobCount);