using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpGet("/api/jobs")]
public class JobsEndpoint : EndpointWithoutRequest<JobsResponse>
{
    private readonly IJobStore _jobStore;

    public JobsEndpoint(IJobStore jobStore)
    {
        _jobStore = jobStore;
    }

    public override async Task HandleAsync(CancellationToken ct)
    {
        IEnumerable<Job> jobs = await _jobStore.Get();
        await SendAsync(new JobsResponse(jobs), cancellation: ct);
    }
}

public record JobsResponse(IEnumerable<Job> Jobs);