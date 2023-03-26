using System.ComponentModel.DataAnnotations;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.JobStore;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpGet("/api/jobs/{page}/{size}")]
public class GetJobRangeEndpoint : Endpoint<GetJobRangeRequest, GetJobRangeResponse>
{
    private readonly IJobStore _jobStore;
    private readonly ILogger<GetJobRangeEndpoint> _logger;

    public GetJobRangeEndpoint(IJobStore jobStore, ILoggerFactory loggerFactory)
    {
        _jobStore = jobStore;
        _logger = loggerFactory.CreateLogger<GetJobRangeEndpoint>();
    }

    public override async Task HandleAsync(GetJobRangeRequest request, CancellationToken ct)
    {
        int start = request.Page * request.Size;
        IEnumerable<Job> jobs = await _jobStore.Get();
        IEnumerable<Job> jobsInRange = jobs.ToArray().Skip(start).Take(request.Size + 1);
        await SendAsync(new GetJobRangeResponse(jobsInRange), cancellation: ct);
    }
}

public sealed record GetJobRangeResponse(IEnumerable<Job> Jobs);

public sealed record GetJobRangeRequest
{
    [Required] public int Page { get; init; } = -1;
    [Required] public int Size { get; init; } = -1;
}