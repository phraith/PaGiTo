using System.ComponentModel.DataAnnotations;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.ResultStore;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpGet("/api/job/{jobId}")]
public class GetJobEndpoint : Endpoint<GetJobRequest, GetJobResponse>
{
    private readonly IResultStore _resultStore;

    public GetJobEndpoint(IResultStore resultStore)
    {
        _resultStore = resultStore;
    }

    public override async Task HandleAsync(GetJobRequest req, CancellationToken ct)
    {
        Result? result = await _resultStore.Get(req.JobId);

        if (result == null)
        {
            throw new InvalidOperationException("Job result does not exist!");
        }

        await SendAsync(new GetJobResponse(result.Data), cancellation: ct);
    }
}

public sealed record GetJobResponse(string Response);
public sealed record GetJobRequest
{
    [Required] public string JobId { get; init; } = string.Empty;
}