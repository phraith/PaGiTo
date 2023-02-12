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

public record GetJobResponse(string Response);

public record GetJobRequest(string JobId)
{
    public GetJobRequest() : this(string.Empty)
    {
    }
}