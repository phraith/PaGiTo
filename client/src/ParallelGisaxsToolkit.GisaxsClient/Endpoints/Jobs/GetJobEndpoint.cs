using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using StackExchange.Redis;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Jobs;

[Authorize]
[HttpGet("/api/job/{id}")]
public class GetJobEndpoint : Endpoint<GetJobRequest, GetJobResponse>
{
    private readonly IDatabase _redisClient;

    public GetJobEndpoint(IDatabase redisClient)
    {
        _redisClient = redisClient;
    }

    public override async Task HandleAsync(GetJobRequest req, CancellationToken ct)
    {
        string? result = await _redisClient.StringGetAsync(req.Id.ToString());

        if (result == null)
        {
            throw new InvalidOperationException("Job result does not exist!");
        }
        
        await SendAsync(new GetJobResponse(result), cancellation: ct);
    }
}

public record GetJobResponse(string Response);

public record GetJobRequest(string Id)
{
    public GetJobRequest() : this(string.Empty)
    {
    }
}