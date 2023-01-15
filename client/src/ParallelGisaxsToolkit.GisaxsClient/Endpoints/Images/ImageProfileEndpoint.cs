using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Images;

[Authorize]
[HttpPost("/api/image/profile")]
public class ImageProfileEndpoint : Endpoint<ImageProfileRequest, ImageProfileResponse>
{
    private readonly IImageStore _imageStore;

    public ImageProfileEndpoint(IImageStore imageStore)
    {
        _imageStore = imageStore;
    }

    public override async Task HandleAsync(ImageProfileRequest request, CancellationToken ct)
    {
        int id = request.Target.Id;
        SimulationTarget target = request.Target.Target;
        ImageProfileResponse response = await CreateResponse(id, target.Start, target.End);
        await SendAsync(response, cancellation: ct);
    }

    private async Task<ImageProfileResponse> CreateResponse(int id, DetectorPosition start, DetectorPosition end)
    {
        if (start.X == 0 && start.Y == end.Y)
        {
            double[] horizontalProfile = await _imageStore.GetHorizontalProfile(id, start.X, end.X, start.Y);
            double[] horizontalLogData = horizontalProfile.Select(x => Math.Log(x + 1)).Reverse().ToArray();
            return new ImageProfileResponse(horizontalLogData, horizontalLogData.Length, 1);
        }

        double[] verticalProfile = await _imageStore.GetVerticalProfile(id, start.Y, end.Y, start.X);
        double[] verticalLogData = verticalProfile.Select(x => Math.Log(x + 1)).Reverse().ToArray();
        return new ImageProfileResponse(verticalLogData, 1, verticalLogData.Length);
    }
}

public record ImageProfileResponse(double[] ModifiedData, int Width, int Height);

public record ImageProfileRequest(SimulationTargetWithId Target)
{
    public ImageProfileRequest() : this(SimulationTargetWithId.Empty)
    {
    }
}