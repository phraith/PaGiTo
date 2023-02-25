using System.ComponentModel.DataAnnotations;
using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Configuration;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images;
using ParallelGisaxsToolkit.Gisaxs.Utility.ImageTransformations;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Images;

[Authorize]
[HttpGet("/api/image/{id}/{colormap}")]
public class GetImageEndpoint : Endpoint<GetImageRequest, GetImageResponse>
{
    private readonly IImageStore _imageStore;

    public GetImageEndpoint(IImageStore imageStore)
    {
        _imageStore = imageStore;
    }

    public override async Task HandleAsync(GetImageRequest request, CancellationToken ct)
    {
        GreyScaleImage? image = await _imageStore.Get(request.Id);
        if (image == null)
        {
            throw new InvalidOperationException("Image does not exist!");
        }

        string colorizedImageAsBase64 = AppearanceModifier.ApplyColorMap(image.GreyscaleData.ToArray(),
            image.Info.Width,
            image.Info.Height, false, request.Colormap);

        await SendAsync(new GetImageResponse(colorizedImageAsBase64), cancellation: ct);
    }
}

public sealed record GetImageResponse(string ImageAsBase64);

public sealed record GetImageRequest
{
    [Required] public int Id { get; init; } = -1;
    [Required] public string Colormap { get; init; } = string.Empty;
}