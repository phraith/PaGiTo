using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images;
using ParallelGisaxsToolkit.Gisaxs.Utility.ImageTransformations;
using ImageInfo = SixLabors.ImageSharp.ImageInfo;

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

        await SendAsync(new GetImageResponse(colorizedImageAsBase64, image.Info.Width, image.Info.Height),
            cancellation: ct);
    }
}

public record GetImageResponse(string ImageAsBase64, int Width, int Height);

public record GetImageRequest(int Id, string Colormap)
{
    public GetImageRequest() : this(-1, string.Empty)
    {
    }
}