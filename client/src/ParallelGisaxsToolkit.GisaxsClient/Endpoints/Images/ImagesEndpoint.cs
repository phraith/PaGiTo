using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Images;

[Authorize]
[HttpGet("/api/images")]
public class ImagesEndpoint: EndpointWithoutRequest<ImagesResponse>
{
    private readonly IImageStore _imageStore;

    public ImagesEndpoint(IImageStore imageStore)
    {
        _imageStore = imageStore;
    }

    public override async Task HandleAsync(CancellationToken ct)
    {
        IEnumerable<ImageInfoWithId> images = await _imageStore.Get();
        await SendAsync(new ImagesResponse(images), cancellation: ct);
    }
}

public sealed record ImagesResponse(IEnumerable<ImageInfoWithId> ImageInfosWithId);