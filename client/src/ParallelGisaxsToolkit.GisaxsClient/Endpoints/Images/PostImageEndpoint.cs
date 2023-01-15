using FastEndpoints;
using Microsoft.AspNetCore.Authorization;
using ParallelGisaxsToolkit.Gisaxs.Core.ImageStore;
using ParallelGisaxsToolkit.Gisaxs.Utility.Images;

namespace ParallelGisaxsToolkit.GisaxsClient.Endpoints.Images;

// [Authorize]
[AllowAnonymous]
[HttpPost("/api/image")]
public class PostImageEndpoint : Endpoint<PostImageRequest>
{
    private readonly IImageStore _imageStore;

    public PostImageEndpoint(IImageStore imageStore)
    {
        _imageStore = imageStore;
    }

    public override async Task HandleAsync(PostImageRequest request, CancellationToken ct)
    {
        await _imageStore.Insert(request.Image);
    }
}

public record PostImageRequest(Image Image)
{
    public PostImageRequest() : this(Image.Empty)
    {
    }
}