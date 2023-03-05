using ImageInfo = ParallelGisaxsToolkit.Gisaxs.Utility.Images.ImageInfo;

namespace ParallelGisaxsToolkit.Gisaxs.Core.ImageStore
{
    public record ImageInfoWithId(int Id, ImageInfo Info);
}