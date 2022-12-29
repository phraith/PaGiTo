using ParallelGisaxsToolkit.Gisaxs.Utility.Images;

namespace ParallelGisaxsToolkit.Gisaxs.Core.ImageStore
{
    public class ImageInfoDto
    {
        public int Id { get; }
        public ImageInfo Info { get; }
        public ImageInfoDto(int id, ImageInfo info)
        {
            Id = id;
            Info = info;
        }
    }
}
