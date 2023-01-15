using ParallelGisaxsToolkit.Gisaxs.Utility.Images;

namespace ParallelGisaxsToolkit.Gisaxs.Core.ImageStore
{
    public class ImageInfoWithId
    {
        public int Id { get; }
        public ImageInfo Info { get; }
        public ImageInfoWithId(int id, ImageInfo info)
        {
            Id = id;
            Info = info;
        }
    }
}
