using Vraith.Gisaxs.Utility.Images;

namespace Vraith.Gisaxs.Core.ImageStore
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
