using ImageStoreClient.ImageUtility;
using System.Text.Json;

namespace GisaxsClient.Core.ImageStore
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
