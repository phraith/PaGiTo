using ScatterStore.Controllers;

namespace ScatterStore
{
    public class Image
    {
        public ImageInfo Info { get; set; }
        public double[] Data { get; set; }
        public Image(ImageInfo info, double[] data)
        {
            Info = info;
            Data = data;
        }
    }
}