namespace Vraith.ImageStoreClient.ImageUtility
{
    public class Image
    {
        public ImageInfo Info { get; }
        public double[] Data { get; }
        public Image(ImageInfo info, double[] data)
        {
            Info = info;
            Data = data;
        }
    }
}