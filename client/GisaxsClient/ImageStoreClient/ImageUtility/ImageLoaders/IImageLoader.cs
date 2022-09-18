namespace ImageStoreClient.ImageUtility.ImageLoaders
{
    public interface IImageLoader
    {
        Image Load(string path);
    }
}