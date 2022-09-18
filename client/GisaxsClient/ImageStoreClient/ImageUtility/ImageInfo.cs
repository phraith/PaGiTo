namespace ImageStoreClient.ImageUtility
{
    public record ImageInfo
    {
        public string Name { get; }
        public int Width { get; }
        public int Height { get; }
        public long Size { get; }
        public ImageInfo(string name, int width, int height, long size)
        {
            Name = name;
            Width = width;
            Height = height;
            Size = size;
        }
    }
}