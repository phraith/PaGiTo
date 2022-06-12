namespace GisaxsClient.Core.ImageStore
{
    public class Image
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public long Size { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public double[] Data { get; set; }
    }
}