namespace ScatterStore.Controllers
{
    public record ImageInfo
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public long Size { get; set; }
    }
}