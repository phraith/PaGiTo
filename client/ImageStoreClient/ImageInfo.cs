// See https://aka.ms/new-console-template for more information

public record ImageInfo
{
    public int Id { get; set; }
    public string Name { get; set; }
    public long Size { get; set; }

    public ImageInfo(int id, string name, long size)
    {
        Id = id;
        Name = name;
        Size = size;
    }
}