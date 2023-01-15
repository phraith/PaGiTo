namespace ParallelGisaxsToolkit.Gisaxs.Utility.Images
{
    public class Image
    {
        public ImageInfo Info { get; }
        public IReadOnlyList<double> RowWiseData { get; }
        public IReadOnlyList<double> ColumnWiseData { get; }
        public byte[] GreyScaleData { get; }
        public static readonly Image Empty = CreateEmptyImage();
        public Image(ImageInfo info, IReadOnlyList<double> rowWiseData, IReadOnlyList<double> columnWiseData,
            byte[] greyscaleData)
        {
            Info = info;
            RowWiseData = rowWiseData;
            ColumnWiseData = columnWiseData;
            GreyScaleData = greyscaleData.ToArray();
        }

        private static Image CreateEmptyImage()
        {
            ImageInfo emptyInfo = new ImageInfo(string.Empty, 0, 0, 0);
            return new Image(emptyInfo, Array.Empty<double>(), Array.Empty<double>(), Array.Empty<byte>());
        }
    }

    public record GreyScaleImage(ImageInfo Info, IReadOnlyList<byte> GreyscaleData);
}