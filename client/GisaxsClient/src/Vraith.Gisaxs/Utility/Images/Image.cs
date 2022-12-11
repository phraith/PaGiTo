namespace Vraith.Gisaxs.Utility.Images
{
    public class Image
    {
        public ImageInfo Info { get; }
        public IReadOnlyList<double> RowWiseData { get; }
        public IReadOnlyList<double> ColumnWiseData { get; }
        public Image(ImageInfo info, IReadOnlyList<double> rowWiseData, IReadOnlyList<double> columnWiseData)
        {
            Info = info;
            RowWiseData = rowWiseData;
            ColumnWiseData = columnWiseData;
        }
    }
}