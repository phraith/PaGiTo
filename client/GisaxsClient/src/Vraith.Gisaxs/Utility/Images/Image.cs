namespace Vraith.Gisaxs.Utility.Images
{
    public static class IntensityNormalizer
    {
        public static byte[] Normalize(IReadOnlyList<double> intensities)
        {
            var maxIntensity = intensities.Max();
            Console.WriteLine($"Max intenity {maxIntensity}");
            byte[] normalizedImage = intensities.Select(x => Normalize(x, maxIntensity)).ToArray();
            return normalizedImage;
        }

        private static byte Normalize(double intensity, double max)
        {
            double logmax = Math.Log(max);
            double logmin = Math.Log(Math.Max(2, 1e-10 * max));

            double logval = Math.Log(intensity);
            logval /= logmax - logmin;
            return (byte)(logval * 255.0);
        }
    }

    public class Image
    {
        public ImageInfo Info { get; }
        public IReadOnlyList<double> RowWiseData { get; }
        public IReadOnlyList<double> ColumnWiseData { get; }
        public byte[] GreyScaleData { get; }

        public Image(ImageInfo info, IReadOnlyList<double> rowWiseData, IReadOnlyList<double> columnWiseData,
            byte[] greyscaleData)
        {
            Info = info;
            RowWiseData = rowWiseData;
            ColumnWiseData = columnWiseData;
            GreyScaleData = greyscaleData.ToArray();
        }
    }

    public record SimpleImage(ImageInfo Info, IReadOnlyList<byte> GreyscaleData);
}