using BitMiracle.LibTiff.Classic;

namespace ParallelGisaxsToolkit.Gisaxs.Utility.Images.ImageLoaders
{
    public class TifLoader : IImageLoader
    {
        public Image Load(string path)
        {
            using Tiff tif = Tiff.Open(path, "r");
            string? name = Path.GetFileNameWithoutExtension(path);
            FieldValue[]? value = tif.GetField(TiffTag.IMAGEWIDTH);
            int width = value[0].ToInt();

            value = tif.GetField(TiffTag.IMAGELENGTH);
            int height = value[0].ToInt();
            int scanlineSize = tif.ScanlineSize();

            double[] imageData = Enumerable.Range(0, height).SelectMany(i => 
            {
                byte[] buffer = new byte[scanlineSize];
                tif.ReadScanline(buffer, i); 

                double[] datad = new double[width];
                for(int j = 0; j < width; ++j)
                {
                    datad[j] = BitConverter.ToInt32(buffer, j * 4);
                }

                return datad; 
            }).ToArray();

            double[] imageDataTransposed = new double[imageData.Length];
            for (int i = 0; i < width * height; i++)
            {
                int mWidth = i % width;
                int dWidth = i / width;

                int newIndex = mWidth * height + dWidth;
                imageDataTransposed[newIndex] = imageData[i];
            }
            byte[] greyscaleImage = IntensityNormalizer.Normalize(imageData);
            return new Image(new ImageInfo(name, width, height, height * width * sizeof(double)), imageData, imageDataTransposed, greyscaleImage);
        }
    }
}
