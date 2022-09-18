using BitMiracle.LibTiff.Classic;

namespace ImageStoreClient.ImageUtility.ImageLoaders
{
    internal class TifLoader : IImageLoader
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

            double[] data = Enumerable.Range(0, height).SelectMany(i => 
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

            return new Image(new ImageInfo(name, width, height, height * width * sizeof(double)), data);
        }
    }
}
