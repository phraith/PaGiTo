namespace Vraith.ImageStoreClient.ImageUtility.ImageLoaders
{
    internal class AsciiLoader : IImageLoader
    {
        private static readonly string Separator = "    ";
        public Image Load(string path)
        {
            string content = File.ReadAllText(path);
            string[] split = content.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            string[] data = split[3..];
            int height = data.Length;
            int width = data[0].Split(Separator, StringSplitOptions.RemoveEmptyEntries).Length - 1;
            List<double> imageData = new(width * height);

            foreach (string row in data)
            {
                string[] rowSplit = row.Split(Separator, StringSplitOptions.RemoveEmptyEntries);
                double[] doubles = rowSplit[..^1].Select(x => double.Parse(x)).ToArray();
                imageData.AddRange(doubles);
            }

            ImageInfo info = new(Path.GetFileNameWithoutExtension(path), width, height, imageData.Count * sizeof(double));

            byte[] greyscaleImage = IntensityNormalizer.Normalize(imageData);
            Image image = new(info, imageData.ToArray(), imageDataTransposed, greyscaleImage);
            return image;
        }
    }
}
