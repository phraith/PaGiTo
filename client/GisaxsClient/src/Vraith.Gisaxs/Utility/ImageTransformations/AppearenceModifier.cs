using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Vraith.Gisaxs.Utility.ImageTransformations
{
    public static class ColormapValueProvider
    {
        private static readonly Dictionary<string, (float[] rValues, float[] gValues, float[] bValues)> DataMapping = new()
        {
            { "twilightshifted", (TwilightShifted.R, TwilightShifted.G, TwilightShifted.B) },
            { "twilight", (Twilight.R, Twilight.G, Twilight.B) },
            { "autumn", (Autumn.R, Autumn.G, Autumn.B) },
            { "parula", (Parula.R, Parula.G, Parula.B) },
            { "bone", (Bone.R, Bone.G, Bone.B) },
            { "cividis", (Cividis.R, Cividis.G, Cividis.B) },
            { "cool", (Cool.R, Cool.G, Cool.B) },
            { "hot", (Hot.R, Hot.G, Hot.B) },
            { "hsv", (Hsv.R, Hsv.G, Hsv.B) },
            { "inferno", (Inferno.R, Inferno.G, Inferno.B) },
            { "jet", (Jet.R, Jet.G, Jet.B) },
            { "magma", (Magma.R, Magma.G, Magma.B) },
            { "ocean", (Ocean.R, Ocean.G, Ocean.B) },
            { "pink", (Pink.R, Pink.G, Pink.B) },
            { "plasma", (Plasma.R, Plasma.G, Plasma.B) },
            { "rainbow", (Rainbow.R, Rainbow.G, Rainbow.B) },
            { "spring", (Spring.R, Spring.G, Spring.B) },
            { "summer", (Summer.R, Summer.G, Summer.B) },
            { "viridis", (Viridis.R, Viridis.G, Viridis.B) },
            { "winter", (Winter.R, Winter.G, Winter.B) }
        };

        public static (byte r, byte g, byte b) ColorValue(string colormapName, byte dataPoint)
        {
            var (rValues, gValues, bValues) = DataMapping[colormapName.ToLower()];
            return ColorValue(rValues, gValues, bValues, dataPoint);
        }

        private static (byte r, byte g, byte b) ColorValue(float[] rValues, float[] gValues, float[] bValues, byte dataPoint)
        {
            var interpolatedR = LinearInterpolate(dataPoint, rValues);
            var interpolatedG = LinearInterpolate(dataPoint, gValues);
            var interpolatedB = LinearInterpolate(dataPoint, bValues);

            byte r = BitConverter.GetBytes((int)(interpolatedR * 255.0)).First();
            byte g = BitConverter.GetBytes((int)(interpolatedG * 255.0)).First();
            byte b = BitConverter.GetBytes((int)(interpolatedB * 255.0)).First();

            return (r, g, b);
        }

        private static double LinearInterpolate(byte dataPoint, float[] data)
        {
            var scaleFactor = data.Length / 256.0;

            int unscaledIndex = dataPoint;
            double t = unscaledIndex * scaleFactor;
            int scaledIndex = (int)Math.Floor(t);
            double diffFactor = t - scaledIndex;

            double rStart = data[scaledIndex];
            if (scaledIndex + 1 >= data.Length)
            {
                return rStart;
            }

            double rEnd = data[scaledIndex + 1];
            double diff = rEnd - rStart;
            double valueToAdd = diff * diffFactor;

            return rStart + valueToAdd;
        }
    }

    public static class ImageExtensions
    {
        public static Image<Rgb24> ApplyColormap(this Image<L8> image, string colormapName)
        {
            var coloredImage = new Image<Rgb24>(image.Width, image.Height);
            for (int i = 0; i < image.Width; i++)
            {
                for (int j = 0; j < image.Height; j++)
                {
                    var greyscale = image[i, j];
                    var rgb = ColormapValueProvider.ColorValue(colormapName, greyscale.PackedValue);
                    coloredImage[i, j] = new Rgb24(rgb.r, rgb.g, rgb.b);
                }
            }
            return coloredImage;
        }
    }


    public static class AppearanceModifier
    {
        public static string ApplyColorMap(byte[] data, int width, int height, bool revertImage = true, string colormapTypeName = "")
        {
            Image<L8> image = Image.LoadPixelData<L8>(data, width, height);
            if (revertImage)
            {
                image.Mutate(x => x.Rotate(RotateMode.Rotate180));
            }
            var newImage = image.ApplyColormap(colormapTypeName);
            var res = newImage.ToBase64String(JpegFormat.Instance);
            return res;
        }
    }
}