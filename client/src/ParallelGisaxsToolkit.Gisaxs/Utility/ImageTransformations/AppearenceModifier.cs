﻿using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ParallelGisaxsToolkit.Gisaxs.Utility.ImageTransformations
{
    public static class ColormapValueProvider
    {
        public static readonly Dictionary<string, (float[] rValues, float[] gValues, float[] bValues)> DataMapping = new()
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
            (float[] rValues, float[] gValues, float[] bValues) = DataMapping[colormapName.ToLower()];
            return ColorValue(rValues, gValues, bValues, dataPoint);
        }

        private static (byte r, byte g, byte b) ColorValue(float[] rValues, float[] gValues, float[] bValues, byte dataPoint)
        {
            double interpolatedR = LinearInterpolate(dataPoint, rValues);
            double interpolatedG = LinearInterpolate(dataPoint, gValues);
            double interpolatedB = LinearInterpolate(dataPoint, bValues);

            byte r = BitConverter.GetBytes((int)(interpolatedR * 255.0)).First();
            byte g = BitConverter.GetBytes((int)(interpolatedG * 255.0)).First();
            byte b = BitConverter.GetBytes((int)(interpolatedB * 255.0)).First();

            return (r, g, b);
        }

        private static double LinearInterpolate(byte dataPoint, float[] data)
        {
            double scaleFactor = data.Length / 256.0;

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
            Image<Rgb24>? coloredImage = image.CloneAs<Rgb24>();
            
            (float[] rValues, float[] gValues, float[] bValues) = ColormapValueProvider.DataMapping[colormapName.ToLower()];
            
            coloredImage.Mutate(c => c.ProcessPixelRowsAsVector4(row =>
            {
                for (int x = 0; x < row.Length; x++)
                {
                    float first = row[x][0];
                    int index = (int)Math.Floor(first * rValues.Length);
                    index = Math.Min(rValues.Length - 1, index);
                    
                    float r = rValues[index];
                    float g = gValues[index];
                    float b = bValues[index];
                    
                    row[x] = new Vector4(r, g, b, 1);
                }
            }));
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
            Image<Rgb24> newImage = image.ApplyColormap(colormapTypeName);
            string? res = newImage.ToBase64String(JpegFormat.Instance);
            return res;
        }
    }
}