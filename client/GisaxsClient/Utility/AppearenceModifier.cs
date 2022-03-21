using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace RedisTest.Controllers
{
    public static class AppearenceModifier
    {
        private static readonly Dictionary<string, ColormapTypes> colormapTypeMapping = new()
        { 
            { "twilightshifted", ColormapTypes.TwilightShifted },
            { "twilight", ColormapTypes.Twilight },
            { "autumn", ColormapTypes.Autumn },
            { "parula", ColormapTypes.Parula },
            { "bone", ColormapTypes.Bone },
            { "cividis", ColormapTypes.Cividis},
            { "cool", ColormapTypes.Cool },
            { "hot", ColormapTypes.Hot },
            { "hsv", ColormapTypes.Hsv },
            { "inferno", ColormapTypes.Inferno },
            { "jet", ColormapTypes.Jet },
            { "magma", ColormapTypes.Magma },
            { "ocean", ColormapTypes.Ocean },
            { "pink", ColormapTypes.Pink },
            { "plasma", ColormapTypes.Plasma },
            { "rainbow", ColormapTypes.Rainbow },
            { "spring", ColormapTypes.Spring },
            { "summer", ColormapTypes.Summer },
            { "viridis", ColormapTypes.Viridis },
            { "winter", ColormapTypes.Winter }
        };

        public static string ApplyColorMap(byte[] data, int width, int height, string colormapTypeName="")
        {
            Mat imageMatrix = new(height, width, MatType.CV_8UC1, data);

             Mat imageMatrixWithColormap = new();
            Mat flippedImageMatrixWithColormap = new();

            ColormapTypes colormapType = ColormapTypes.TwilightShifted;
            if (colormapTypeMapping.TryGetValue(colormapTypeName.ToLower(), out ColormapTypes foundColormapType))
            {
                colormapType = foundColormapType;
            }

            Cv2.ApplyColorMap(imageMatrix, imageMatrixWithColormap, colormapType);
            Cv2.Flip(imageMatrixWithColormap, flippedImageMatrixWithColormap, FlipMode.X);
            Cv2.ImEncode(".jpg", flippedImageMatrixWithColormap, out byte[] output, new ImageEncodingParam[] { new ImageEncodingParam(ImwriteFlags.JpegQuality, 95) });
            return Convert.ToBase64String(output, 0, output.Length);
        }
    }
}