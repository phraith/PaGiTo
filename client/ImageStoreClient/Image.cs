// See https://aka.ms/new-console-template for more information

//var maxIntensity = imageData.Max();
//byte[] normalizedImage = imageData.Select(x => (x / maxIntensity) * 255.0).Select(x => (byte) x).ToArray();



//var base64 = AppearenceModifier.ApplyColorMap(normalizedImage, width, height, "twilight");

//var f = b.Where(x => x != 0).ToArray();


//Cv2.ImEncode(".jpg", mat, out byte[] output, new ImageEncodingParam[] { new ImageEncodingParam(ImwriteFlags.JpegQuality, 95) });
//Cv2.ImWrite(@"C:\Users\phili\OneDrive\Bilder\test.jpg", mat);


//using (new Window("dst image", mat))
//{
//    Cv2.WaitKey();
//}

public class Image
{
    public ImageInfo Info { get; set; }

    public Image(ImageInfo imageInfo, double[] data)
    {
        Info = imageInfo;
        Data = data;
    }

    public double[] Data { get; set; }
}
