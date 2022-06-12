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
    public int Id { get; set; }
    public string Name { get; set; }
    public long Size { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public double[] Data { get; set; }
}
