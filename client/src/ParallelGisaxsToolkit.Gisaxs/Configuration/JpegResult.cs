namespace ParallelGisaxsToolkit.Gisaxs.Configuration
{
    public record JpegResult(string Data, int Width, int Height);
    public record NumericResult(double[] ModifiedData, int Width, int Height);
}
