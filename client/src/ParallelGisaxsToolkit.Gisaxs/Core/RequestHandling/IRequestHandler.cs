#nullable enable


namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling
{
    public interface IRequestHandler
    {
        RequestResult? HandleRequest(Request request);
    }
}
