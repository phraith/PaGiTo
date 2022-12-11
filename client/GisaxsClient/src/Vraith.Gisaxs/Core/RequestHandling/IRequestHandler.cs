#nullable enable


namespace Vraith.Gisaxs.Core.RequestHandling
{
    public interface IRequestHandler
    {
        RequestResult? HandleRequest(Request request);
    }
}
