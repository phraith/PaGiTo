#nullable enable


namespace Vraith.GisaxsClient.Core.RequestHandling
{
    public interface IRequestHandler
    {
        public RequestResult? HandleRequest(Request request);
    }
}
