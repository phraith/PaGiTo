#nullable enable


namespace GisaxsClient.Core.RequestHandling
{
    public interface IRequestHandler
    {
        public RequestResult? HandleRequest(Request request);
    }
}
