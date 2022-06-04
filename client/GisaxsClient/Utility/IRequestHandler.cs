#nullable enable

using GisaxsClient.Utility;

namespace GisaxsClient.Utility
{
    public interface IRequestHandler
    {
        public RequestResult? HandleRequest(Request request);
    }
}
