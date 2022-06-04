#nullable enable

using GisaxsClient.Utility;

namespace GisaxsClient.Utility
{
    internal interface IRequestFactory
    {
        public Request? CreateRequest(string request);
    }
}