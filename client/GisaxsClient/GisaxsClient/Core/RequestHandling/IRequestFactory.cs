#nullable enable


namespace GisaxsClient.Core.RequestHandling
{
    internal interface IRequestFactory
    {
        public Request? CreateRequest(string request);
    }
}