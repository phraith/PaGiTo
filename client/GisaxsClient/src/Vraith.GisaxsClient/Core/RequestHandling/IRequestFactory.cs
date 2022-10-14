#nullable enable


namespace Vraith.GisaxsClient.Core.RequestHandling
{
    internal interface IRequestFactory
    {
        public Request? CreateRequest(string request);
    }
}