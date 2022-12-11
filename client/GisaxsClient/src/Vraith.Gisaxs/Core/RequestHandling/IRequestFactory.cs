#nullable enable


namespace Vraith.Gisaxs.Core.RequestHandling
{
    public interface IRequestFactory
    {
        public Request? CreateRequest(string request, string dataAccessor);
    }
}