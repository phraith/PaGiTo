namespace ParallelGisaxsToolkit.Gisaxs.Core.RequestHandling
{
    public interface IRequestFactory
    {
        public Request? CreateRequest(string request, string clientId);
    }
}