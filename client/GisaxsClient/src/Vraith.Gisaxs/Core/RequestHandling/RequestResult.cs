namespace Vraith.Gisaxs.Core.RequestHandling
{
    public record RequestResult
    {
        public string DataAccessor { get; init; }
        public string SignalREndpoint { get; init; }
    }
}