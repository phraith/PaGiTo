namespace GisaxsClient.Utility
{
    public interface IGisaxsMessage
    {
        public string ID { get; }
        public byte[] Message { get; }
    }
}