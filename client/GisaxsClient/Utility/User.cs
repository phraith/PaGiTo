namespace GisaxsClient.Utility
{
    public class User
    {
        public long UserId { get; set; }
        public byte[] PasswordHash { get; set; }
        public byte[] PasswordSalt { get; set; }
    }
}
