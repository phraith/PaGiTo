namespace Vraith.GisaxsClient.Core.UserStore
{
    public class User
    {
        public long UserId { get; set; }
        public byte[] PasswordHash { get; set; }
        public byte[] PasswordSalt { get; set; }
    }
}
