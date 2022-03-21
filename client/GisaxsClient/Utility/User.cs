namespace UserDataProvider
{
    public class User
    {
        public long Id { get; set; } 
        public byte[] PasswordHash { get; set; }
        public byte[] PasswordSalt { get; set; }
    }
}
