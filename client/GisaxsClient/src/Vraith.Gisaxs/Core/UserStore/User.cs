namespace Vraith.Gisaxs.Core.UserStore
{
    public record User(long UserId, byte[] PasswordSalt, byte[] PasswordHash);
}