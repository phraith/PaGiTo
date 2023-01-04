namespace ParallelGisaxsToolkit.Gisaxs.Core.UserStore
{
    public record User(long UserId, byte[] PasswordSalt, byte[] PasswordHash);
}