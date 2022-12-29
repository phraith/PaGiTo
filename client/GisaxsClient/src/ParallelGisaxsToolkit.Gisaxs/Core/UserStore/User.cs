namespace ParallelGisaxsToolkit.Gisaxs.Core.UserStore
{
    public record User(long UserId, IReadOnlyList<byte> PasswordHash, IReadOnlyList<byte> PasswordSalt);
}