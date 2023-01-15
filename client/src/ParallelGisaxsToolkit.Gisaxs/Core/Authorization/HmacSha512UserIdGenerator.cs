using System.Security.Cryptography;
using System.Text;

namespace ParallelGisaxsToolkit.Gisaxs.Core.Authorization;

public class HmacSha512UserIdGenerator : IUserIdGenerator
{
    private readonly HMACSHA512 _userIdGenerator;

    public HmacSha512UserIdGenerator()
    {
        _userIdGenerator = new HMACSHA512("MySecretKey"u8.ToArray());
    }

    public long Generate(string username)
    {
        var hash = _userIdGenerator.ComputeHash(Encoding.UTF8.GetBytes(username));
        return BitConverter.ToInt64(hash);
    }
}