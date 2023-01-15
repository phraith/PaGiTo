using System.Security.Cryptography;
using System.Text;

namespace ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer
{
    public class Sha256HashComputer : IHashComputer
    {
        public string Hash(params string[] input)
        {
            byte[] bytes = input.SelectMany(x => Encoding.UTF8.GetBytes(x)).ToArray();
            return BitConverter.ToString(SHA256.HashData(bytes));
        }
    }
}