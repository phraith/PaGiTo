using System;
using System.Security.Cryptography;
using System.Text;

#nullable enable

namespace GisaxsClient.Utility
{
    internal class Sha256HashComputer : IHashComputer
    {
        public string Hash(string input)
        {
            return BitConverter.ToString(SHA256.HashData(Encoding.UTF8.GetBytes(input)));
        }
    }
}