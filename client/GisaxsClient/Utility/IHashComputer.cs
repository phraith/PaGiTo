#nullable enable

using GisaxsClient.Utility;

namespace GisaxsClient.Utility
{
    internal interface IHashComputer
    {
        string Hash(string input);
    }
}