﻿namespace GisaxsClient.Utility
{
    public record RequestResult
    {
        public string Body { get; init; }
        public string Command { get; init; }
    }
}