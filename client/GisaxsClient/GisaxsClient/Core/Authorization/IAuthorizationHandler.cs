﻿using GisaxsClient.Controllers;
using GisaxsClient.Core.UserStore;

namespace GisaxsClient.Core.Authorization
{
    public interface IAuthorizationHandler
    {
        (long userId, byte[] passwordHash, byte[] passwordSalt) CreatePasswordHash(string password, string username);
        string CreateJwtToken(User user);
        bool VerifyPasswordHash(User user, string password);
        long CreateUserId(string username);
    }
}
