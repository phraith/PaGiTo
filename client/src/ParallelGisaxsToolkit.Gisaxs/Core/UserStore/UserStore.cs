using System.Data;
using Dapper;

namespace ParallelGisaxsToolkit.Gisaxs.Core.UserStore
{
    public class UserStore : IUserStore
    {
        private readonly IDbConnection _connection;

        public UserStore(IDbConnection connection)
        {
            _connection = connection;
            connection.Execute(
                @$"CREATE TABLE IF NOT EXISTS users (
                    UserId BIGINT NOT NULL PRIMARY KEY,
                    PasswordSalt BYTEA NOT NULL,
                    PasswordHash BYTEA NOT NULL); "
            );
        }

        public async Task<IEnumerable<User>> Get()
        {
            return await _connection.QueryAsync<User>(@"SELECT * FROM users");
        }

        public async Task<IEnumerable<User>> Get(long id)
        {
            return await _connection.QueryAsync<User>(@$"SELECT * FROM users WHERE userId = {id}");
        }

        public async Task Delete(long id)
        {
            await _connection.ExecuteAsync($@"DELETE * FROM users WHERE Id = {id}");
        }

        public async Task Insert(User user)
        {
            await _connection.ExecuteAsync($@"
                    INSERT INTO users (userid, passwordsalt, passwordhash)
                    VALUES ({user.UserId}, @salt, @hash)",
                new { salt = user.PasswordSalt, hash = user.PasswordHash });
        }
    }
}