#nullable enable

using System.Data;
using Dapper;
using Npgsql;

namespace Vraith.GisaxsClient.Core.UserStore
{
    public class UserStore
    {
        private readonly string connectionString;
        public UserStore(string connectionString)
        {
            _connectionString = connectionString;
            using IDbConnection connection = new NpgsqlConnection(connectionString);
            connection.Execute(
                @$"CREATE TABLE IF NOT EXISTS users (
                    UserId BIGINT NOT NULL PRIMARY KEY,
                    PasswordSalt BYTEA NOT NULL,
                    PasswordHash BYTEA NOT NULL); "
                );
        }

        public async Task<IEnumerable<User>> Get()
        {
            using IDbConnection connection = new NpgsqlConnection(connectionString);
            return await connection.QueryAsync<User>(@"SELECT * FROM users");
        }

        public async Task<IEnumerable<User>> Get(long id)
        {
            using IDbConnection connection = new NpgsqlConnection(connectionString);
            return await connection.QueryAsync<User>(@$"SELECT * FROM users WHERE Id = {id}");
        }

        public async void Delete(long id)
        {
            using IDbConnection connection = new NpgsqlConnection(connectionString);
            await connection.ExecuteAsync($@"DELETE * FROM users WHERE Id = {id}");
        }

        public async void Insert(User user)
        {
            using IDbConnection connection = new NpgsqlConnection(connectionString);
            await connection.ExecuteAsync($@"
                    INSERT INTO users (userid, passwordsalt, passwordhash)
                    VALUES ({user.UserId}, @salt, @hash)",
                    new { salt = user.PasswordSalt, hash = user.PasswordHash });
        }
    }
}
