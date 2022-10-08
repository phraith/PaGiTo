using Npgsql;
using NpgsqlTypes;
using static Dapper.SqlMapper;
using System.Data;

namespace GisaxsClient.Utility.Database
{
    public class JsonParameter : ICustomQueryParameter
    {
        private readonly string value;

        public JsonParameter(string value)
        {
            this.value = value;
        }

        public void AddParameter(IDbCommand command, string name)
        {
            var parameter = new NpgsqlParameter(name, NpgsqlDbType.Jsonb)
            {
                Value = value
            };

            command.Parameters.Add(parameter);
        }
    }
}
