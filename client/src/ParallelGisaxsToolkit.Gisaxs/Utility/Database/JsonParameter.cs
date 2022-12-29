using System.Data;
using Npgsql;
using NpgsqlTypes;
using static Dapper.SqlMapper;

namespace ParallelGisaxsToolkit.Gisaxs.Utility.Database
{
    public class JsonParameter : ICustomQueryParameter
    {
        private readonly string _value;
        public JsonParameter(string value)
        {
            _value = value;
        }

        public void AddParameter(IDbCommand command, string name)
        {
            var parameter = new NpgsqlParameter(name, NpgsqlDbType.Jsonb)
            {
                Value = _value
            };

            command.Parameters.Add(parameter);
        }
    }
}
