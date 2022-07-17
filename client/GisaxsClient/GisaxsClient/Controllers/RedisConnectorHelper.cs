//using StackExchange.Redis;

//namespace GisaxsClient.Controllers
//{
//    internal class RedisConnectorHelper
//    {
//        private Lazy<ConnectionMultiplexer> lazyConnection;
//        internal RedisConnectorHelper(IConfiguration configuration)
//        {

//            var connectionString = configuration.GetConnectionString("Redis");
//            lazyConnection = new Lazy<ConnectionMultiplexer>(() =>
//            {
//                return ConnectionMultiplexer.Connect(connectionString);
//            });
//        }


//        public ConnectionMultiplexer Connection
//        {
//            get
//            {
//                return lazyConnection.Value;
//            }
//        }
//    }
//}