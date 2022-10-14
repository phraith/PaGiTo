using System.Runtime.Serialization;

namespace Vraith.GisaxsClient.Core.RequestHandling
{
    [Serializable]
    internal class TransientException : Exception
    {
        public TransientException()
        {
        }

        public TransientException(string message) : base(message)
        {
        }

        public TransientException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected TransientException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}