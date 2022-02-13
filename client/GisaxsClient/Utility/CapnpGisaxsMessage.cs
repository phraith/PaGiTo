using Capnp;
using CapnpGen;
using RedisTest.Controllers;
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace GisaxsClient.Utility
{
    public class CapnpGisaxsMessage : IGisaxsMessage
    {
        private readonly string gisaxsConfig;
        private readonly string instrumentationConfig;
        public CapnpGisaxsMessage(GisaxsConfig gisaxsConfig, InstrumentationConfig instrumentationConfig)
        {
            this.gisaxsConfig = JsonSerializer.Serialize(gisaxsConfig);
            this.instrumentationConfig = JsonSerializer.Serialize(instrumentationConfig);
            message = new Lazy<byte[]>(() => CreateMessage(this.gisaxsConfig, this.instrumentationConfig));
        }

        private readonly Lazy<byte[]> message; 
        public byte[] Message => message.Value;
        public string ID => BitConverter.ToString(SHA256.HashData(Encoding.UTF8.GetBytes(gisaxsConfig).Concat(Encoding.UTF8.GetBytes(instrumentationConfig)).ToArray()));
        public byte[] CreateMessage(string gisaxsConfig, string instrumentationConfig)
        {
            SerializedSimulationDescription descr = new()
            {
                IsLast = false,
                Timestamp = 1001,
                InstrumentationData = instrumentationConfig,
                ConfigData = gisaxsConfig,
                ClientId = ID
            };

            var msg = MessageBuilder.Create();
            var root = msg.BuildRoot<SerializedSimulationDescription.WRITER>();
            descr.serialize(root);
            var mems = new MemoryStream();
            var pump = new FramePump(mems);
            pump.Send(msg.Frame);
            mems.Seek(0, SeekOrigin.Begin);
            return mems.ToArray();
        }
    }
}
