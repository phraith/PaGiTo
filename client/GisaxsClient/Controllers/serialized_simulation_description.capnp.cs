using Capnp;
using Capnp.Rpc;
using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace CapnpGen
{
    [System.CodeDom.Compiler.GeneratedCode("capnpc-csharp", "1.3.0.0"), TypeId(0xed44f32b9b4359bcUL)]
    public class SerializedSimulationDescription : ICapnpSerializable
    {
        public const UInt64 typeId = 0xed44f32b9b4359bcUL;
        void ICapnpSerializable.Deserialize(DeserializerState arg_)
        {
            var reader = READER.create(arg_);
            Timestamp = reader.Timestamp;
            ClientId = reader.ClientId;
            ConfigData = reader.ConfigData;
            InstrumentationData = reader.InstrumentationData;
            IsLast = reader.IsLast;
            applyDefaults();
        }

        public void serialize(WRITER writer)
        {
            writer.Timestamp = Timestamp;
            writer.ClientId = ClientId;
            writer.ConfigData = ConfigData;
            writer.InstrumentationData = InstrumentationData;
            writer.IsLast = IsLast;
        }

        void ICapnpSerializable.Serialize(SerializerState arg_)
        {
            serialize(arg_.Rewrap<WRITER>());
        }

        public void applyDefaults()
        {
        }

        public uint Timestamp
        {
            get;
            set;
        }

        public string ClientId
        {
            get;
            set;
        }

        public string ConfigData
        {
            get;
            set;
        }

        public string InstrumentationData
        {
            get;
            set;
        }

        public bool IsLast
        {
            get;
            set;
        }

        public struct READER
        {
            readonly DeserializerState ctx;
            public READER(DeserializerState ctx)
            {
                this.ctx = ctx;
            }

            public static READER create(DeserializerState ctx) => new READER(ctx);
            public static implicit operator DeserializerState(READER reader) => reader.ctx;
            public static implicit operator READER(DeserializerState ctx) => new READER(ctx);
            public uint Timestamp => ctx.ReadDataUInt(0UL, 0U);
            public string ClientId => ctx.ReadText(0, null);
            public string ConfigData => ctx.ReadText(1, null);
            public string InstrumentationData => ctx.ReadText(2, null);
            public bool IsLast => ctx.ReadDataBool(32UL, false);
        }

        public class WRITER : SerializerState
        {
            public WRITER()
            {
                this.SetStruct(1, 3);
            }

            public uint Timestamp
            {
                get => this.ReadDataUInt(0UL, 0U);
                set => this.WriteData(0UL, value, 0U);
            }

            public string ClientId
            {
                get => this.ReadText(0, null);
                set => this.WriteText(0, value, null);
            }

            public string ConfigData
            {
                get => this.ReadText(1, null);
                set => this.WriteText(1, value, null);
            }

            public string InstrumentationData
            {
                get => this.ReadText(2, null);
                set => this.WriteText(2, value, null);
            }

            public bool IsLast
            {
                get => this.ReadDataBool(32UL, false);
                set => this.WriteData(32UL, value, false);
            }
        }
    }

    [System.CodeDom.Compiler.GeneratedCode("capnpc-csharp", "1.3.0.0"), TypeId(0xf75f5cc4ade48759UL)]
    public class SerializedSimResult : ICapnpSerializable
    {
        public const UInt64 typeId = 0xf75f5cc4ade48759UL;
        void ICapnpSerializable.Deserialize(DeserializerState arg_)
        {
            var reader = READER.create(arg_);
            Timestamp = reader.Timestamp;
            ClientId = reader.ClientId;
            SimulatedIntensities = reader.SimulatedIntensities;
            SimulatedQx = reader.SimulatedQx;
            SimulatedQy = reader.SimulatedQy;
            SimulatedQz = reader.SimulatedQz;
            XDim = reader.XDim;
            YDim = reader.YDim;
            DeviceTimingData = reader.DeviceTimingData?.ToReadOnlyList(_ => CapnpSerializable.Create<CapnpGen.SerializedSimResult.TimingData>(_));
            applyDefaults();
        }

        public void serialize(WRITER writer)
        {
            writer.Timestamp = Timestamp;
            writer.ClientId = ClientId;
            writer.SimulatedIntensities.Init(SimulatedIntensities);
            writer.SimulatedQx.Init(SimulatedQx);
            writer.SimulatedQy.Init(SimulatedQy);
            writer.SimulatedQz.Init(SimulatedQz);
            writer.XDim = XDim;
            writer.YDim = YDim;
            writer.DeviceTimingData.Init(DeviceTimingData, (_s1, _v1) => _v1?.serialize(_s1));
        }

        void ICapnpSerializable.Serialize(SerializerState arg_)
        {
            serialize(arg_.Rewrap<WRITER>());
        }

        public void applyDefaults()
        {
        }

        public uint Timestamp
        {
            get;
            set;
        }

        public string ClientId
        {
            get;
            set;
        }

        public IReadOnlyList<float> SimulatedIntensities
        {
            get;
            set;
        }

        public IReadOnlyList<float> SimulatedQx
        {
            get;
            set;
        }

        public IReadOnlyList<float> SimulatedQy
        {
            get;
            set;
        }

        public IReadOnlyList<float> SimulatedQz
        {
            get;
            set;
        }

        public uint XDim
        {
            get;
            set;
        }

        public uint YDim
        {
            get;
            set;
        }

        public IReadOnlyList<CapnpGen.SerializedSimResult.TimingData> DeviceTimingData
        {
            get;
            set;
        }

        public struct READER
        {
            readonly DeserializerState ctx;
            public READER(DeserializerState ctx)
            {
                this.ctx = ctx;
            }

            public static READER create(DeserializerState ctx) => new READER(ctx);
            public static implicit operator DeserializerState(READER reader) => reader.ctx;
            public static implicit operator READER(DeserializerState ctx) => new READER(ctx);
            public uint Timestamp => ctx.ReadDataUInt(0UL, 0U);
            public string ClientId => ctx.ReadText(0, null);
            public IReadOnlyList<float> SimulatedIntensities => ctx.ReadList(1).CastFloat();
            public IReadOnlyList<float> SimulatedQx => ctx.ReadList(2).CastFloat();
            public IReadOnlyList<float> SimulatedQy => ctx.ReadList(3).CastFloat();
            public IReadOnlyList<float> SimulatedQz => ctx.ReadList(4).CastFloat();
            public uint XDim => ctx.ReadDataUInt(32UL, 0U);
            public uint YDim => ctx.ReadDataUInt(64UL, 0U);
            public IReadOnlyList<CapnpGen.SerializedSimResult.TimingData.READER> DeviceTimingData => ctx.ReadList(5).Cast(CapnpGen.SerializedSimResult.TimingData.READER.create);
        }

        public class WRITER : SerializerState
        {
            public WRITER()
            {
                this.SetStruct(2, 6);
            }

            public uint Timestamp
            {
                get => this.ReadDataUInt(0UL, 0U);
                set => this.WriteData(0UL, value, 0U);
            }

            public string ClientId
            {
                get => this.ReadText(0, null);
                set => this.WriteText(0, value, null);
            }

            public ListOfPrimitivesSerializer<float> SimulatedIntensities
            {
                get => BuildPointer<ListOfPrimitivesSerializer<float>>(1);
                set => Link(1, value);
            }

            public ListOfPrimitivesSerializer<float> SimulatedQx
            {
                get => BuildPointer<ListOfPrimitivesSerializer<float>>(2);
                set => Link(2, value);
            }

            public ListOfPrimitivesSerializer<float> SimulatedQy
            {
                get => BuildPointer<ListOfPrimitivesSerializer<float>>(3);
                set => Link(3, value);
            }

            public ListOfPrimitivesSerializer<float> SimulatedQz
            {
                get => BuildPointer<ListOfPrimitivesSerializer<float>>(4);
                set => Link(4, value);
            }

            public uint XDim
            {
                get => this.ReadDataUInt(32UL, 0U);
                set => this.WriteData(32UL, value, 0U);
            }

            public uint YDim
            {
                get => this.ReadDataUInt(64UL, 0U);
                set => this.WriteData(64UL, value, 0U);
            }

            public ListOfStructsSerializer<CapnpGen.SerializedSimResult.TimingData.WRITER> DeviceTimingData
            {
                get => BuildPointer<ListOfStructsSerializer<CapnpGen.SerializedSimResult.TimingData.WRITER>>(5);
                set => Link(5, value);
            }
        }

        [System.CodeDom.Compiler.GeneratedCode("capnpc-csharp", "1.3.0.0"), TypeId(0xe0086356b58c20dfUL)]
        public class TimingData : ICapnpSerializable
        {
            public const UInt64 typeId = 0xe0086356b58c20dfUL;
            void ICapnpSerializable.Deserialize(DeserializerState arg_)
            {
                var reader = READER.create(arg_);
                DeviceName = reader.DeviceName;
                KernelTime = reader.KernelTime;
                SimulationTime = reader.SimulationTime;
                applyDefaults();
            }

            public void serialize(WRITER writer)
            {
                writer.DeviceName = DeviceName;
                writer.KernelTime = KernelTime;
                writer.SimulationTime = SimulationTime;
            }

            void ICapnpSerializable.Serialize(SerializerState arg_)
            {
                serialize(arg_.Rewrap<WRITER>());
            }

            public void applyDefaults()
            {
            }

            public string DeviceName
            {
                get;
                set;
            }

            public float KernelTime
            {
                get;
                set;
            }

            public float SimulationTime
            {
                get;
                set;
            }

            public struct READER
            {
                readonly DeserializerState ctx;
                public READER(DeserializerState ctx)
                {
                    this.ctx = ctx;
                }

                public static READER create(DeserializerState ctx) => new READER(ctx);
                public static implicit operator DeserializerState(READER reader) => reader.ctx;
                public static implicit operator READER(DeserializerState ctx) => new READER(ctx);
                public string DeviceName => ctx.ReadText(0, null);
                public float KernelTime => ctx.ReadDataFloat(0UL, 0F);
                public float SimulationTime => ctx.ReadDataFloat(32UL, 0F);
            }

            public class WRITER : SerializerState
            {
                public WRITER()
                {
                    this.SetStruct(1, 1);
                }

                public string DeviceName
                {
                    get => this.ReadText(0, null);
                    set => this.WriteText(0, value, null);
                }

                public float KernelTime
                {
                    get => this.ReadDataFloat(0UL, 0F);
                    set => this.WriteData(0UL, value, 0F);
                }

                public float SimulationTime
                {
                    get => this.ReadDataFloat(32UL, 0F);
                    set => this.WriteData(32UL, value, 0F);
                }
            }
        }
    }
}