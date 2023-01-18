using System.Text.Json.Serialization;

namespace ParallelGisaxsToolkit.Gisaxs.Configuration
{
    public record RequestInformation(JobProperties JobProperties, MetaInformation MetaInformation);

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum IntensityFormat
    {
        Greyscale,
        DoublePrecision
    }

    public record JobProperties(IReadOnlyList<SimulationTarget> SimulationTargets, IntensityFormat IntensityFormat,
        long? ImageId = null);

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum JobType
    {
        Fitting,
        Simulation
    }

    public record MetaInformation(JobType Type, string Notification, string? Colormap = null);

    public record SimulationTarget(DetectorPosition Start, DetectorPosition End)
    {
        public IEnumerable<byte> ToBytes()
        {
            return Start.ToBytes().Concat(End.ToBytes());
        }
    }

    public record SimulationTargetWithId(SimulationTarget Target, int Id)
    {
        public static readonly SimulationTargetWithId Empty =
            new(new SimulationTarget(new DetectorPosition(0, 0), new DetectorPosition(0, 0)),
                -1);
    }

    public record DetectorPosition(int X, int Y)
    {
        public IEnumerable<byte> ToBytes()
        {
            return BitConverter.GetBytes(X).Concat(BitConverter.GetBytes(Y));
        }
    }
}