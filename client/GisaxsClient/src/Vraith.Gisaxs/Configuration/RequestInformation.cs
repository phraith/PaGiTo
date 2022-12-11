namespace Vraith.Gisaxs.Configuration
{
    public record RequestInformation(JobInformation JobInformation, ClientInformation ClientInformation);

    public record JobInformation(long ClientId, IReadOnlyList<SimulationTarget> SimulationTargets, long JobId, long? ImageId = null);

    public record ClientInformation(string JobType);
    
    public record SimulationTarget(DetectorPosition Start, DetectorPosition End);

    public record SimulationTargetWithId(SimulationTarget Target, int Id);
    
    public record DetectorPosition(int X, int Y);
}