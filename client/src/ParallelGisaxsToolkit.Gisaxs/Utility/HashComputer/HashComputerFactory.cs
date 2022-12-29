namespace ParallelGisaxsToolkit.Gisaxs.Utility.HashComputer;

public static class HashComputerFactory
{
    public static IHashComputer CreateSha256HashComputer()
    {
        return new Sha256HashComputer();
    }
}