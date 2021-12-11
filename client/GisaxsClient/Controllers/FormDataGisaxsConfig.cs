using System.Collections.Generic;

namespace RedisTest.Controllers
{
    public class FormDataGisaxsConfig
    {
        public IReadOnlyCollection<FormDataShapeConfig> shapes { get; set; }
        public FormDataUnitcellConfig unitcell { get; set; }

        public FormDataInstrumentationConfig instrumentation { get; set; }
    }

    public class FormDataInstrumentationConfig
    {
        public FormDataScattering scattering { get; set; }
        public FormDataDetector detector { get; set; }
    }

    public class FormDataDetector
    {
        public int width { get; set; }
        public int height { get; set; }
        public int beamDirX { get; set; }
        public int beamDirY { get; set; }
    }

    public class FormDataScattering
    {
        public double alphai { get; set; }
        public double beamev { get; set; }
        public double pixelsize { get; set; }
        public int detectorDistance { get; set; }
    }

    public class FormDataUnitcellConfig
    {
        public MyRepetitions repetitions { get; set; }
        public MyDistances distances { get; set; }
    }

    public class MyRepetitions
    {
        public int repetitionsInX { get; set; }
        public int repetitionsInY { get; set; }
        public int repetitionsInZ { get; set; }
    }

    public class MyDistances
    {
        public int distOnX { get; set; }
        public int distOnY { get; set; }
        public int distOnZ { get; set; }
    }

    public class FormDataShapeConfig
    {
        public FormDataParameters parameters { get; set; }
        public RefractionIndex refindex { get; set; }
        public FormDataLocation location { get; set; }
    }

    public class FormDataParameters
    {
        public FormDataParameter radius { get; set; }
        public FormDataParameter height { get; set; }
    }

    public class FormDataParameter
    {
        public double mean { get; set; }
        public double stddev { get; set; }
    }

    public class FormDataLocation
    {
        public int posX { get; set; }
        public int posY { get; set; }
        public int posZ { get; set; }
    }
}
