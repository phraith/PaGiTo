using System;
using System.Collections.Generic;

namespace RedisTest.Controllers
{
    public static class GisaxsConfigCreator
    {
        public static GisaxsConfig CreateValidConfigFromFormDataConfig(FormDataGisaxsConfig config)
        {
            GisaxsConfig gisaxsConfig = new GisaxsConfig();
            gisaxsConfig.name = "Test";
            gisaxsConfig.substrate = CreateValidSubstrateConfig();

            List<ShapeConfig> gisaxsShapes = new List<ShapeConfig>();
            List<ComponentConfig> gisaxsComponents = new List<ComponentConfig>();
            int c = 0;
            foreach (FormDataShapeConfig shape in config.shapes)
            {
                FormDataParameters parameters = shape.parameters;
                RefractionIndex refindex = shape.refindex;

                if (parameters.height != null && parameters.radius != null)
                {
                    var gisaxsShapeConfig = new ShapeConfig { name = $"{c}", refindex = refindex, type = "cylinder", parameters = new List<Parameter> { new Parameter { type = "radius", mean = parameters.radius.mean, stddev = parameters.radius.stddev }, new Parameter { type = "height", mean = parameters.height.mean, stddev = parameters.height.stddev } } };
                    var componentConfig = new ComponentConfig { shape = $"{c}", locations = new List<int[]> { new int[] { shape.location.posX, shape.location.posY, shape.location.posZ } } };
                    gisaxsShapes.Add(gisaxsShapeConfig);
                    gisaxsComponents.Add(componentConfig);
                }
                else if (parameters.radius != null)
                {
                    var gisaxsShapeConfig = new ShapeConfig { name = $"{c}", refindex = refindex, type = "sphere", parameters = new List<Parameter> { new Parameter { type = "radius", mean = parameters.radius.mean, stddev = parameters.radius.stddev } } };
                    var componentConfig = new ComponentConfig { shape = $"{c}", locations = new List<int[]> { new int[] { shape.location.posX, shape.location.posY, shape.location.posZ } } };
                    gisaxsShapes.Add(gisaxsShapeConfig);
                    gisaxsComponents.Add(componentConfig);
                }
                ++c;
            }

            gisaxsConfig.unitcell = new UnitcellConfig { components = gisaxsComponents, distances = new double[] { config.unitcell.distances.distOnX, config.unitcell.distances.distOnY, config.unitcell.distances.distOnZ }, repetitions = new int[] { config.unitcell.repetitions.repetitionsInX, config.unitcell.repetitions.repetitionsInY, config.unitcell.repetitions.repetitionsInZ } };
            gisaxsConfig.shapes = gisaxsShapes;
            return gisaxsConfig;
        }

        public static InstrumentationConfig CreateValidInstrumentationConfigFromFormData(FormDataInstrumentationConfig instrumentation)
        {
            Fitting fitting = new Fitting { evolutions=1, generations=2, populations=4, individuals=100 };
            Scattering scattering = new Scattering { alphai=instrumentation.scattering.alphai, photon=new Photon { ev=instrumentation.scattering.beamev } };
            Detector detector = new Detector { directbeam = new int[] { instrumentation.detector.beamDirX, instrumentation.detector.beamDirY }, resolution=new int[] { instrumentation.detector.width, instrumentation.detector.height }, pixelsize=instrumentation.scattering.pixelsize, sdd=instrumentation.scattering.detectorDistance };
        
            return new InstrumentationConfig { detector = detector, fitting = fitting, scattering = scattering };
        }

        private static SubstrateConfig CreateValidSubstrateConfig()
        {
            var config = new SubstrateConfig();
            config.order = -1;
            config.refindex = new RefractionIndex() { beta = 2e-08, delta = 6e-06 };
            return config;
        }
    }

    public class InstrumentationConfig
    {
        public Scattering scattering { get; set; }
        public Detector detector { get; set; }

        public Fitting fitting { get; set; } 
    }

    public class Fitting
    {
        public int evolutions { get; set; }
        public int generations { get; set; }
        public int populations { get; set; }
        public int individuals { get; set; }
    }

    public class Detector
    {
        public double pixelsize { get;  set; }
        public int[] resolution { get; set; }
        public int sdd {get; set; }
        public int[] directbeam { get; set; }
    }

    public class Scattering
    {
        public double alphai { get; set; }

        public Photon photon { get; set; }
    }

    public class Photon
    {
        public double ev { get; set; }
    }
}
