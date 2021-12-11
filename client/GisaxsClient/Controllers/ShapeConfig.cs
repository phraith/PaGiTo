using System.Collections.Generic;

namespace RedisTest.Controllers
{
    public class Parameter
    {
        public string type { get; set; }
        public double mean { get; set; }
        public double stddev { get; set; }
    }

    public class ShapeConfig
    {
        public string name { get; set; }
        public string type { get; set; }
        public RefractionIndex refindex { get; set; }
        public IReadOnlyCollection<Parameter> parameters {get; set;}
    }
}