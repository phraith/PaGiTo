using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;

namespace Vraith.Optimization.Cmaes
{
    public class CMA
    {
        private readonly int _mu;
        private readonly double _mu_eff;
        private readonly double _cc;
        private readonly double _c1;
        private readonly double _cmu;
        private readonly double _c_sigma;
        private readonly double _d_sigma;
        private readonly int _cm;
        private readonly double _chi_n;
        private readonly Vector<double> _weights;
        private Vector<double> _p_sigma;
        private Vector<double> _pc;
        private Vector<double> _mean;
        private Matrix<double> _C;
        private double _sigma;
        private Vector<double> _D;
        private Matrix<double> _B;
        private Matrix<double> _bounds;
        private readonly int _n_max_resampling;
        private readonly Normal _normalDistribution;
        private readonly double _epsilon;
        private readonly double _tol_sigma;
        private readonly double _tol_C;
        private readonly int _funhist_term;

        private const double _tolxup = 1e4;
        private const double _tolfun = 1e-12;
        private const double _tolconditioncov = 1e14;

        private readonly double _tolx;

        private double[] _funhist_values;

        public int Dim { get; }

        public int PopulationSize { get; private set; }

        public int Generation { get; private set; }


        public CMA(IList<double> mean, double sigma, Matrix<double> bounds = null, int nMaxResampling = 100, int seed = 0, double tol_sigma = 1e-4, double tol_C = 1e-4)
        {


            if (!(sigma > 0))
            {
                throw new ArgumentOutOfRangeException("sigma must be non-zero positive value");
            }

            int nDim = mean.Count;
            if (!(nDim > 1))
            {
                throw new ArgumentOutOfRangeException("The dimension of mean must be larger than 1");
            }

            int populationSize = 4 + (int)Math.Floor(3 * Math.Log(nDim));  // # (eq. 48)

            int mu = populationSize / 2;

            Vector<double> weightsPrime = Vector<double>.Build.Dense(populationSize);
            for (int i = 0; i < populationSize; i++)
            {
                weightsPrime[i] = Math.Log((populationSize + 1) / (double)2) - Math.Log(i + 1);
            }

            Vector<double> weightsPrimeMuEff = Vector<double>.Build.Dense(weightsPrime.Take(mu).ToArray());
            double mu_eff = Math.Pow(weightsPrimeMuEff.Sum(), 2) / Math.Pow(weightsPrimeMuEff.L2Norm(), 2);
            Vector<double> weightsPrimeMuEffMinus = Vector<double>.Build.Dense(weightsPrime.Skip(mu).ToArray());
            double muEffMinus = Math.Pow(weightsPrimeMuEffMinus.Sum(), 2) / Math.Pow(weightsPrimeMuEffMinus.L2Norm(), 2);

            int alphacCov = 2;
            double c1 = alphacCov / (Math.Pow(nDim + 1.3, 2) + mu_eff);
            double cmu = Math.Min(1 - c1, alphacCov * (mu_eff - 2 + 1 / mu_eff) / (Math.Pow(nDim + 2, 2) + alphacCov * mu_eff / 2));
            if (!(c1 <= 1 - cmu))
            {
                throw new Exception("invalid learning rate for the rank-one update");
            }
            if (!(cmu <= 1 - c1))
            {
                throw new Exception("invalid learning rate for the rank-μ update");
            }

            double minAlpha = Math.Min(1 + c1 / cmu, Math.Min(1 + 2 * muEffMinus / (mu_eff + 2), (1 - c1 - cmu) / (nDim * cmu)));

            double positiveSum = weightsPrime.Where(x => x > 0).Sum();
            double negativeSum = Math.Abs(weightsPrime.Where(x => x < 0).Sum());

            Vector<double> weights = Vector<double>.Build.Dense(weightsPrime.Count);
            weightsPrime.CopyTo(weights);
            bool[] weightsIsNotNegative = weightsPrime.Select(x => x >= 0).ToArray();
            for (int i = 0; i < weights.Count; i++)
            {
                weights[i] = weightsIsNotNegative[i] ? 1 / positiveSum * weightsPrime[i] : minAlpha / negativeSum * weightsPrime[i];
            }
            int cm = 1;

            double c_sigma = (mu_eff + 2) / (nDim + mu_eff + 5);
            double d_sigma = 1 + 2 * Math.Max(0, Math.Sqrt((mu_eff - 1) / (nDim + 1)) - 1) + c_sigma;
            if (!(c_sigma < 1))
            {
                throw new Exception("invalid learning rate for cumulation for the step-size control");
            }

            double cc = (4 + mu_eff / nDim) / (nDim + 4 + 2 * mu_eff / nDim);
            if (!(cc <= 1))
            {
                throw new Exception("invalid learning rate for cumulation for the rank-one update");
            }

            Dim = nDim;
            PopulationSize = populationSize;
            _mu = mu;
            _mu_eff = mu_eff;

            _cc = cc;
            _c1 = c1;
            _cmu = cmu;
            _c_sigma = c_sigma;
            _d_sigma = d_sigma;
            _cm = cm;

            _chi_n = Math.Sqrt(Dim) * (1.0 - 1.0 / (4.0 * Dim) + 1.0 / (21.0 * Math.Pow(Dim, 2)));

            _weights = weights;

            _p_sigma = Vector<double>.Build.Dense(Dim, 0);
            _pc = Vector<double>.Build.Dense(Dim, 0);

            _mean = Vector<double>.Build.DenseOfArray(mean.ToArray());
            _C = Matrix<double>.Build.DenseIdentity(Dim, Dim);
            _sigma = sigma;

            if (!(bounds == null || bounds.RowCount == Dim && bounds.ColumnCount == 2))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }
            _bounds = bounds;
            _n_max_resampling = nMaxResampling;

            Generation = 0;

            _normalDistribution = new Normal(0, 1);

            _epsilon = 1e-8;

            _tol_sigma = tol_sigma;
            _tol_C = tol_C;

            _funhist_term = 10 + (int)Math.Ceiling(30.0 * Dim / PopulationSize);
            _funhist_values = new double[_funhist_term * 2];

            _tolx = 1e-12 * _sigma;
        }

        public (Matrix<double> B, Vector<double> D) EigenDecomposition()
        {
            if (_B != null && _D != null)
                return (_B, _D);

            _C = (_C + _C.Transpose()) / 2;
            Evd<double> evd_C = _C.Evd();
            Matrix<double> B = evd_C.EigenVectors;
            Vector<double> D = Vector<double>.Build.DenseOfArray(evd_C.EigenValues.Select(x => RootTemp(x.Real)).ToArray());

            var diagSquaredD = Matrix<double>.Build.DiagonalOfDiagonalArray(D.AsArray().Select(x => x * x).ToArray());
            var bTimesDiagSquaredD = B * diagSquaredD;
            var bTimesDiagSquaredDTimesBT = bTimesDiagSquaredD * B.Transpose();
            _C = bTimesDiagSquaredDTimesBT;
            return (B, D);
        }

        public static (Matrix<double> B, Vector<double> D) EigenDecomposition2(Matrix<double> C)
        {

            C = (C + C.Transpose()) / 2;
            Evd<double> evd_C = C.Evd();
            Matrix<double> B = evd_C.EigenVectors;
            Vector<double> D = Vector<double>.Build.DenseOfArray(evd_C.EigenValues.Select(x => RootTemp(x.Real)).ToArray());

            var diagSquaredD = Matrix<double>.Build.DiagonalOfDiagonalArray(D.AsArray().Select(x => x * x).ToArray());
            var bTimesDiagSquaredD = B * diagSquaredD;
            var bTimesDiagSquaredDTimesBT = bTimesDiagSquaredD * B.Transpose();
            C = bTimesDiagSquaredDTimesBT;
            return (B, D);
        }

        static double RootTemp(double input)
        {
            return input >= 0 ? Math.Sqrt(input) : Math.Sqrt(1e-8);
        }

        public bool ShouldStop()
        {

            (Matrix<double> B, Vector<double> D) = EigenDecomposition();

            var dC = _C.Diagonal();

            if (Generation > _funhist_term && _funhist_values.Max() - _funhist_values.Min() < _tolfun)
            {
                return true;
            }

            if (dC.All(x => _sigma * x < _tolx) && _pc.All(x => _sigma * x < _tolx))
            {
                return true;
            }

            if (_sigma * D.Max() > _tolxup)
            {
                return true;
            }

            if (dC.Any(x => _mean == _mean + 0.2 * _sigma * Math.Sqrt(x)))
            {
                return true;
            }

            int i = Generation / Dim;
            if (D.Select((x, i) => (x, i)).All(entry => _mean == _mean + 0.1 * _sigma * entry.x * B.Row(entry.i)))
            {
                return true;
            }

            double condition_cov = D.Max() / D.Min();
            if (condition_cov > _tolconditioncov)
            {
                return true;
            }

            return false;
        }

        public bool IsConverged()
        {
            return _sigma < _tol_sigma && _C.L2Norm() < _tol_C;
        }

        public void SetBounds(Matrix<double> bounds = null)
        {
            if (!(bounds == null || bounds.RowCount == Dim && bounds.ColumnCount == 2))
            {
                throw new Exception("bounds should be (n_dim, 2)-dim matrix");
            }
            _bounds = bounds;
        }

        public Vector<double> Ask()
        {
            for (int i = 0; i < _n_max_resampling; i++)
            {
                Vector<double> x = SampleSolution();
                if (IsFeasible(x)) { return x; }
            }
            Vector<double> xNew = SampleSolution();
            xNew = RepairInfeasibleParams(xNew);
            return xNew;
        }


        public void Tell(List<Solution> solutions)
        {
            if (solutions.Count != PopulationSize)
            {
                throw new ArgumentException("Must tell popsize-length solutions.");
            }

            Generation += 1;
            var sortedSolutions = solutions.OrderBy(x => x.Fitness).ToArray();

            int funhist_idx = 2 * (Generation % _funhist_term);
            _funhist_values[funhist_idx] = sortedSolutions.First().Fitness;
            _funhist_values[funhist_idx + 1] = sortedSolutions.Last().Fitness;

            // Sample new population of search_points, for k=1, ..., popsize
            (Matrix<double> B, Vector<double> D) = EigenDecomposition();

            _B = null;
            _D = null;

            Matrix<double> x_k = Matrix<double>.Build.DenseOfRowVectors(sortedSolutions.Select(x => x.Parameters));
            var yKRows = x_k.EnumerateRows().Select((x, i) => (x - _mean) / _sigma).ToArray();
            Matrix<double> y_k = Matrix<double>.Build.DenseOfRowVectors(yKRows);

            // Selection and recombination
            Vector<double> y_w = PointwiseMultiplicationOnRows(y_k.SubMatrix(0, _mu, 0, Dim).Transpose(), _weights.SubVector(0, _mu));
            _mean += _cm * _sigma * y_w;

            var diag = Matrix<double>.Build.DenseOfDiagonalArray((1.0 / D).AsArray());
            Matrix<double> C_2 = B * diag * B.Transpose();

            double sqrtFactor = Math.Sqrt(_c_sigma * (2 - _c_sigma) * _mu_eff);
            var g = sqrtFactor * (C_2 * y_w);
            var q = C_2 * y_w;
            _p_sigma = (1 - _c_sigma) * _p_sigma + sqrtFactor * (C_2 * y_w);

            double norm_pSigma = _p_sigma.L2Norm();
            _sigma *= Math.Exp(_c_sigma / _d_sigma * (norm_pSigma / _chi_n - 1));
            //Console.WriteLine($"Sigma: | {string.Join(", ", _p_sigma)} | {norm_pSigma}, {Math.Exp(_c_sigma / _d_sigma * (norm_pSigma / _chi_n - 1))}, {_sigma}");
            double h_sigma_cond_left = norm_pSigma / Math.Sqrt(1 - Math.Pow(1 - _c_sigma, 2 * (Generation + 1)));
            double h_sigma_cond_right = (1.4 + 2 / (double)(Dim + 1)) * _chi_n;
            double h_sigma = h_sigma_cond_left < h_sigma_cond_right ? 1.0 : 0.0;

            _pc = (1 - _cc) * _pc + h_sigma * Math.Sqrt(_cc * (2 - _cc) * _mu_eff) * y_w;

            Vector<double> w_io = Vector<double>.Build.Dense(_weights.Count, 1);
            Vector<double> w_iee = (C_2 * y_k.Transpose()).ColumnNorms(2).PointwisePower(2);
            for (int i = 0; i < _weights.Count; i++)
            {
                if (_weights[i] >= 0)
                {
                    w_io[i] = _weights[i] * 1;
                }
                else
                {
                    w_io[i] = _weights[i] * Dim / (w_iee[i] + _epsilon);
                }
            }

            double delta_h_sigma = (1 - h_sigma) * _cc * (2 - _cc);
            if (!(delta_h_sigma <= 1))
            {
                throw new Exception("invalid value of delta_h_sigma");
            }

            Matrix<double> rank_one = _pc.OuterProduct(_pc);
            Matrix<double> rank_mu = Matrix<double>.Build.Dense(y_k.ColumnCount, y_k.ColumnCount, 0);
            for (int i = 0; i < w_io.Count; i++)
            {
                rank_mu += w_io[i] * y_k.Row(i).OuterProduct(y_k.Row(i));
            }
            _C =
                    (
                    1
                    + _c1 * delta_h_sigma
                    - _c1
                    - _cmu * _weights.Sum()
                    )
                    * _C
                    + _c1 * rank_one
                    + _cmu * rank_mu
                ;
        }

        private Vector<double> RepairInfeasibleParams(Vector<double> param)
        {
            if (_bounds == null)
            {
                return param;
            }
            Vector<double> newParam = param.PointwiseMaximum(_bounds.Column(0));
            newParam = newParam.PointwiseMinimum(_bounds.Column(1));
            return newParam;
        }

        private bool IsFeasible(Vector<double> param)
        {
            if (_bounds == null)
            {
                return true;
            }
            bool isCorrectLower = true;
            bool isCorrectUpper = true;
            for (int i = 0; i < param.Count; i++)
            {
                isCorrectLower &= param[i] >= _bounds[i, 0];
                isCorrectUpper &= param[i] <= _bounds[i, 1];
            }
            return isCorrectLower & isCorrectUpper;
        }

        private Vector<double> SampleSolution()
        {
            (Matrix<double> B, Vector<double> D) = EigenDecomposition();

            Vector<double> z = Vector<double>.Build.Dense(Dim);
            for (int i = 0; i < z.Count; i++)
            {
                z[i] = _normalDistribution.Sample();
            }
            var h = B * Matrix<double>.Build.DenseOfDiagonalArray(D.AsArray());
            var y = PointwiseMultiplicationOnRows(h, z);
            Vector<double> x = _mean + _sigma * y;
            return x;
        }

        private Vector<double> PointwiseMultiplicationOnRows(Matrix<double> matrix, Vector<double> vector)
        {
            var rows = matrix.EnumerateRows().Select(x => x.PointwiseMultiply(vector)).ToArray();
            return Vector<double>.Build.DenseOfArray(rows.Select(x => x.Sum()).ToArray());
        }
    }
}
