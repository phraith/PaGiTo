#include "cmaes/cmaes.h"
#include <cmath>

Cmaes::Cmaes(std::vector<double> mean, double sigma, std::vector<double> upper,
             std::vector<double> lower, int n_max_resampling, int seed, double tol_sigma, double tol_C)
        :
        sigma_(sigma),
        n_max_resampling_(n_max_resampling),
        tolx_(1e-12 * sigma_),
        tol_sigma_(tol_sigma),
        tol_C_(tol_C),
        dim_(mean.size()),
        bounds_(CreateBoundsMatrix(dim_, upper, lower)),
        population_size_(4 + (int) std::floor(3.0 * std::log(dim_))), // # (eq. 48)
        generation_(0),
        mu_(population_size_ / 2),
        cm_(1),
        weights_information_(CreateInitialWeights(population_size_, mu_, dim_)),
        c_sigma_(CalculateInitialCSigma(weights_information_.MuEff(), dim_)),
        cc_(CalculateInitialCc(weights_information_.MuEff(), dim_)),
        d_sigma_(CalculateInitialDSigma(c_sigma_, weights_information_.MuEff(), dim_)),
        chi_n_(CalculateInitialChiN(dim_)),
        p_sigma_(Eigen::VectorX<double>::Zero(dim_)),
        pc_(Eigen::VectorX<double>::Zero(dim_)),
        C_(std::make_shared<Eigen::MatrixX<double>>(Eigen::MatrixX<double>::Identity(dim_, dim_))),
        mean_(Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(mean.data(), mean.size())),
        funhist_term_(10 + (int) std::ceil(30.0 * dim_ / population_size_)),
        function_history_(std::vector<double>(funhist_term_ * 2)) {

    if (sigma <= 0) {
        throw std::exception();
    }

    if (dim_ <= 1) {
        throw std::exception();
    }

    if (c_sigma_ >= 1) {
        throw std::exception();
    }

    if (cc_ > 1) {
        throw std::exception();
    }

    EigenDecomposition();
}


void
Cmaes::EigenDecomposition() {
    *C_ = (*C_ + C_->transpose()) / 2;

    Eigen::EigenSolver<Eigen::MatrixX<double>> solver(*C_, true);
    Eigen::MatrixXcd eigenvalues = solver.eigenvalues();
    std::vector<double> extracted_eigenvalues(eigenvalues.size());
    for (size_t i = 0; i < eigenvalues.size(); i++) {
        auto complex_scalar = eigenvalues(i);
        double rooted_real = RootTemp(complex_scalar.real());
        extracted_eigenvalues[i] = rooted_real;
    }

    D_ = std::make_shared<Eigen::VectorXd>(Eigen::Map<Eigen::VectorXd,
            Eigen::Unaligned>(extracted_eigenvalues.data(), extracted_eigenvalues.size()));

    Eigen::MatrixXd diag_squared_d = D_->cwiseProduct(*D_).asDiagonal();
    B_ = std::make_shared<Eigen::MatrixX<double>>(
            solver.eigenvectors().real());
    Eigen::MatrixXd b_times_diag_squared_d = *B_ * diag_squared_d;
    Eigen::MatrixXd b_times_diag_squared_d_times_bt = b_times_diag_squared_d * B_->transpose();

    *C_ = b_times_diag_squared_d_times_bt;
}

double Cmaes::RootTemp(double input) {
    return input >= 0 ? std::sqrt(input) : std::sqrt(1e-8);
}

bool Cmaes::ShouldStop() {
//    EigenDecomposition();
    auto dC = C_->asDiagonal();

    const auto [min, max] = std::minmax_element(std::begin(function_history_), std::end(function_history_));
    if (generation_ > funhist_term_ && *max - *min < tolfun_) {
        return true;
    }

//    bool dc_is_small_enough = true;
//    for (size_t i = 0; i < dC.size(); i++) {
//        auto value = dC.diagonal()(i);
//        if (sigma_ * value >= tolx_) {
//            dc_is_small_enough = false;
//            break;
//        }
//    }
//
//    bool pc_is_small_enough = true;
//    for (size_t i = 0; i < pc_.size(); i++) {
//        auto value = pc_.diagonal()(i);;
//        if (sigma_ * value >= tolx_) {
//            pc_is_small_enough = false;
//            break;
//        }
//    }
//
//    if (dc_is_small_enough && pc_is_small_enough) {
//        return true;
//    }

    if (sigma_ * D_->maxCoeff() > tolxup_) {
        return true;
    }

//    for (size_t i = 0; i < dC.size(); i++) {
//        double value = dC.diagonal()(i);
//        if (!mean_.array().isApprox(mean_.array() + 0.2 * sigma_ * std::sqrt(value))) {
//            return true;
//        }
//    }

    int factor = generation_ % dim_;
    bool should_stop = true;
    double d_value = D_->array()[factor];
    for (size_t i = 0; i < B_->col(factor).size(); i++) {
        double value = B_->col(factor)[i];
        if (!mean_.array().isApprox(mean_.array() + 0.1 * sigma_ * value * d_value)) {
            should_stop = false;
            break;
        }
    }

    if (should_stop) {
        return true;
    }

    double condition_cov = D_->maxCoeff() / D_->minCoeff();
    if (condition_cov > tolconditioncov_) {
        return true;
    }
    return false;
}

bool Cmaes::IsConverged() {
    return sigma_ < tol_sigma_ && C_->norm() < tol_C_;
}

Eigen::MatrixXd Cmaes::CreateBoundsMatrix(int dim, std::vector<double> upper, std::vector<double> lower) {
    Eigen::MatrixX<double> bounds = Eigen::MatrixX<double>::Constant(2, dim, 0);
    if (upper.empty() || lower.empty() || upper.size() != dim || lower.size() != dim) {
        return bounds;
    }

    bounds.row(0) = Eigen::Map<Eigen::VectorXd,
            Eigen::Unaligned>(upper.data(), upper.size());
    bounds.row(1) = Eigen::Map<Eigen::VectorXd,
            Eigen::Unaligned>(lower.data(), lower.size());
    return bounds;
}

Eigen::VectorX<double> Cmaes::Ask() const {
    for (int i = 0; i < n_max_resampling_; i++) {
        auto x = SampleSolution();
        if (IsFeasable(x)) { return x; }
    }
    auto x_new = SampleSolution();
    x_new = RepairInfeasableParams(x_new);
    return x_new;
}

Eigen::VectorX<double> Cmaes::SampleSolution() const {
//    std::tuple<std::shared_ptr<Eigen::MatrixX<double>>, std::shared_ptr<Eigen::VectorX<double>>> bd_tuple = EigenDecomposition();
//    auto B = std::get<std::shared_ptr<Eigen::MatrixX<double>>>(bd_tuple);
//    auto D = std::get<std::shared_ptr<Eigen::VectorX<double>>>(bd_tuple);

    std::vector<double> vector_data(dim_);
    static thread_local std::mt19937 gen = std::mt19937{std::random_device{}()};
    std::normal_distribution distribution(0.0, 1.0);
    for (int i = 0; i < dim_; ++i) {
        vector_data[i] = distribution(gen);
    }

    Eigen::VectorX<double> z = Eigen::Map<Eigen::VectorXd,
            Eigen::Unaligned>(vector_data.data(), vector_data.size());

    auto h = *B_ * D_->asDiagonal();
    auto y = PointwiseMultiplicationOnRows(h, z);
    auto x = mean_ + sigma_ * y;

    std::vector<int> vec(x.size());
    for (int i = 0; i < x.size(); ++i) {
        vec.at(i) = x[i];
    }

    std::vector<int> vec2(x.size());
    for (int i = 0; i < mean_.size(); ++i) {
        vec2.at(i) = mean_[i];
    }


    return x;
}

Eigen::VectorX<double>
Cmaes::PointwiseMultiplicationOnRows(Eigen::MatrixX<double> matrix, Eigen::VectorX<double> vector) {
    std::vector<double> t(matrix.rows());
    for (int i = 0; i < matrix.rows(); ++i) {
        auto value = matrix.row(i);
        auto m = value * vector;
        auto sum = m.sum();
        t[i] = sum;
    }

    Eigen::VectorX<double> z = Eigen::Map<Eigen::VectorXd,
            Eigen::Unaligned>(t.data(), t.size());

    return z;
}

bool Cmaes::IsFeasable(Eigen::VectorX<double> param) const {
    if (bounds_.isZero(0)) {
        return true;
    }

    bool isCorrectLower = true;
    bool isCorrectUpper = true;
    for (int i = 0; i < param.size(); i++) {
        isCorrectLower &= param[i] >= bounds_.coeff(i, 0);
        isCorrectUpper &= param[i] <= bounds_.coeff(i, 1);
    }
    return isCorrectLower & isCorrectUpper;
}

Eigen::VectorX<double> Cmaes::RepairInfeasableParams(Eigen::VectorX<double> params) const {
    if (bounds_.isZero(0)) {
        return params;
    }

    Eigen::VectorX<double> newParam = params.transpose().cwiseMax(bounds_.row(1));
    newParam = newParam.transpose().cwiseMin(bounds_.row(0));

    std::vector<double> vec(params.size());
    for (int i = 0; i < params.size(); ++i) {
        vec.at(i) = newParam[i];
    }

    std::vector<double> vec2(newParam.size());
    for (int i = 0; i < newParam.size(); ++i) {
        vec2.at(i) = newParam[i];
    }

    return newParam;
}

void Cmaes::Tell(std::vector<Solution> &solutions) {
    if (solutions.size() != population_size_) {
        throw std::exception();
    }

    generation_ += 1;
    std::sort(solutions.begin(), solutions.end());
    int function_history_idx = 2 * (generation_ % funhist_term_);
    function_history_[function_history_idx] = solutions[0].Fitness();
    function_history_[function_history_idx + 1] = solutions[solutions.size() - 1].Fitness();

    EigenDecomposition();

    Eigen::MatrixX<double> x_k = Eigen::MatrixX<double>::Ones(solutions.size(), solutions[0].Parameters().size());
    for (int i = 0; i < solutions.size(); ++i) {
        x_k.row(i) = solutions[i].Parameters();
    }

    std::vector<Eigen::VectorX<double>> y_k_rows(x_k.rows());
    for (int i = 0; i < x_k.rows(); ++i) {
        Eigen::VectorX<double> row = x_k.row(i);
        Eigen::VectorX<double> new_vec = (row - mean_) / sigma_;
        y_k_rows[i] = new_vec;
    }

    Eigen::MatrixX<double> y_k = Eigen::MatrixX<double>::Ones(y_k_rows.size(), y_k_rows[0].size());
    for (int i = 0; i < solutions.size(); ++i) {
        y_k.row(i) = y_k_rows[i];
    }

    auto y_w = PointwiseMultiplicationOnRows(y_k.block(0, 0, mu_, dim_).transpose(),
                                             weights_information_.Weights().block(0, 0, mu_, 1));
    mean_ += cm_ * sigma_ * y_w;
    auto diag = (1.0 / D_->array()).matrix().asDiagonal();
    auto C_2 = *B_ * diag * B_->transpose();
    double sqrt_factor = std::sqrt(c_sigma_ * (2.0 - c_sigma_) * weights_information_.MuEff());

    p_sigma_ = (1.0 - c_sigma_) * p_sigma_ + sqrt_factor * (C_2 * y_w);
    double norm_psigma = p_sigma_.norm();
    sigma_ *= std::exp(c_sigma_ / d_sigma_ * (norm_psigma / chi_n_ - 1.0));

    double h_sigma_cond_left = norm_psigma / std::sqrt(1.0 - std::pow(1.0 - c_sigma_, 2.0 * (generation_ + 1)));
    double h_sigma_cond_right = (1.4 + 2.0 / (double) (dim_ + 1)) * chi_n_;
    double h_sigma = h_sigma_cond_left < h_sigma_cond_right ? 1.0 : 0.0;

    pc_ = (1 - cc_) * pc_ + h_sigma * std::sqrt(cc_ * (2.0 - cc_) * weights_information_.MuEff()) * y_w;

    Eigen::VectorX<double> w_io = Eigen::VectorX<double>::Ones(weights_information_.Weights().size());
    Eigen::VectorX<double> w_iee = (C_2 * y_k.transpose()).colwise().norm();
    w_iee = w_iee.array().square();

    for (int i = 0; i < weights_information_.Weights().size(); i++) {
        if (weights_information_.Weights()(i) >= 0) {
            w_io[i] = weights_information_.Weights()(i) * 1.0;
        } else {
            w_io[i] = weights_information_.Weights()(i) * dim_ / (w_iee(i) + epsilon_);
        }
    }

    double delta_h_sigma = (1.0 - h_sigma) * cc_ * (2.0 - cc_);
    if (delta_h_sigma > 1.0) {
        throw std::exception();
    }

    Eigen::MatrixX<double> rank_one = pc_ * pc_.transpose();
    Eigen::MatrixX<double> rank_mu = Eigen::MatrixX<double>::Zero(y_k.cols(), y_k.cols());
    for (int i = 0; i < w_io.size(); i++) {
        double scalar = w_io[i];
        Eigen::MatrixX<double> temp = (y_k.row(i).transpose() * y_k.row(i));
        rank_mu += scalar * temp;
    }

    C_ = std::make_shared<Eigen::MatrixX<double>>(
            (
                    1.0
                    + weights_information_.C1() * delta_h_sigma
                    - weights_information_.C1()
                    - weights_information_.Cmu() * weights_information_.Weights().sum()
            )
            * *C_
            + weights_information_.C1() * rank_one
            + weights_information_.Cmu() * rank_mu);
}

int Cmaes::PopulationSize() const {
    return population_size_;
}

WeightsInformation Cmaes::CreateInitialWeights(int population_size, int mu, int dim) {

    Eigen::VectorX<double> weightsPrime(population_size);
    for (int i = 0; i < population_size; i++) {
        weightsPrime[i] = std::log((population_size + 1) / (double) 2) - std::log(i + 1);
    }

    Eigen::VectorX<double> weightsPrimeMuEff = weightsPrime.block(0, 0, mu, 1);
    int alpha_cov = 2;

    double mu_eff = std::pow(weightsPrimeMuEff.sum(), 2) / std::pow(weightsPrimeMuEff.norm(), 2);
    double c1 = alpha_cov / (std::pow(dim + 1.3, 2) + mu_eff);
    double cmu = std::min(1 - c1,
                          alpha_cov * (mu_eff - 2 + 1 / mu_eff) / (std::pow(dim + 2, 2) + alpha_cov * mu_eff / 2));

    if (c1 > 1 - cmu || cmu > 1 - c1) {
        throw std::exception();
    }

    Eigen::VectorX<double> weightsPrimeMuEffMinus = weightsPrime.block(mu + 1, 0, weightsPrime.rows() - (mu + 1), 1);
    double muEffMinus = std::pow(weightsPrimeMuEffMinus.sum(), 2) /
                        std::pow(weightsPrimeMuEffMinus.norm(), 2);


    double minAlpha = std::min(1 + c1 / cmu,
                               std::min(1 + 2 * muEffMinus / (mu_eff + 2), (1 - c1 - cmu) / (dim * cmu)));

    double positiveSum = 0;
    double negativeSum = 0;

    for (int i = 0; i < weightsPrime.size(); ++i) {
        auto value = weightsPrime(i);
        if (value >= 0) { positiveSum += value; }
        else { negativeSum += std::abs(value); }
    }

    Eigen::VectorX<double> weights = weightsPrime;

    for (int i = 0; i < weights.size(); ++i) {
        auto value = weights(i);
        if (value >= 0) {
            weights(i) = 1.0 / positiveSum * value;
        } else {
            weights(i) = minAlpha / negativeSum * value;
        }
    }

    return {weights, c1, cmu, mu_eff};
}

double Cmaes::CalculateInitialCSigma(double mu_eff, int dim) {
    double c_sigma = (mu_eff + 2) / (dim + mu_eff + 5);
    return c_sigma;
}

double Cmaes::CalculateInitialDSigma(double c_sigma, double mu_eff, int dim) {
    double d_sigma = 1 + 2 * std::max(0.0, std::sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma;
    return d_sigma;
}

double Cmaes::CalculateInitialCc(double mu_eff, int dim) {
    double cc = (4 + mu_eff / dim) / (dim + 4 + 2.0 * mu_eff / dim);
    return cc;
}

double Cmaes::CalculateInitialChiN(int dim) {
    double chi_n = std::sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * std::pow(dim, 2)));
    return chi_n;
}
