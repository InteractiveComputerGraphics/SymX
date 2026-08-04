#include "FirstOrderAlgorithm.h"

#include <cmath>
#include <stdexcept>

using namespace symx;

void FirstOrderAlgorithm::prepare(
    int n_variables,
    const std::vector<int32_t>& dof_set_offsets,
    const std::vector<int32_t>& dof_set_sizes,
    const FirstOrderSettings& settings)
{
    if (n_variables <= 0) {
        throw std::invalid_argument("First-order optimization requires at least one variable.");
    }
    if (settings.type == FirstOrderOptType::LBFGS && settings.lbfgs.history_size <= 0) {
        throw std::invalid_argument("L-BFGS requires a positive history_size.");
    }

    this->_prepare_vector_adam_blocks(dof_set_offsets, dof_set_sizes, settings);
    this->reset(n_variables);
}

void FirstOrderAlgorithm::reset(int n_variables)
{
    this->adam_t = 0;
    this->adam_m.setZero(n_variables);
    this->adam_v.setZero(n_variables);

    this->vector_adam_t = 0;
    this->vector_adam_m.setZero(n_variables);
    this->vector_adam_v.setZero((int)this->vector_adam_block_offsets.size());

    this->reset_lbfgs_history();
    this->direction.resize(n_variables);
}

const Eigen::VectorXd& FirstOrderAlgorithm::compute_direction(
    const Eigen::VectorXd& gradient,
    const FirstOrderSettings& settings)
{
    switch (settings.type) {
        case FirstOrderOptType::GradientDescent: this->_compute_gradient_descent(gradient, settings); break;
        case FirstOrderOptType::Adam:            this->_compute_adam(gradient, settings); break;
        case FirstOrderOptType::VectorAdam:      this->_compute_vector_adam(gradient, settings); break;
        case FirstOrderOptType::LBFGS:           this->_compute_lbfgs(gradient, settings); break;
    }
    return this->direction;
}

Eigen::VectorXd FirstOrderAlgorithm::default_direction(
    const Eigen::VectorXd& gradient,
    const FirstOrderSettings& settings) const
{
    if (settings.type != FirstOrderOptType::LBFGS) {
        return -gradient;
    }

    const double gradient_max = gradient.cwiseAbs().maxCoeff();
    const double scale = (gradient_max > 0.0)
        ? settings.lbfgs.bootstrap_step_length / gradient_max
        : 1.0;
    return -scale * gradient;
}

bool FirstOrderAlgorithm::update_lbfgs_history(
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y,
    const FirstOrderSettings& settings)
{
    const double ys = y.dot(s);
    if (ys <= settings.lbfgs.curvature_threshold) {
        this->lbfgs_history_skip_count++;
        return false;
    }

    this->lbfgs_history_skip_count = 0;
    this->lbfgs_s.push_back(s);
    this->lbfgs_y.push_back(y);
    this->lbfgs_rho.push_back(1.0 / ys);
    if ((int)this->lbfgs_s.size() > settings.lbfgs.history_size) {
        this->lbfgs_s.pop_front();
        this->lbfgs_y.pop_front();
        this->lbfgs_rho.pop_front();
    }
    return true;
}

void FirstOrderAlgorithm::reset_lbfgs_history()
{
    this->lbfgs_alpha.clear();
    this->lbfgs_s.clear();
    this->lbfgs_y.clear();
    this->lbfgs_rho.clear();
    this->lbfgs_history_skip_count = 0;
}

int FirstOrderAlgorithm::get_lbfgs_history_skip_count() const
{
    return this->lbfgs_history_skip_count;
}

void FirstOrderAlgorithm::_prepare_vector_adam_blocks(
    const std::vector<int32_t>& dof_set_offsets,
    const std::vector<int32_t>& dof_set_sizes,
    const FirstOrderSettings& settings)
{
    this->vector_adam_block_offsets.clear();
    if (settings.type != FirstOrderOptType::VectorAdam) {
        return;
    }

    const int block_size = settings.vector_adam.vector_block_size;
    if (block_size <= 0) {
        throw std::invalid_argument("VectorAdam requires a positive vector_block_size.");
    }
    if (dof_set_offsets.size() != dof_set_sizes.size() + 1) {
        throw std::invalid_argument("First-order optimizer received inconsistent DoF-set metadata.");
    }

    for (int set_i = 0; set_i < (int)dof_set_sizes.size(); ++set_i) {
        const int n_set_dofs = dof_set_sizes[set_i];
        if (n_set_dofs % block_size != 0) {
            throw std::invalid_argument(
                "VectorAdam requires every DoF set to be divisible by vector_block_size.");
        }
        for (int offset = 0; offset < n_set_dofs; offset += block_size) {
            this->vector_adam_block_offsets.push_back(dof_set_offsets[set_i] + offset);
        }
    }
}

void FirstOrderAlgorithm::_compute_gradient_descent(
    const Eigen::VectorXd& gradient,
    const FirstOrderSettings& settings)
{
    this->direction = -settings.gradient_descent.learning_rate * gradient;
}

void FirstOrderAlgorithm::_compute_adam(
    const Eigen::VectorXd& gradient,
    const FirstOrderSettings& settings)
{
    const auto& adam = settings.adam;
    this->adam_t++;
    this->adam_m = adam.beta1 * this->adam_m + (1.0 - adam.beta1) * gradient;
    this->adam_v = adam.beta2 * this->adam_v + (1.0 - adam.beta2) * gradient.cwiseProduct(gradient);

    const double bias1 = 1.0 - std::pow(adam.beta1, this->adam_t);
    const double bias2 = 1.0 - std::pow(adam.beta2, this->adam_t);
    const Eigen::VectorXd m_hat = this->adam_m / bias1;
    const Eigen::VectorXd v_hat = this->adam_v / bias2;
    this->direction = -adam.learning_rate * m_hat.cwiseQuotient(
        v_hat.cwiseSqrt() + adam.epsilon * Eigen::VectorXd::Ones(gradient.size()));
}

void FirstOrderAlgorithm::_compute_vector_adam(
    const Eigen::VectorXd& gradient,
    const FirstOrderSettings& settings)
{
    const auto& adam = settings.vector_adam;
    this->vector_adam_t++;
    this->vector_adam_m = adam.beta1 * this->vector_adam_m + (1.0 - adam.beta1) * gradient;
    this->direction.setZero(gradient.size());

    const double bias1 = 1.0 - std::pow(adam.beta1, this->vector_adam_t);
    const double bias2 = 1.0 - std::pow(adam.beta2, this->vector_adam_t);
    for (int block_i = 0; block_i < (int)this->vector_adam_block_offsets.size(); ++block_i) {
        const int offset = this->vector_adam_block_offsets[block_i];
        const double gradient_norm_squared = gradient.segment(offset, adam.vector_block_size).squaredNorm();
        this->vector_adam_v[block_i] = adam.beta2 * this->vector_adam_v[block_i]
            + (1.0 - adam.beta2) * gradient_norm_squared;
        this->direction.segment(offset, adam.vector_block_size) = -adam.learning_rate
            * (this->vector_adam_m.segment(offset, adam.vector_block_size) / bias1)
            / (std::sqrt(this->vector_adam_v[block_i] / bias2) + adam.epsilon);
    }
}

void FirstOrderAlgorithm::_compute_lbfgs(
    const Eigen::VectorXd& gradient,
    const FirstOrderSettings& settings)
{
    const int history_size = (int)this->lbfgs_s.size();
    if (history_size == 0) {
        this->direction = this->default_direction(gradient, settings);
        return;
    }

    Eigen::VectorXd q = gradient;
    this->lbfgs_alpha.resize(history_size);
    for (int i = history_size - 1; i >= 0; --i) {
        this->lbfgs_alpha[i] = this->lbfgs_rho[i] * this->lbfgs_s[i].dot(q);
        q.noalias() -= this->lbfgs_alpha[i] * this->lbfgs_y[i];
    }

    const Eigen::VectorXd& s_last = this->lbfgs_s.back();
    const Eigen::VectorXd& y_last = this->lbfgs_y.back();
    const double yy = y_last.squaredNorm();
    const double gamma = (yy > 0.0) ? s_last.dot(y_last) / yy : 1.0;
    Eigen::VectorXd result = gamma * q;

    for (int i = 0; i < history_size; ++i) {
        const double beta = this->lbfgs_rho[i] * this->lbfgs_y[i].dot(result);
        result.noalias() += this->lbfgs_s[i] * (this->lbfgs_alpha[i] - beta);
    }
    this->direction = -result;
}
