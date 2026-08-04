#include "solver_utils.h"

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <vector>
#ifdef SYMX_ENABLE_MKL
#include <Eigen/PardisoSupport>
#include <mkl.h>
#endif

namespace symx
{
    namespace
    {
        using SparseMatrix = Eigen::SparseMatrix<double>;
        using StorageIndex = SparseMatrix::StorageIndex;

        struct SparsityPattern
        {
            int rows = 0;
            int cols = 0;
            int nonzeros = 0;
            std::vector<StorageIndex> outer_indices;
            std::vector<StorageIndex> inner_indices;

            bool matches(const SparseMatrix& matrix) const
            {
                if (rows != matrix.rows() || cols != matrix.cols() || nonzeros != matrix.nonZeros()) {
                    return false;
                }
                return std::equal(outer_indices.begin(), outer_indices.end(), matrix.outerIndexPtr())
                    && std::equal(inner_indices.begin(), inner_indices.end(), matrix.innerIndexPtr());
            }

            void assign(const SparseMatrix& matrix)
            {
                rows = matrix.rows();
                cols = matrix.cols();
                nonzeros = matrix.nonZeros();
                outer_indices.assign(matrix.outerIndexPtr(), matrix.outerIndexPtr() + matrix.outerSize() + 1);
                inner_indices.assign(matrix.innerIndexPtr(), matrix.innerIndexPtr() + matrix.nonZeros());
            }
        };
    }

    struct LinearSolveContext::Impl
    {
#ifdef SYMX_ENABLE_MKL
        Eigen::PardisoLLT<SparseMatrix> solver;
#else
        Eigen::SimplicialLLT<SparseMatrix> solver;
#endif
        bool analyze_pattern_done = false;
        SparsityPattern analyzed_pattern;
    };

    LinearSolveContext::LinearSolveContext() : impl(std::make_unique<Impl>()) {}
    LinearSolveContext::~LinearSolveContext() = default;

    void LinearSolveContext::reset()
    {
        this->impl = std::make_unique<Impl>();
    }

    bool LinearSolveContext::solve_direct_llt(
        const ElementHessians::spBSM& hess,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& rhs,
        const int n_threads)
    {
        const int n = (int)rhs.size();
        std::vector<Eigen::Triplet<double>> triplets;
        hess->to_triplets(triplets);

        SparseMatrix sparse_hess(n, n);
        sparse_hess.setFromTriplets(triplets.begin(), triplets.end());
        sparse_hess.makeCompressed();

#ifdef SYMX_ENABLE_MKL
        if (n_threads > 0) {
            mkl_set_num_threads_local(n_threads);
        }
#endif

        const bool pattern_matches = this->impl->analyze_pattern_done
            && this->impl->analyzed_pattern.matches(sparse_hess);
        if (!pattern_matches) {
            this->impl->analyze_pattern_done = false;
            this->impl->solver.analyzePattern(sparse_hess);
            if (this->impl->solver.info() != Eigen::Success) {
                return false;
            }
            this->impl->analyze_pattern_done = true;
            this->impl->analyzed_pattern.assign(sparse_hess);
        }
        this->impl->solver.factorize(sparse_hess);

        if (this->impl->solver.info() != Eigen::Success) {
            return false;
        }

        x = this->impl->solver.solve(rhs);
        return this->impl->solver.info() == Eigen::Success && x.allFinite();
    }

    bool solve_linear_system_direct_llt(
        const ElementHessians::spBSM& hess,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& rhs)
    {
        LinearSolveContext context;
        return context.solve_direct_llt(hess, x, rhs, 0);
    }

    bsm::PCGInfo solve_linear_system_bdpcg(
        const ElementHessians::spBSM& hess,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& rhs,
        const LinearSolveSettings& settings,
        int n_threads,
        bsm::PCGContext& pcg_context,
        double residual_norm)
    {
        const int n = (int)rhs.size();

        // Forcing sequence: adaptively tighten CG tolerance based on Newton residual
        const double forcing_tol = std::min(1e-2, residual_norm * std::min(0.5, std::sqrt(residual_norm)));
        const double abs_tol = std::max(forcing_tol, settings.cg_abs_tolerance);

        // Prepare block-diagonal preconditioning
        hess->set_preconditioner(bsm::Preconditioner::BlockDiagonal);
        hess->prepare_preconditioning(n_threads);

        // Zero initial guess
        x.resize(n);
        x.setZero();

        return bsm::solve_pcg(
            *hess,
            x.data(),
            rhs.data(),
            n,
            abs_tol,
            settings.cg_rel_tolerance,
            settings.cg_max_iterations,
            n_threads,
            settings.cg_stop_on_indefiniteness,
            pcg_context);
    }
}
