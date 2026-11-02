#include "solver_utils.h"

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <limits>
#include <vector>
#ifdef SYMX_ENABLE_MKL
#include <Eigen/PardisoSupport>
#include <mkl.h>
#endif

namespace symx
{
    namespace
    {
        using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
        using StorageIndex = SparseMatrix::StorageIndex;

        struct SparsityPattern
        {
            int rows = -1;
            int cols = -1;
            std::vector<uint64_t> block_row_offsets;
            std::vector<int> block_column_indices;

            bool matches(const ElementHessians::spBSM& matrix) const
            {
                return rows == matrix->get_n_rows() && cols == matrix->get_n_cols()
                    && block_row_offsets == matrix->get_block_row_offsets()
                    && block_column_indices == matrix->get_block_column_indices();
            }

            void assign(const ElementHessians::spBSM& matrix)
            {
                rows = matrix->get_n_rows();
                cols = matrix->get_n_cols();
                block_row_offsets = matrix->get_block_row_offsets();
                block_column_indices = matrix->get_block_column_indices();
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
        SparseMatrix sparse_hess;
        SparsityPattern pattern;
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
        constexpr int block_size = ElementHessians::BLOCK_SIZE;
        constexpr int block_entries = block_size * block_size;
        // The Newton Hessian consists of complete square blocks.
        if (rhs.size() != hess->get_n_rows() || rhs.size() != hess->get_n_cols()
            || rhs.size() % block_size != 0) {
            return false;
        }
        const int n = (int)rhs.size();
        const auto& rows = hess->get_block_row_offsets();
        const auto& cols = hess->get_block_column_indices();
        const auto* values = hess->get_block_values();
        if (rows.back() > std::numeric_limits<StorageIndex>::max() / block_entries) {
            return false;
        }
        const StorageIndex nonzeros = static_cast<StorageIndex>(rows.back() * block_entries);
        SparseMatrix& sparse_hess = this->impl->sparse_hess;
        if (!this->impl->pattern.matches(hess)) {
            this->impl->analyze_pattern_done = false;
            sparse_hess.resize(n, n);
            sparse_hess.reserve(nonzeros);
            // BSM block columns are sorted. Expand directly into compressed rows,
            // retaining explicit zeros so numeric changes cannot alter the pattern.
            for (int block_row = 0; block_row < n / block_size; ++block_row) {
                for (int row = 0; row < block_size; ++row) {
                    const int global_row = block_size * block_row + row;
                    sparse_hess.startVec(global_row);
                    for (uint64_t block = rows[block_row]; block < rows[block_row + 1]; ++block) {
                        for (int col = 0; col < block_size; ++col) {
                            sparse_hess.insertBack(global_row, cols[block] + col) =
                                static_cast<double>(values[block_entries * block + block_size * col + row]);
                        }
                    }
                }
            }
            sparse_hess.finalize();
            this->impl->pattern.assign(hess);
        }
        else {
            double* destination = sparse_hess.valuePtr();
            const StorageIndex* outer = sparse_hess.outerIndexPtr();
            const int copy_threads = n_threads > 0 ? n_threads : 1;
            #pragma omp parallel for schedule(static) num_threads(copy_threads) if(copy_threads > 1 && nonzeros >= 32768)
            for (int block_row = 0; block_row < n / block_size; ++block_row) {
                // Each worker owns complete scalar rows; no shared writes.
                for (int row = 0; row < block_size; ++row) {
                    StorageIndex offset = outer[block_size * block_row + row];
                    for (uint64_t block = rows[block_row]; block < rows[block_row + 1]; ++block) {
                        for (int col = 0; col < block_size; ++col) {
                            destination[offset++] =
                                static_cast<double>(values[block_entries * block + block_size * col + row]);
                        }
                    }
                }
            }
        }

#ifdef SYMX_ENABLE_MKL
        if (n_threads > 0) {
            mkl_set_num_threads_local(n_threads);
        }
#endif

        if (!this->impl->analyze_pattern_done) {
            this->impl->solver.analyzePattern(sparse_hess);
            if (this->impl->solver.info() != Eigen::Success) {
                return false;
            }
            this->impl->analyze_pattern_done = true;
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
