#include "catch.hpp"
#include "utils.h"

#include <symx>
#include "../symx/src/solver/second_order/SecondOrderCompiledGlobal.h"
#include "../symx/src/solver/first_order/FirstOrderCompiledGlobal.h"
#include "../symx/src/solver/adjoint/AdjointNewton.h"

#include <Eigen/Sparse>

using namespace symx;

namespace
{
Eigen::MatrixXd dense_hessian(bsm::BlockedSparseMatrix<3, 3, ElementHessians::HESS_STORAGE_FLOAT>& matrix, int n)
{
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);
    std::vector<Eigen::Triplet<double>> entries;
    matrix.to_triplets(entries);
    for (const auto& entry : entries) { result(entry.row(), entry.col()) = entry.value(); }
    return result;
}

struct Result {
    double energy;
    Eigen::VectorXd gradient;
    Eigen::MatrixXd hessian;
};

Result evaluate(spGlobalPotential potential, int n_dofs)
{
    auto context = Context::create();
    context->n_threads = 1;
    context->output->set_enabled(false);
    SecondOrderCompiledGlobal compiled(potential, context);
    Result result;
    auto elements = compiled.evaluate_P__dP_du__local_d2P_du2(result.energy, result.gradient);
    result.hessian = dense_hessian(*elements->assemble_global(1, n_dofs), n_dofs);
    return result;
}

void require_same(const Result& a, const Result& b, double tol = 1e-9)
{
    REQUIRE(approx(a.energy, b.energy, tol));
    REQUIRE(a.gradient.isApprox(b.gradient, tol));
    REQUIRE(a.hessian.isApprox(b.hessian, tol));
}

Scalar density(const Matrix& F)
{
    Scalar energy = 0.5 * F(0, 0).powN(2);
    for (int a = 0; a < F.rows(); a++) {
        for (int b = 0; b < F.cols(); b++) {
            if (a == 0 && b == 0) { continue; }
            energy = energy + 0.5 * F(a, b).powN(2);
        }
    }
    return energy + 0.17 * F(0, 0) * F(1, F.cols() - 1) + 0.11 * sin(F(2, 0));
}

enum class Map { Translated, Affine, Nonlinear };
}

TEST_CASE("Factorized FEM selects direct translation and exact mapped kernels", "[factorized]")
{
    std::vector<std::array<int32_t, 3>> connectivity = {{0, 1, 2}};
    std::vector<Eigen::Vector3d> step = {{0.12, -0.03, 0.07}, {-0.08, 0.04, 0.02}, {0.03, 0.09, -0.05}};
    std::vector<Eigen::Vector3d> previous = {{0.0, 0.1, 0.0}, {1.0, 0.0, 0.1}, {0.1, 1.0, -0.1}};
    const int n_dofs = 9;

    auto make = [&](Map map, int n_features, bool factorized, const std::string& name = "same_fem_name") {
        auto G = GlobalPotential::create();
        auto build = [&](MappedWorkspace<double>& mws, Element& element) {
            auto y = mws.make_vectors(step, element);
            auto x0 = mws.make_vectors(previous, element);
            std::vector<Vector> x;
            for (int l = 0; l < 3; l++) {
                Vector mapped = x0[l] + y[l];
                if (map != Map::Translated) {
                    mapped = Vector({
                        x0[l][0] + 1.2 * y[l][0] + 0.1 * y[l][1],
                        x0[l][1] + 0.8 * y[l][1],
                        x0[l][2] + 1.1 * y[l][2]
                    });
                    if (map == Map::Nonlinear) {
                        mapped[0] = mapped[0] + 0.25 * y[l][0].powN(2);
                    }
                }
                x.push_back(mapped);
            }
            Matrix X(collect_scalars(x), {3, 3});
            std::vector<Scalar> coefficients;
            for (int l = 0; l < 3; l++) {
                for (int b = 0; b < n_features; b++) {
                    coefficients.push_back(y[0][0].make_constant(0.2 + 0.1 * l - 0.03 * b));
                }
            }
            return std::make_pair(X, Matrix(coefficients, {3, n_features}));
        };
        if (factorized) {
            G->add_potential_fem(name, connectivity,
                [&](MappedWorkspace<double>& mws, Element& e) {
                    auto [x, B] = build(mws, e);
                    return FemEnergySplit{x, B, density};
                });
        }
        else {
            G->add_potential(name, connectivity,
                [&](MappedWorkspace<double>& mws, Element& e) {
                    auto [x, B] = build(mws, e);
                    return density(x.transpose() * B);
                });
        }
        G->add_dof(step);
        return G;
    };

    for (Map map : {Map::Translated, Map::Affine, Map::Nonlinear}) {
        for (int features : {2, 5}) {
            auto generic = make(map, features, false);
            auto fem = make(map, features, true);
            // These compilations intentionally reuse the logical name and directory.
            require_same(evaluate(generic, n_dofs), evaluate(fem, n_dofs));

            DeferredParallelTasks tasks;
            SecondOrderCompiledPotential kernel(*fem->get_potentials()[0], fem->get_dof_maps(),
                get_codegen_dir(), tasks);
            const auto expected = map == Map::Translated
                ? SecondOrderCompiledPotential::FemKernel::Direct
                : SecondOrderCompiledPotential::FemKernel::Mapped;
            REQUIRE(kernel.get_fem_kernel() == expected);
            tasks.run(1);
            DeferredParallelTasks warm_tasks;
            SecondOrderCompiledPotential warm(*fem->get_potentials()[0], fem->get_dof_maps(),
                get_codegen_dir(), warm_tasks);
            REQUIRE(warm.was_cached());
            warm_tasks.run(1);
        }
    }

    // The separate first-order optimizer still differentiates the full energy. Exercise
    // generic and FEM definitions with identical logical names in the same process.
    auto evaluate_first_order = [](spGlobalPotential potential) {
        auto context = Context::create();
        context->n_threads = 1;
        context->output->set_enabled(false);
        FirstOrderCompiledGlobal compiled(potential, context);
        double energy = 0.0;
        Eigen::VectorXd gradient;
        compiled.evaluate_P__dP_du(energy, gradient);
        return std::make_pair(energy, gradient);
    };
    const auto first_generic = evaluate_first_order(make(Map::Translated, 2, false));
    const auto first_fem = evaluate_first_order(make(Map::Translated, 2, true));
    REQUIRE(approx(first_generic.first, first_fem.first, 1e-9));
    REQUIRE(first_generic.second.isApprox(first_fem.second, 1e-9));

    // Distinct readable names permit simultaneous loaded kernels.
    auto affine = make(Map::Affine, 2, true, "affine_fem_name");
    auto nonlinear = make(Map::Nonlinear, 2, true, "nonlinear_fem_name");
    DeferredParallelTasks first_tasks;
    SecondOrderCompiledPotential first(*affine->get_potentials()[0], affine->get_dof_maps(),
        get_codegen_dir(), first_tasks);
    first_tasks.run(1);
    DeferredParallelTasks second_tasks;
    SecondOrderCompiledPotential second(*nonlinear->get_potentials()[0], nonlinear->get_dof_maps(),
        get_codegen_dir(), second_tasks);
    second_tasks.run(1);
    REQUIRE(first.get_fem_kernel() == SecondOrderCompiledPotential::FemKernel::Mapped);
    REQUIRE(second.get_fem_kernel() == SecondOrderCompiledPotential::FemKernel::Mapped);
}

TEST_CASE("Factorized FEM wrappers match explicit quadrature definitions", "[factorized]")
{
    std::vector<std::array<int32_t, 2>> connectivity = {{0, 1}, {1, 2}};
    std::vector<Eigen::Vector3d> x = {{0.1, 0.0, 0.0}, {0.9, 0.2, 0.1}, {1.8, -0.1, 0.3}};
    std::vector<std::array<double, 2>> rule = {{0.4, -0.2}, {0.6, 0.5}};
    std::vector<std::array<double, 2>> indexed;
    for (size_t e = 0; e < connectivity.size(); e++) {
        indexed.insert(indexed.end(), rule.begin(), rule.end());
    }
    std::vector<double> factors(indexed.size(), 1.0);
    std::vector<int64_t> offsets = {0, 1, 4};
    EigenMatrixRM<double> eigen_rule(2, 2);
    for (int q = 0; q < 2; q++) for (int i = 0; i < 2; i++) {
        eigen_rule(q, i) = rule[q][i];
    }

    auto make = [&](int mode, bool factorized) {
        auto G = GlobalPotential::create();
        auto definition = [&](MappedWorkspace<double>& mws, Element& e) {
            auto xv = mws.make_vectors(x, e);
            auto split = [&](const Vector& ip) {
                Scalar scale = mode == 2 ? mws.make_quadrature_scalar(factors) : ip[0].make_constant(1.0);
                Matrix nodes(collect_scalars(xv), {2, 3});
                Matrix B({
                    -ip[1], -ip[1].powN(2),
                     ip[1],  ip[1].powN(2)
                }, {2, 2});
                Scalar weight = ip[0] * scale;
                auto material = [=](const Matrix& F) {
                    // A conditional material checks substitution and compact derivatives together.
                    Scalar basic = density(F);
                    return weight * branch(F(0, 0) + 0.1, basic, 0.3 * basic);
                };
                return std::make_pair(FemEnergySplit{nodes, B, material},
                                      xv[0][0] + 1.0);
            };
            if (factorized) {
                if (mode == 0) { return mws.add_for_each(rule, split); }
                if (mode == 1) {
                    return mws.add_for_each(static_cast<int32_t>(rule.size()), [&]() {
                        return split(mws.make_quadrature_vector(indexed));
                    });
                }
                if (mode == 2) { return mws.add_for_each(eigen_rule, split); }
                if (mode == 3) {
                    return mws.add_for_each(offsets, [&]() {
                        return split(mws.make_quadrature_vector(indexed));
                    });
                }
                return mws.add_for_each(
                    [&]() { return offsets.data(); },
                    [&]() { return static_cast<int64_t>(offsets.size()); },
                    [&]() { return split(mws.make_quadrature_vector(indexed)); });
            }
            // Generic symbolic energies are the reference, using explicit declarations.
            if (mode == 1) { mws.make_summation(static_cast<int32_t>(rule.size())); }
            if (mode >= 3) { mws.make_summation(offsets); }
            return split((mode == 1 || mode >= 3)
                ? mws.make_quadrature_vector(indexed) : mws.make_summation_vector(rule));
        };
        if (factorized) {
            G->add_potential_fem("qp_fem_same_name", connectivity, definition);
        }
        else {
            G->add_potential("qp_fem_same_name", connectivity,
                [&](MappedWorkspace<double>& mws, Element& e) {
                    auto [split, condition] = definition(mws, e);
                    return std::make_pair(split.material(split.x.transpose() * split.B), condition);
                });
        }
        G->add_dof(x);
        return G;
    };

    const Result fixed = evaluate(make(0, false), 9);
    for (int mode : {0, 1, 2}) {
        require_same(fixed, evaluate(make(mode, true), 9));
    }
    const Result variable = evaluate(make(3, false), 9);
    require_same(variable, evaluate(make(3, true), 9));
    require_same(variable, evaluate(make(4, true), 9));
}

TEST_CASE("Factorized volume quadrature matches the rubber-style reference gradient", "[factorized]")
{
    for (FEM_Element element_type : {FEM_Element::Hex8, FEM_Element::Tet10}) {
        auto mesh = generate_cuboid_mesh(element_type, Eigen::Vector3d(1, 1, 1), {1, 1, 1});
        std::vector<Eigen::Vector3d> x = mesh.vertices;
        std::vector<Eigen::Vector3d> rest = mesh.vertices;
        x[0].x() += 0.04;
        x[4].z() -= 0.03;
        const auto rule = get_integration_rule(element_type);
        const std::string name = "volume_" + get_name(element_type);
        auto material = [](const Matrix& F) {
            return 0.5 * F.frobenius_norm_sq() + 0.2 * F.det().powN(2);
        };

        auto generic = GlobalPotential::create();
        generic->add_potential(name, mesh.connectivity, mesh.connectivity_stride,
            [&](MappedWorkspace<double>& mws, Element& e) {
                auto current = mws.make_vectors(x, e);
                auto reference = mws.make_vectors(rest, e);
                return mws.add_for_each(rule, [&](Vector& ip) {
                    Vector xi({ip[1], ip[2], ip[3]});
                    Matrix Jx = fem_jacobian(element_type, current, xi);
                    Matrix JX = fem_jacobian(element_type, reference, xi);
                    return ip[0] * JX.det() * material(Jx * JX.inv());
                });
            });
        generic->add_dof(x);

        auto factorized = GlobalPotential::create();
        factorized->add_potential_fem(name, mesh.connectivity, mesh.connectivity_stride,
            [&](MappedWorkspace<double>& mws, Element& e) {
                auto current = mws.make_vectors(x, e);
                auto reference = mws.make_vectors(rest, e);
                return mws.add_for_each(rule, [&](Vector& ip) {
                    Vector xi({ip[1], ip[2], ip[3]});
                    Matrix JX = fem_jacobian(element_type, reference, xi);
                    Matrix nodes(collect_scalars(current), {static_cast<int32_t>(current.size()), 3});
                    Matrix B = fem_deformation_gradient_B(element_type, xi, JX);
                    return FemEnergySplit{nodes, B, [=](const Matrix& small_F) {
                        return ip[0] * JX.det() * material(small_F);
                    }};
                });
            });
        factorized->add_dof(x);
        const int n_dofs = static_cast<int>(x.size()) * 3;
        INFO(get_name(element_type));
        require_same(evaluate(generic, n_dofs), evaluate(factorized, n_dofs), 1e-8);
    }
}

TEST_CASE("Standard FEM B reproduces deformation gradients", "[factorized]")
{
    for (FEM_Element type : {FEM_Element::Tet4, FEM_Element::Tet10, FEM_Element::Hex8, FEM_Element::Hex27}) {
        const auto mesh = generate_cuboid_mesh(type, Eigen::Vector3d(1, 1, 1), {1, 1, 1});
        const auto point = get_integration_rule(type).front();
        Workspace ws;
        Vector xi = ws.make_vector(3);
        xi.set_values(std::array<double, 3>{point[1], point[2], point[3]});
        auto X = ws.make_vectors(3, mesh.connectivity_stride);
        auto x = ws.make_vectors(3, mesh.connectivity_stride);
        for (int l = 0; l < mesh.connectivity_stride; l++) {
            const Eigen::Vector3d& rest = mesh.vertices[mesh.connectivity[l]];
            for (int a = 0; a < 3; a++) {
                X[l][a].set_value(rest[a]);
                x[l][a].set_value(rest[a] + 0.02 * (l + 1) * (a + 1));
            }
        }
        Matrix JX = fem_jacobian(type, X, xi);
        Matrix Jx = fem_jacobian(type, x, xi);
        Matrix D = fem_shape_gradients(type, xi);
        Matrix B = fem_deformation_gradient_B(type, xi, JX);
        Matrix X_nodes(collect_scalars(X), {mesh.connectivity_stride, 3});
        Matrix x_nodes(collect_scalars(x), {mesh.connectivity_stride, 3});
        INFO(get_name(type));
        REQUIRE(B.rows() == mesh.connectivity_stride);
        REQUIRE(B.cols() == 3);
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                REQUIRE(approx((X_nodes.transpose() * D)(a, b).eval(), JX(a, b).eval(), 1e-10));
                REQUIRE(approx((x_nodes.transpose() * B)(a, b).eval(), (Jx * JX.inv())(a, b).eval(), 1e-10));
            }
        }
    }
}

TEST_CASE("Factorized FEM supports rectangular B for lines and membranes", "[factorized]")
{
    // Rest configurations lie on the x axis (r = 1) or in the xy plane (r = 2), so the first
    // r rest coordinates are the reference coordinates. B is n_nodes x r and F = x^T B is 3 x r.
    for (int r : {1, 2}) {
        std::vector<Eigen::Vector3d> rest;
        std::vector<int32_t> connectivity;
        if (r == 1) {
            rest = {{0.0, 0.0, 0.0}, {0.7, 0.0, 0.0}, {1.5, 0.0, 0.0}};
            connectivity = {0, 1, 1, 2};
        }
        else {
            rest = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.1, 0.9, 0.0}, {1.2, 1.0, 0.0}};
            connectivity = {0, 1, 2, 1, 3, 2};
        }
        const int32_t stride = r + 1;
        std::vector<Eigen::Vector3d> x = rest;
        for (size_t i = 0; i < x.size(); i++) {
            // Stretch in-plane and lift out of the rest plane.
            x[i] += Eigen::Vector3d(0.05 * i, -0.03 * i * i, 0.04 + 0.02 * i);
        }

        auto split = [&](MappedWorkspace<double>& mws, Element& e) {
            auto X = mws.make_vectors(rest, e);
            auto xv = mws.make_vectors(x, e);
            Matrix nodes(collect_scalars(xv), {stride, 3});
            std::vector<Scalar> b;
            Scalar measure = X[1][0] - X[0][0];  // Segment length for r = 1
            if (r == 1) {
                b = {-1.0 / measure, 1.0 / measure};
            }
            else {
                Matrix Dm({X[1][0] - X[0][0], X[2][0] - X[0][0],
                           X[1][1] - X[0][1], X[2][1] - X[0][1]}, {2, 2});
                Matrix Dm_inv = Dm.inv();
                b = {-Dm_inv(0, 0) - Dm_inv(1, 0), -Dm_inv(0, 1) - Dm_inv(1, 1),
                      Dm_inv(0, 0), Dm_inv(0, 1),
                      Dm_inv(1, 0), Dm_inv(1, 1)};
                measure = 0.5 * Dm.det();  // Triangle area
            }
            auto material = [=](const Matrix& F) {
                Matrix C = F.transpose() * F;
                if (r == 1) { return measure * (sqrt(C(0, 0)) - 1.0).powN(2); }
                Matrix E = (C - Matrix::identity(2, F(0, 0))) * 0.5;
                return measure * (E.frobenius_norm_sq() + 0.3 * E.trace().powN(2));
            };
            return FemEnergySplit{nodes, Matrix(b, {stride, r}), material};
        };

        auto make = [&](bool factorized) {
            auto G = GlobalPotential::create();
            const std::string name = "rectangular_B_r" + std::to_string(r);
            if (factorized) {
                G->add_potential_fem(name, connectivity, stride, split);
            }
            else {
                G->add_potential(name, connectivity, stride,
                    [&](MappedWorkspace<double>& mws, Element& e) {
                        const FemEnergySplit s = split(mws, e);
                        return s.material(s.x.transpose() * s.B);
                    });
            }
            G->add_dof(x);
            return G;
        };

        const int n_dofs = 3 * static_cast<int>(x.size());
        INFO("r = " << r);
        require_same(evaluate(make(false), n_dofs), evaluate(make(true), n_dofs));

        // At rest C = I, so energy and gradient vanish only if B maps x to the correct F.
        x = rest;
        const Result at_rest = evaluate(make(true), n_dofs);
        REQUIRE(approx(at_rest.energy, 0.0, 1e-12));
        REQUIRE(at_rest.gradient.norm() < 1e-12);
    }
}

TEST_CASE("Factorized material ignores unreachable conditional branches", "[factorized]")
{
    std::vector<std::array<int32_t, 1>> connectivity = {{0}};
    std::vector<Eigen::Vector3d> x = {{0.35, 0.2, -0.1}};
    auto material = [](const Matrix& F) {
        Scalar value = 0.0 * F(0, 0);
        for (int i = 0; i < 5; i++) {
            Scalar shifted = F(0, 0) - (0.1 + 0.1 * i);
            value = value + branch(shifted, shifted.powN(2), 0.3 * shifted.powN(2));
        }
        return value + 0.2 * F(1, 0).powN(2);
    };
    auto make = [&](bool factorized) {
        auto G = GlobalPotential::create();
        if (factorized) {
            G->add_potential_fem("five_material_branches", connectivity,
                [&](MappedWorkspace<double>& mws, Element& e) {
                    auto node = mws.make_vectors(x, e);
                    Matrix positions(collect_scalars(node), {1, 3});
                    Matrix B({node[0][0].make_constant(1.0)}, {1, 1});
                    return FemEnergySplit{positions, B, material};
                });
        }
        else {
            G->add_potential("five_material_branches", connectivity,
                [&](MappedWorkspace<double>& mws, Element& e) {
                    auto node = mws.make_vectors(x, e);
                    Matrix positions(collect_scalars(node), {1, 3});
                    Matrix B({node[0][0].make_constant(1.0)}, {1, 1});
                    return material(positions.transpose() * B);
                });
        }
        G->add_dof(x);
        return G;
    };
    require_same(evaluate(make(false), 3), evaluate(make(true), 3));

#if !defined(_WIN32)
    const auto too_many = run_expected_failure([] {
        Workspace ws;
        Scalar x = ws.make_scalar();
        Scalar value = x;
        for (int i = 0; i < 9; i++) {
            value = value + branch(x - (0.1 + i), x.powN(2), 0.25 * x);
        }
        Sequence sequence({value});
    });
    INFO(too_many.diagnostic);
    REQUIRE(too_many.failed);
    REQUIRE(too_many.diagnostic.find("Max allowed is 8") != std::string::npos);
#endif
}

TEST_CASE("Factorized FEM adjoint retains ordinary material, geometry, and map parameters", "[factorized][Adjoint]")
{
    struct Data {
        std::vector<Eigen::Vector3d> x = {{0, 0, 0}, {1, 0, 0}};
        std::vector<Eigen::Vector3d> rest = x;
        std::vector<double> weights = {0.3, 0.7};
        std::vector<std::array<double, 2>> targets = {{0.4, 0.6}, {0.6, 0.8}};
        double stiffness = 100.0;
        double geometry_scale = 1.1;
        double map_scale = 0.02;
        double bc_stiffness = 1000.0;
    } data;
    std::vector<std::array<int32_t, 2>> spring = {{0, 1}};
    std::vector<std::array<int32_t, 1>> boundary = {{0}, {1}};

    auto G = GlobalPotential::create();
    G->add_potential_fem("fem_adjoint_spring", spring,
        [&](MappedWorkspace<double>& mws, Element& e) {
            auto y = mws.make_vectors(data.x, e);
            Scalar scale = mws.make_scalar(data.map_scale);
            Scalar geometry = mws.make_scalar(data.geometry_scale);
            Scalar k = mws.make_scalar(data.stiffness);
            mws.make_summation(2);
            Scalar w = mws.make_quadrature_scalar(data.weights);
            std::vector<Vector> mapped = y;
            for (int l = 0; l < 2; l++) {
                mapped[l][0] = mapped[l][0] + scale * y[l][0].powN(2);
            }
            Matrix x(collect_scalars(mapped), {2, 3});
            Matrix B({-geometry, geometry}, {2, 1});
            return FemEnergySplit{x, B, [=](const Matrix& F) {
                return 0.5 * k * w * (F(0, 0).powN(2) + F(1, 0).powN(2) + F(2, 0).powN(2));
            }};
        });
    G->add_potential("fem_adjoint_boundary", boundary,
        [&](MappedWorkspace<double>& mws, Element& e) {
            Vector x = mws.make_vector(data.x, e[0]);
            Vector rest = mws.make_vector(data.rest, e[0]);
            Scalar k = mws.make_scalar(data.bc_stiffness);
            return 0.5 * k * (x - rest).squared_norm();
        });
    G->add_dof(data.x);
    auto context = Context::create();
    context->n_threads = 1;
    context->output->set_enabled(false);
    auto newton = NewtonsMethod::create(G, context);
    newton->settings.residual_tolerance_abs = 1e-6;

    auto L = GlobalPotential::create();
    L->add_potential("fem_adjoint_loss", spring,
        [&](MappedWorkspace<double>& mws, Element& e) {
            auto x = mws.make_vectors(data.x, e);
            return mws.add_for_each(2, [&]() {
                Vector target = mws.make_quadrature_vector(data.targets);
                return target[0] * (x[1][0] - x[0][0] - target[1]).powN(2);
            });
        });
    L->add_dof(data.stiffness);
    L->add_dof(data.geometry_scale);
    L->add_dof(data.map_scale);

    LinearSolveSettings linear;
    linear.linear_solver = LinearSolver::DirectLLT;
    auto adjoint = AdjointNewton::create(newton, L, linear);
    const auto numerical = adjoint->evaluate_dL_dp_finite_differences(1e-3);
    const auto analytic = adjoint->run_forward_and_evaluate_dL_dp();
    INFO("finite difference status: " << to_string(numerical.status));
    INFO("analytic status: " << to_string(analytic.status));
    REQUIRE(numerical.success);
    REQUIRE(analytic.success);
    REQUIRE(analytic.gradient.size() == 3);
    REQUIRE(data.weights.size() == 2);
    REQUIRE(data.weights[0] != data.weights[1]);
    REQUIRE((data.targets[0][0] != data.targets[1][0] || data.targets[0][1] != data.targets[1][1]));
    for (int i = 0; i < 3; i++) {
        INFO("ordinary parameter " << i);
        REQUIRE(std::abs(analytic.gradient[i]) > 1e-7);
        REQUIRE(approx(analytic.gradient[i], numerical.gradient[i], 1e-3));
    }
}

TEST_CASE("Factorized FEM adjoint differentiates through indexed quadrature data", "[factorized][Adjoint]")
{
    struct Data {
        std::vector<Eigen::Vector3d> x = {{0, 0, 0}, {0.6, 0.02, 0}};
        std::vector<Eigen::Vector3d> target = x;
        std::vector<double> weights = {0.4, 1.1};
        std::vector<double> quadrature_targets = {0.2, 0.9};
        double stiffness = 2.0;
        double boundary_stiffness = 40.0;
    } data;
    const std::vector<std::array<int32_t, 2>> spring = {{0, 1}};
    const std::vector<std::array<int32_t, 1>> nodes = {{0}, {1}};

    auto G = GlobalPotential::create();
    G->add_potential_fem("fem_indexed_qp_adjoint", spring,
        [&](MappedWorkspace<double>& mws, Element& e) {
            auto x = mws.make_vectors(data.x, e);
            mws.make_summation(2);
            Scalar weight = mws.make_quadrature_scalar(data.weights);
            Scalar target = mws.make_quadrature_scalar(data.quadrature_targets);
            Scalar stiffness = mws.make_scalar(data.stiffness);
            Matrix positions(collect_scalars(x), {2, 3});
            Matrix B({x[0][0].make_constant(-1.0), x[0][0].make_constant(1.0)}, {2, 1});
            return FemEnergySplit{positions, B, [=](const Matrix& F) {
                return 0.5 * stiffness * weight * (F(0, 0) - target).powN(2);
            }};
        });
    G->add_potential("fem_indexed_qp_pin", nodes,
        [&](MappedWorkspace<double>& mws, Element& e) {
            Vector x = mws.make_vector(data.x, e[0]);
            Vector target = mws.make_vector(data.target, e[0]);
            return 0.5 * data.boundary_stiffness * (x - target).squared_norm();
        });
    G->add_dof(data.x);

    auto context = Context::create();
    context->n_threads = 1;
    context->output->set_enabled(false);
    auto newton = NewtonsMethod::create(G, context);
    newton->settings.residual_tolerance_abs = 1e-10;
    auto L = GlobalPotential::create();
    L->add_potential("fem_indexed_qp_loss", spring,
        [&](MappedWorkspace<double>& mws, Element& e) {
            auto x = mws.make_vectors(data.x, e);
            return (x[1][0] - x[0][0] - 0.8).powN(2);
        });
    L->add_dof(data.stiffness);
    LinearSolveSettings linear;
    linear.linear_solver = LinearSolver::DirectLLT;
    auto adjoint = AdjointNewton::create(newton, L, linear);

    const auto analytic = adjoint->run_forward_and_evaluate_dL_dp();
    const auto numerical = adjoint->evaluate_dL_dp_finite_differences(1e-5);
    INFO("finite difference status: " << to_string(numerical.status));
    INFO("analytic status: " << to_string(analytic.status));
    REQUIRE(analytic.success);
    REQUIRE(numerical.success);
    REQUIRE(data.weights[0] != data.weights[1]);
    REQUIRE(data.quadrature_targets[0] != data.quadrature_targets[1]);
    REQUIRE(analytic.gradient.size() == 1);
    REQUIRE(std::abs(analytic.gradient[0]) > 1e-5);
    REQUIRE(approx(analytic.gradient[0], numerical.gradient[0], 2e-4));
}

TEST_CASE("Factorized FEM adjoint differentiates both material branch selections", "[factorized][Adjoint]")
{
    struct Data {
        std::vector<Eigen::Vector3d> x = {{0, 0, 0}, {0.6, 0.03, 0}};
        std::vector<Eigen::Vector3d> target = x;
        double stiffness = 2.0;
        double boundary_stiffness = 40.0;
    };
    const std::vector<std::array<int32_t, 2>> spring = {{0, 1}};
    const std::vector<std::array<int32_t, 1>> nodes = {{0}, {1}};

    // Both branch conditions stay far from zero as stiffness is perturbed for finite differences.
    for (double threshold : {0.1, 1.0}) {
        Data data;
        const std::string name = threshold < 0.5 ? "fem_material_branch_positive" : "fem_material_branch_negative";
        auto G = GlobalPotential::create();
        G->add_potential_fem(name, spring,
            [&](MappedWorkspace<double>& mws, Element& e) {
                auto x = mws.make_vectors(data.x, e);
                Scalar k = mws.make_scalar(data.stiffness);
                Matrix positions(collect_scalars(x), {2, 3});
                Matrix B({-1.0 * x[0][0].make_constant(1.0), x[0][0].make_constant(1.0)}, {2, 1});
                return FemEnergySplit{positions, B, [=](const Matrix& F) {
                    Scalar f = F(0, 0);
                    Scalar positive = 0.5 * k * (f - 1.2).powN(2) + 0.2 * k;
                    Scalar negative = 0.35 * k * (f + 0.2).powN(2) + 0.15 * k;
                    return branch(f - threshold, positive, negative);
                }};
            });
        G->add_potential("fem_material_branch_pin", nodes,
            [&](MappedWorkspace<double>& mws, Element& e) {
                Vector x = mws.make_vector(data.x, e[0]);
                Vector target = mws.make_vector(data.target, e[0]);
                return 0.5 * data.boundary_stiffness * (x - target).squared_norm();
            });
        G->add_dof(data.x);

        auto context = Context::create();
        context->n_threads = 1;
        context->output->set_enabled(false);
        auto newton = NewtonsMethod::create(G, context);
        newton->settings.residual_tolerance_abs = 1e-10;
        auto L = GlobalPotential::create();
        L->add_potential("fem_material_branch_loss", spring,
            [&](MappedWorkspace<double>& mws, Element& e) {
                auto x = mws.make_vectors(data.x, e);
                return (x[1][0] - x[0][0] - 0.8).powN(2);
            });
        L->add_dof(data.stiffness);
        LinearSolveSettings linear;
        linear.linear_solver = LinearSolver::DirectLLT;
        auto adjoint = AdjointNewton::create(newton, L, linear);

        const auto analytic = adjoint->run_forward_and_evaluate_dL_dp();
        REQUIRE(analytic.success);
        const double branch_distance = data.x[1][0] - data.x[0][0] - threshold;
        if (threshold < 0.5) { REQUIRE(branch_distance > 0.3); }
        else { REQUIRE(branch_distance < -0.3); }
        const auto numerical = adjoint->evaluate_dL_dp_finite_differences(1e-5);
        INFO("threshold " << threshold << "; FD status: " << to_string(numerical.status));
        REQUIRE(numerical.success);
        REQUIRE(analytic.gradient.size() == 1);
        REQUIRE(std::abs(analytic.gradient[0]) > 1e-5);
        REQUIRE(approx(analytic.gradient[0], numerical.gradient[0], 2e-4));
    }
}

TEST_CASE("Factorized FEM adjoint respects potential-level conditions", "[factorized][Adjoint]")
{
    struct Data {
        std::vector<Eigen::Vector3d> x = {{0, 0, 0}, {0.6, 0.02, 0}};
        std::vector<Eigen::Vector3d> target = x;
        double active_stiffness = 2.0;
        double conditional_stiffness = 1.5;
        double enabled = 1.0;
        double boundary_stiffness = 40.0;
    };
    const std::vector<std::array<int32_t, 2>> spring = {{0, 1}};
    const std::vector<std::array<int32_t, 1>> nodes = {{0}, {1}};

    for (double enabled : {1.0, -1.0}) {
        Data data;
        data.enabled = enabled;
        auto G = GlobalPotential::create();
        G->add_potential_fem("fem_condition_always_active", spring,
            [&](MappedWorkspace<double>& mws, Element& e) {
                auto x = mws.make_vectors(data.x, e);
                Scalar k = mws.make_scalar(data.active_stiffness);
                Matrix positions(collect_scalars(x), {2, 3});
                Matrix B({x[0][0].make_constant(-1.0), x[0][0].make_constant(1.0)}, {2, 1});
                return FemEnergySplit{positions, B, [=](const Matrix& F) {
                    return 0.5 * k * (F(0, 0) - 0.9).powN(2);
                }};
            });
        G->add_potential_fem("fem_condition_toggle", spring,
            [&](MappedWorkspace<double>& mws, Element& e) {
                auto x = mws.make_vectors(data.x, e);
                Scalar k = mws.make_scalar(data.conditional_stiffness);
                Scalar condition = mws.make_scalar(data.enabled);
                Matrix positions(collect_scalars(x), {2, 3});
                Matrix B({x[0][0].make_constant(-1.0), x[0][0].make_constant(1.0)}, {2, 1});
                FemEnergySplit split{positions, B, [=](const Matrix& F) {
                    return 0.5 * k * (F(0, 0) + 0.2).powN(2);
                }};
                return std::make_pair(split, condition);
            });
        G->add_potential("fem_condition_pin", nodes,
            [&](MappedWorkspace<double>& mws, Element& e) {
                Vector x = mws.make_vector(data.x, e[0]);
                Vector target = mws.make_vector(data.target, e[0]);
                return 0.5 * data.boundary_stiffness * (x - target).squared_norm();
            });
        G->add_dof(data.x);

        auto context = Context::create();
        context->n_threads = 1;
        context->output->set_enabled(false);
        auto newton = NewtonsMethod::create(G, context);
        newton->settings.residual_tolerance_abs = 1e-10;
        auto L = GlobalPotential::create();
        L->add_potential("fem_condition_loss", spring,
            [&](MappedWorkspace<double>& mws, Element& e) {
                auto x = mws.make_vectors(data.x, e);
                return (x[1][0] - x[0][0] - 0.8).powN(2);
            });
        L->add_dof(data.active_stiffness);
        L->add_dof(data.conditional_stiffness);
        LinearSolveSettings linear;
        linear.linear_solver = LinearSolver::DirectLLT;
        auto adjoint = AdjointNewton::create(newton, L, linear);

        const auto analytic = adjoint->run_forward_and_evaluate_dL_dp();
        const auto numerical = adjoint->evaluate_dL_dp_finite_differences(1e-5);
        INFO("condition gate " << enabled << "; FD status: " << to_string(numerical.status));
        REQUIRE(analytic.success);
        REQUIRE(numerical.success);
        REQUIRE(analytic.gradient.size() == 2);
        REQUIRE(std::abs(analytic.gradient[0]) > 1e-5);
        if (enabled > 0.0) { REQUIRE(std::abs(analytic.gradient[1]) > 1e-5); }
        else { REQUIRE(approx(analytic.gradient[1], 0.0, 1e-9)); }
        for (int i = 0; i < 2; i++) {
            INFO("parameter " << i << " with gate " << enabled);
            REQUIRE(approx(analytic.gradient[i], numerical.gradient[i], 2e-4));
        }
    }
}

TEST_CASE("Factorized FEM adjoint composes mapped state, vector geometry, and explicit loss parameters", "[factorized][Adjoint]")
{
    struct Data {
        std::vector<Eigen::Vector3d> x = {{0, 0, 0}, {1.03, 0, 0}, {0, 1.02, 0}, {0, 0, 0.98}};
        std::vector<Eigen::Vector3d> reference = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        std::vector<Eigen::Vector3d> target = {{0, 0, 0}, {1.08, 0, 0}, {0, 1.04, 0}, {0, 0, 0.95}};
        double stiffness = 6.0;
        double map_scale = 0.025;
        double boundary_stiffness = 80.0;
    } data;
    const std::vector<std::array<int32_t, 4>> tet = {{0, 1, 2, 3}};
    const std::vector<std::array<int32_t, 1>> nodes = {{0}, {1}, {2}, {3}};
    const auto rule = get_integration_rule(FEM_Element::Tet4);

    auto G = GlobalPotential::create();
    G->add_potential_fem("fem_mixed_parameter_adjoint", tet,
        [&](MappedWorkspace<double>& mws, Element& e) {
            auto state = mws.make_vectors(data.x, e);
            auto reference = mws.make_vectors(data.reference, e);
            Scalar map_scale = mws.make_scalar(data.map_scale);
            Scalar stiffness = mws.make_scalar(data.stiffness);
            std::vector<Vector> mapped = state;
            for (Vector& node : mapped) {
                node[0] = node[0] + map_scale * node[0].powN(2);
            }
            Vector point = mws.make_summation_vector(rule);
            Vector xi({point[1], point[2], point[3]});
            Matrix JX = fem_jacobian(FEM_Element::Tet4, reference, xi);
            Matrix B = fem_deformation_gradient_B(FEM_Element::Tet4, xi, JX);
            Matrix positions(collect_scalars(mapped), {4, 3});
            return FemEnergySplit{positions, B, [=](const Matrix& F) {
                Scalar strain = 0.0 * F(0, 0);
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        Scalar delta = F(a, b) - F(a, b).make_constant(a == b ? 1.0 : 0.0);
                        strain = strain + delta.powN(2);
                    }
                }
                return 0.5 * stiffness * point[0] * JX.det() * strain;
            }};
        });
    G->add_potential("fem_mixed_parameter_pin", nodes,
        [&](MappedWorkspace<double>& mws, Element& e) {
            Vector x = mws.make_vector(data.x, e[0]);
            Vector target = mws.make_vector(data.target, e[0]);
            return 0.5 * data.boundary_stiffness * (x - target).squared_norm();
        });
    G->add_dof(data.x, "state");

    auto context = Context::create();
    context->n_threads = 1;
    context->output->set_enabled(false);
    auto newton = NewtonsMethod::create(G, context);
    // This mixed nonlinear map is intentionally tested with the normal solver tolerance;
    // driving the already small residual to machine precision can stall Armijo backtracking.
    newton->settings.residual_tolerance_abs = 1e-7;

    auto L = GlobalPotential::create();
    const std::vector<std::array<int32_t, 1>> observed_node = {{1}};
    L->add_potential("fem_mixed_parameter_loss", observed_node,
        [&](MappedWorkspace<double>& mws, Element& e) {
            Vector x = mws.make_vector(data.x, e[0]);
            Vector reference = mws.make_vector(data.reference, e[0]);
            Scalar stiffness = mws.make_scalar(data.stiffness);
            Scalar map_scale = mws.make_scalar(data.map_scale);
            return (x[0] - 1.0).powN(2) + 1.0 * stiffness + 0.02 * map_scale + 0.05 * reference[0];
        });
    L->add_dof(data.stiffness, "material stiffness");
    L->add_dof(data.map_scale, "map scale");
    L->add_dof(data.reference, "reference vertices");

    LinearSolveSettings linear;
    linear.linear_solver = LinearSolver::DirectLLT;
    auto adjoint = AdjointNewton::create(newton, L, linear);
    const auto analytic = adjoint->run_forward_and_evaluate_dL_dp();
    const auto numerical = adjoint->evaluate_dL_dp_finite_differences(1e-5);
    INFO("finite difference status: " << to_string(numerical.status));
    INFO("analytic status: " << to_string(analytic.status));
    REQUIRE(analytic.success);
    REQUIRE(numerical.success);
    REQUIRE(analytic.gradient.size() == 14); // stiffness, map scale, then four 3D reference vertices
    REQUIRE(std::abs(analytic.gradient[0]) > 1e-5);
    REQUIRE(std::abs(analytic.gradient[1]) > 1e-5);
    REQUIRE(std::abs(analytic.gradient[5]) > 1e-5); // explicit loss dependence on reference vertex 1 x
    REQUIRE(analytic.gradient.segment(2, 12).norm() > 1e-5);
    for (int i = 0; i < analytic.gradient.size(); i++) {
        INFO("parameter component " << i);
        REQUIRE(approx(analytic.gradient[i], numerical.gradient[i], 2e-4));
    }
}

#if !defined(_WIN32)
TEST_CASE("Factorized FEM rejects state-dependent coefficients and cross-node maps", "[factorized]")
{
    auto rejected = [](bool invalid_B) {
        return run_expected_failure([=] {
            std::vector<std::array<int32_t, 2>> connectivity = {{0, 1}};
            std::vector<Eigen::Vector3d> y = {{0, 0, 0}, {1, 0, 0}};
            auto G = GlobalPotential::create();
            G->add_potential_fem("invalid_fem", connectivity,
                [&](MappedWorkspace<double>& mws, Element& e) {
                    auto positions = mws.make_vectors(y, e);
                    if (!invalid_B) { positions[0][0] = positions[0][0] + positions[1][0]; }
                    Matrix x(collect_scalars(positions), {2, 3});
                    Matrix B({invalid_B ? positions[0][0] : positions[0][0].make_constant(-1.0),
                              positions[0][0].make_constant(1.0)}, {2, 1});
                    return FemEnergySplit{x, B, density};
                });
            G->add_dof(y);
            DeferredParallelTasks tasks;
            SecondOrderCompiledPotential kernel(*G->get_potentials()[0], G->get_dof_maps(),
                get_codegen_dir(), tasks);
        });
    };
    const auto invalid_B = rejected(true);
    INFO(invalid_B.diagnostic);
    REQUIRE(invalid_B.failed);
    REQUIRE(invalid_B.diagnostic.find("B depends on state DoFs") != std::string::npos);
    const auto cross_node = rejected(false);
    INFO(cross_node.diagnostic);
    REQUIRE(cross_node.failed);
    REQUIRE(cross_node.diagnostic.find("another node's DoFs") != std::string::npos);
}

TEST_CASE("Factorized FEM compiler validates symbolic block dimensions", "[factorized]")
{
	Workspace workspace;
	Scalar seed = workspace.make_scalar();
	Vector P(std::vector<Scalar>(3, seed));
	Matrix A(std::vector<Scalar>(4, seed), {2, 2});
	Matrix B(std::vector<Scalar>(2, seed), {2, 1});
	Compilation compilation;

	const auto direct = run_expected_failure([&] {
		compilation.compile_factorized<double>(seed, P, A, B, "invalid_factorized_dimensions", get_codegen_dir());
	});
	INFO(direct.diagnostic);
	REQUIRE(direct.failed);
	REQUIRE(direct.diagnostic.find("inconsistent symbolic dimensions") != std::string::npos);

	Matrix valid_A(std::vector<Scalar>(9, seed), {3, 3});
	Matrix mapped_B(std::vector<Scalar>(1, seed), {1, 1});
	Matrix invalid_J(std::vector<Scalar>(6, seed), {2, 3});
	Matrix Hgeo(std::vector<Scalar>(27, seed), {9, 3});
	const auto mapped = run_expected_failure([&] {
		compilation.compile_factorized_mapped<double>(seed, P, valid_A, mapped_B, invalid_J, Hgeo,
			"invalid_mapped_factorized_dimensions", get_codegen_dir());
	});
	INFO(mapped.diagnostic);
	REQUIRE(mapped.failed);
	REQUIRE(mapped.diagnostic.find("inconsistent symbolic dimensions") != std::string::npos);
}
#endif
