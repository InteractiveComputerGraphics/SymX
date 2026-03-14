#include "catch.hpp"
#include <symx>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <array>

using namespace symx;

TEST_CASE("mapped_workspace_make_all", "[mapped_workspace]")
{
    // 1. Setup Data
    // Connectivity: 2 elements, stride 2.
    // Element 0: [0, 1]
    // Element 1: [1, 2]
    std::vector<std::array<int, 2>> connectivity = {{0, 1}, {1, 2}};
    
    // Data Arrays
    // Scalars: [1, 2, 3]
    std::vector<double> scalar_data = {1.0, 2.0, 3.0};
    
    // Vectors (size 2): [[1,1], [2,2], [3,3]]
    EigenMatrixRM<double> vector_data(3, 2);
    vector_data << 1.0, 1.0, 
                   2.0, 2.0, 
                   3.0, 3.0;
                   
    // Matrices (2x2): 
    // P0: [[1,0],[0,1]] -> trace 2
    // P1: [[2,0],[0,2]] -> trace 4
    // P2: [[3,0],[0,3]] -> trace 6
    EigenMatrixRM<double> matrix_data(3, 4);
    matrix_data << 1.0, 0.0, 0.0, 1.0,
                   2.0, 0.0, 0.0, 2.0,
                   3.0, 0.0, 0.0, 3.0;

    // Fixed data
    double fixed_scalar_val = 10.0;
    std::vector<double> fixed_vector_val = {10.0, 20.0};
    std::vector<double> fixed_matrix_val = {10.0, 0.0, 0.0, 10.0}; // 2x2 identity-like * 10

    // 2. Create Workspace
    auto [mws, conn] = MappedWorkspace<double>::create(connectivity);

    // 3. Make Symbols

    // --- Fixed Data ---
    // make_scalar(lambda)
    Scalar s_fixed_lambda = mws->make_scalar(
        [&]() { return reinterpret_cast<std::uintptr_t>(&fixed_scalar_val); }, 
        [&]() { return &fixed_scalar_val; });

    // make_scalar(val)
    Scalar s_fixed_val = mws->make_scalar(fixed_scalar_val);
    
    // make_vector(lambda, stride)
    Vector v_fixed_lambda = mws->make_vector(
        [&]() { return reinterpret_cast<std::uintptr_t>(&fixed_vector_val); }, 
        [&]() { return fixed_vector_val.data(); }, 
        2);
    
    // make_vector(arr) - STATIC_VECTOR
    Eigen::VectorXd v_fixed_eigen(2); v_fixed_eigen << 10.0, 20.0;
    Vector v_fixed_arr = mws->make_vector(v_fixed_eigen);

    // make_matrix(lambda, shape)
    Matrix m_fixed_lambda = mws->make_matrix(
        [&]() { return reinterpret_cast<std::uintptr_t>(&m_fixed_lambda); }, 
        std::function<const double*()>([&]() { return fixed_matrix_val.data(); }), 
        {2, 2});
    
    // make_matrix(arr, shape)
    Eigen::MatrixXd m_fixed_eigen(2, 2); m_fixed_eigen << 10.0, 0.0, 0.0, 10.0;
    Matrix m_fixed_arr = mws->make_matrix(m_fixed_eigen, {2, 2});


    // --- Indexed Data (Scalars) ---
    // make_scalar(lambda, n_elements, idx)
    // Accessing index 0 of connectivity (A)
    Scalar s_idx_lambda = mws->make_scalar(
        [&]() { return reinterpret_cast<std::uintptr_t>(&scalar_data); }, 
        [&]() { return scalar_data.data(); }, 
        [&]() { return (int)scalar_data.size(); }, 
        conn[0]
    );
    
    // make_scalar(arr, idx)
    // Accessing index 1 of connectivity (B)
    Scalar s_idx_arr = mws->make_scalar(scalar_data, conn[1]);

    // make_scalars(lambda, n_elements, indices)
    std::vector<Scalar> s_vec_lambda = mws->make_scalars(
        [&]() { return reinterpret_cast<std::uintptr_t>(&scalar_data); }, 
        [&]() { return scalar_data.data(); },
        [&]() { return (int)scalar_data.size(); },
        {conn[0], conn[1]}
    );

    // make_scalars(arr, indices)
    std::vector<Scalar> s_vec_arr = mws->make_scalars(scalar_data, {conn[0], conn[1]});


    // --- Indexed Data (Vectors) ---
    // make_vector(lambda, n_elements, stride, idx)
    Vector v_idx_lambda = mws->make_vector(
        [&]() { return reinterpret_cast<std::uintptr_t>(&vector_data); }, 
        [&]() { return vector_data.data(); },
        [&]() { return (int)vector_data.rows(); },
        2,
        conn[0]
    );

    // make_vector(arr, idx)
    Vector v_idx_arr = mws->make_vector(vector_data, conn[1]);

    // make_vectors(lambda, n_elements, stride, indices)
    std::vector<Vector> v_vec_lambda = mws->make_vectors(
        [&]() { return reinterpret_cast<std::uintptr_t>(&vector_data); }, 
        [&]() { return vector_data.data(); },
        [&]() { return (int)vector_data.rows(); },
        2,
        {conn[0], conn[1]}
    );

    // make_vectors(arr, indices)
    std::vector<Vector> v_vec_arr = mws->make_vectors(vector_data, {conn[0], conn[1]});

    // make_vectors(arr, element) - convenience for all indices in element
    std::vector<Vector> v_vec_elem = mws->make_vectors(vector_data, conn);


    // --- Indexed Data (Matrices) ---
    // make_matrix(lambda, n_elements, shape, idx)
    Matrix m_idx_lambda = mws->make_matrix(
        [&]() { return reinterpret_cast<std::uintptr_t>(&matrix_data); }, 
        [&]() { return matrix_data.data(); },
        [&]() { return (int)matrix_data.rows(); },
        {2, 2},
        conn[0]
    );

    // make_matrix(arr, shape, idx)
    Matrix m_idx_arr = mws->make_matrix(matrix_data, {2, 2}, conn[1]);

    // make_matrices(lambda, n_elements, shape, indices)
    std::vector<Matrix> m_vec_lambda = mws->make_matrices(
        [&]() { return reinterpret_cast<std::uintptr_t>(&matrix_data); }, 
        [&]() { return matrix_data.data(); },
        [&]() { return (int)matrix_data.rows(); },
        {2, 2},
        {conn[0], conn[1]}
    );

    // make_matrices(arr, shape, indices)
    std::vector<Matrix> m_vec_arr = mws->make_matrices(matrix_data, {2, 2}, {conn[0], conn[1]});


    // --- Constants ---
    Scalar c_zero = mws->make_zero();
    Scalar c_one = mws->make_one();
    Vector c_zero_vec = mws->make_zero_vector(2);
    Matrix c_zero_mat = mws->make_zero_matrix({2, 2});
    Matrix c_identity = mws->make_identity_matrix(2);


    // 4. Expression
    // Let's sum everything up into a single scalar to verify.
    // We need to reduce vectors and matrices to scalars (e.g. norm or trace or sum of components).
    
    Scalar total = mws->make_zero();

    auto add_scalar = [&](Scalar s) { total = total + s; };
    auto add_vector = [&](Vector v) { total = total + v[0] + v[1]; };
    auto add_matrix = [&](Matrix m) { total = total + m.trace(); }; // trace is sum of diagonal

    // Fixed
    add_scalar(s_fixed_lambda);
    add_scalar(s_fixed_val);
    add_vector(v_fixed_lambda);
    add_vector(v_fixed_arr);
    add_matrix(m_fixed_lambda);
    add_matrix(m_fixed_arr);

    // Indexed Scalars
    add_scalar(s_idx_lambda);
    add_scalar(s_idx_arr);
    for(auto& s : s_vec_lambda) add_scalar(s);
    for(auto& s : s_vec_arr) add_scalar(s);

    // Indexed Vectors
    add_vector(v_idx_lambda);
    add_vector(v_idx_arr);
    for(auto& v : v_vec_lambda) add_vector(v);
    for(auto& v : v_vec_arr) add_vector(v);
    for(auto& v : v_vec_elem) add_vector(v);

    // Indexed Matrices
    add_matrix(m_idx_lambda);
    add_matrix(m_idx_arr);
    for(auto& m : m_vec_lambda) add_matrix(m);
    for(auto& m : m_vec_arr) add_matrix(m);

    // Constants
    add_scalar(c_zero);
    add_scalar(c_one);
    add_vector(c_zero_vec);
    add_matrix(c_zero_mat);
    add_matrix(c_identity);

    // 5. Compile and Run
    CompiledInLoop<double> loop(mws, {total}, "test_make_all", symx::get_codegen_dir());

    loop.run(1, [&](View<double> solution, const int32_t i, const int32_t thread_id, const View<int32_t> connectivity) {
        // Verification Logic
        // Calculate expected value for element i
        double expected = 0.0;
        
        int idx0 = connectivity[0];
        int idx1 = connectivity[1];

        // Fixed
        expected += fixed_scalar_val; // s_fixed_lambda
        expected += fixed_scalar_val; // s_fixed_val
        expected += fixed_vector_val[0] + fixed_vector_val[1]; // v_fixed_lambda
        expected += 10.0 + 20.0; // v_fixed_arr
        expected += fixed_matrix_val[0] + fixed_matrix_val[3]; // m_fixed_lambda (trace: 10+10)
        expected += 10.0 + 10.0; // m_fixed_arr (trace)

        // Indexed Scalars
        expected += scalar_data[idx0]; // s_idx_lambda
        expected += scalar_data[idx1]; // s_idx_arr
        expected += scalar_data[idx0] + scalar_data[idx1]; // s_vec_lambda
        expected += scalar_data[idx0] + scalar_data[idx1]; // s_vec_arr

        // Indexed Vectors (sum of components)
        // vector_data[idx] is [idx+1, idx+1]
        auto get_vec_sum = [&](int idx) { return vector_data(idx, 0) + vector_data(idx, 1); };
        
        expected += get_vec_sum(idx0); // v_idx_lambda
        expected += get_vec_sum(idx1); // v_idx_arr
        expected += get_vec_sum(idx0) + get_vec_sum(idx1); // v_vec_lambda
        expected += get_vec_sum(idx0) + get_vec_sum(idx1); // v_vec_arr
        expected += get_vec_sum(idx0) + get_vec_sum(idx1); // v_vec_elem

        // Indexed Matrices (trace)
        // matrix_data[idx] is [[idx+1, 0], [0, idx+1]]
        auto get_mat_trace = [&](int idx) { return matrix_data(idx, 0) + matrix_data(idx, 3); };

        expected += get_mat_trace(idx0); // m_idx_lambda
        expected += get_mat_trace(idx1); // m_idx_arr
        expected += get_mat_trace(idx0) + get_mat_trace(idx1); // m_vec_lambda
        expected += get_mat_trace(idx0) + get_mat_trace(idx1); // m_vec_arr

        // Constants
        expected += 0.0; // c_zero
        expected += 1.0; // c_one
        expected += 0.0; // c_zero_vec
        expected += 0.0; // c_zero_mat
        expected += 2.0; // c_identity (trace of 2x2 identity is 2)

        REQUIRE(solution[0] == Approx(expected));
    });
}
