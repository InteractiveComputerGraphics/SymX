#include "GlobalPotential.h"
#include "../symbol/diff.h"
#include "../symbol/substitute.h"

using namespace symx;

// Helpers ====================================================================
void _check_unique_name(const std::vector<std::unique_ptr<Potential>>& potentials, const std::string& name)
{
    for (auto& potential : potentials) {
        if (potential->get_name() == name) {
            std::cout << "symx error: cannot add_energy with already existing name \"" + name + "\"" << std::endl;
            exit(-1);
        }
    }
}
// ============================================================================

std::shared_ptr<GlobalPotential> symx::GlobalPotential::create()
{
    return std::make_shared<GlobalPotential>();
}

void symx::GlobalPotential::add_potential(const std::string &name, spMWS<double> mws, Element &elem, DefFunc definition_callback)
{
    _check_unique_name(this->potentials, name);
    const Scalar P = definition_callback(*mws, elem);
    this->verify_potential_has_no_quadrature_dofs(*mws);
    std::unique_ptr<Potential> energy_ptr = std::make_unique<Potential>(name, mws, P);
    this->potentials.push_back(std::move(energy_ptr));
}
void symx::GlobalPotential::add_potential(const std::string& name, spMWS<double> mws, Element& elem, DefWCondFunc definition_with_conditional_callback)
{
    _check_unique_name(this->potentials, name);
    auto [P, C] = definition_with_conditional_callback(*mws, elem);
    this->verify_potential_has_no_quadrature_dofs(*mws);
    std::unique_ptr<Potential> energy_ptr = std::make_unique<Potential>(name, mws, P, C);
    this->potentials.push_back(std::move(energy_ptr));
}

static std::unique_ptr<Potential> build_fem_potential(
    const std::string& name, spMWS<double> mws,
    const FemEnergySplit& split, const Scalar* condition)
{
    if (split.x.rows() != split.B.rows() || split.x.cols() != 3 || split.B.cols() < 1 || !split.material) {
        std::cout << "symx error: add_potential_fem() expects x: n_nodes x 3 and B: n_nodes x n_features (n_features >= 1)." << std::endl;
        exit(-1);
    }

    // Differentiate the material with respect to an independent F, then substitute F = x^T B.
    // P and A are the material gradient and Hessian at that F. With B independent of the state,
    // the factorized kernel applies the chain rule to form the nodal gradient and Hessian.
    const Matrix real_F = split.x.transpose() * split.B;
    const Matrix abstract_F = mws->get_workspace().make_matrix(real_F.shape());
    const Scalar material = split.material(abstract_F);
    const auto symbols = abstract_F.values();
    const auto actual = real_F.values();
    DiffCache cache;
    const Matrix A = substitute(hessian(material, symbols, cache), symbols, actual);
    const Vector P = substitute(gradient(material, symbols, cache), symbols, actual);
    const Scalar energy = substitute(material, symbols, actual);

    auto potential = condition
        ? std::make_unique<Potential>(name, mws, energy, *condition)
        : std::make_unique<Potential>(name, mws, energy);
    // Keep the evaluated blocks for second-order compilation and the original material for checks.
    potential->set_factorization(std::make_unique<FactorizationData>(
        FactorizationData{A, P, split.B, split.x, material}));
    return potential;
}

void symx::GlobalPotential::add_potential_fem(const std::string& name, spMWS<double> mws, Element& elem, FemSplitFunc callback)
{
    _check_unique_name(this->potentials, name);
    const FemEnergySplit split = callback(*mws, elem);
    this->verify_potential_has_no_quadrature_dofs(*mws);
    this->potentials.push_back(build_fem_potential(name, mws, split, nullptr));
}
void symx::GlobalPotential::add_potential_fem(const std::string& name, spMWS<double> mws, Element& elem, FemSplitWCondFunc callback)
{
    _check_unique_name(this->potentials, name);
    auto [split, condition] = callback(*mws, elem);
    this->verify_potential_has_no_quadrature_dofs(*mws);
    this->potentials.push_back(build_fem_potential(name, mws, split, &condition));
}
void symx::GlobalPotential::add_potential_fem(const std::string& name, const std::vector<int32_t>& arr, int32_t stride, FemSplitFunc callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(arr, stride);
    this->add_potential_fem(name, mws, elem, callback);
}
void symx::GlobalPotential::add_potential_fem(const std::string& name, const std::vector<int32_t>& arr, int32_t stride, FemSplitWCondFunc callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(arr, stride);
    this->add_potential_fem(name, mws, elem, callback);
}
void symx::GlobalPotential::add_potential_fem(const std::string& name, std::function<const int32_t* ()> data,
    std::function<int32_t()> n_elements, int32_t stride, FemSplitFunc callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(data, n_elements, stride);
    this->add_potential_fem(name, mws, elem, callback);
}
void symx::GlobalPotential::add_potential_fem(const std::string& name, std::function<const int32_t* ()> data,
    std::function<int32_t()> n_elements, int32_t stride, FemSplitWCondFunc callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(data, n_elements, stride);
    this->add_potential_fem(name, mws, elem, callback);
}
void symx::GlobalPotential::add_potential(const std::string& name, std::function<const int32_t* ()> data, std::function<int32_t()> n_elements, const int32_t stride, 
    std::function<Scalar(MappedWorkspace<double>&, Element&)> definition_callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(data, n_elements, stride);
    this->add_potential(name, mws, elem, definition_callback);
}
void symx::GlobalPotential::add_potential(const std::string& name, std::function<const int32_t* ()> data, std::function<int32_t()> n_elements, const int32_t stride,
    std::function<std::pair<Scalar, Scalar>(MappedWorkspace<double>&, Element&)> definition_with_conditional_callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(data, n_elements, stride);
    this->add_potential(name, mws, elem, definition_with_conditional_callback);
}
void symx::GlobalPotential::add_potential(const std::string &name, const std::vector<int32_t> &arr, const int32_t n_items_per_element, DefFunc definition_callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(arr, n_items_per_element);
    this->add_potential(name, mws, elem, definition_callback);
}
void symx::GlobalPotential::add_potential(const std::string& name, const std::vector<int32_t>& arr, const int32_t n_items_per_element, DefWCondFunc definition_with_conditional_callback)
{
    auto [mws, elem] = MappedWorkspace<double>::create(arr, n_items_per_element);
    this->add_potential(name, mws, elem, definition_with_conditional_callback);
}

void symx::GlobalPotential::add_dof(std::function<std::uintptr_t()> id, std::function<double *()> data, std::function<int32_t()> flat_size, const std::string &name)
{
    // Check DoF is not quadrature data
    for (const auto& potential : this->potentials) {
        if (potential->get_mws()->has_iteration_map_with_id(id())) {
            std::cout << "GlobalPotential::add_dof() cannot register quadrature data as a DoF." << std::endl;
            exit(-1);
        }
    }

    // DOF maps store flat arrays: stride=1, no connectivity, no symbol binding (first_symbol_idx=-1)
    // flat_size returns total number of doubles, treated as n_elements with stride=1
    this->dof_maps.emplace_back(id, data, flat_size, /*stride=*/ 1, /*connectivity_index=*/-1, /*first_symbol_idx=*/-1);
    this->dof_labels.push_back(name);
}

void symx::GlobalPotential::verify_potential_has_no_quadrature_dofs(const MappedWorkspace<double>& mws) const
{
    for (const auto& dof_map : this->dof_maps) {
        if (mws.has_iteration_map_with_id(dof_map.id())) {
            std::cout << "GlobalPotential::add_potential() cannot use an existing DoF as quadrature data." << std::endl;
            exit(-1);
        }
    }
}
void symx::GlobalPotential::add_dof(EigenMatrixRM<double> &arr, const std::string &name)
{
    this->add_dof(
        [&arr]() { return reinterpret_cast<std::uintptr_t>(&arr); },
        [&arr]() { return arr.data(); }, 
        [&arr]() { return (int32_t)(arr.size()); }, 
        name);
}
void symx::GlobalPotential::add_dof(double &scalar, const std::string &name)
{
    this->add_dof(
        [&scalar]() { return reinterpret_cast<std::uintptr_t>(&scalar); },
        [&scalar]() { return &scalar; }, 
        []() { return 1; }, 
        name);
}

const std::vector<int32_t> &symx::GlobalPotential::get_dofs_offsets()
{
    this->dof_offsets.clear();
    this->dof_offsets.push_back(0);
    for (const DataMap<double>& dof_map : this->dof_maps) {
        this->dof_offsets.push_back(this->dof_offsets.back() + (int32_t)dof_map.size());
    }
    return this->dof_offsets;
}

int32_t symx::GlobalPotential::get_n_dof_sets() const
{
    return (int32_t)this->dof_maps.size();
}

int32_t symx::GlobalPotential::get_n_dofs(int32_t set_i) const
{
    return (int32_t)this->dof_maps[set_i].size();
}

int32_t symx::GlobalPotential::get_total_n_dofs() const
{
    int32_t total_n_dofs = 0;
    for (const DataMap<double>& dof_map : this->dof_maps) {
        total_n_dofs += (int32_t)dof_map.size();
    }
    return total_n_dofs;
}
void symx::GlobalPotential::get_dofs(double *u) const
{
    int32_t offset = 0;
    for (const DataMap<double>& dof_map : this->dof_maps) {
        const int32_t n_dofs = (int32_t)dof_map.size();
        if (n_dofs == 0) { continue; }
        std::memcpy(u + offset, dof_map.data(), n_dofs * sizeof(double));
        offset += n_dofs;
    }
}
void symx::GlobalPotential::apply_dof_increment(const double *du)
{
    int32_t offset = 0;
    for (const DataMap<double>& dof_map : this->dof_maps) {
        const int32_t n_dofs = (int32_t)dof_map.size();
        if (n_dofs == 0) { continue; }
        double* dof_data = dof_map.data();
        for (int32_t i = 0; i < n_dofs; i++) {
            dof_data[i] += du[offset + i];
        }
        offset += n_dofs;
    }
}
void symx::GlobalPotential::set_dofs(const double *u)
{
    int32_t offset = 0;
    for (const DataMap<double>& dof_map : this->dof_maps) {
        const int32_t n_dofs = (int32_t)dof_map.size();
        if (n_dofs == 0) { continue; }
        double* dof_data = dof_map.data();
        std::memcpy(dof_data, u + offset, n_dofs * sizeof(double));
        offset += n_dofs;
    }
}

const std::vector<std::unique_ptr<Potential>> &symx::GlobalPotential::get_potentials() const
{
    return this->potentials;
}

const std::vector<DataMap<double>> &symx::GlobalPotential::get_dof_maps() const
{
    return this->dof_maps;
}

std::string symx::GlobalPotential::get_dof_label(int set_i) const
{
    if (set_i >= this->get_n_dof_sets()) {
        std::cout << "symx error GlobalPotential::get_dof_label(): set " << set_i << " out of bounds." << std::endl;
        exit(-1);
    }
    return this->dof_labels[set_i];
}

int32_t symx::GlobalPotential::get_n_potentials() const
{
    return (int32_t)this->potentials.size();
}

void GlobalPotential::add_dof(std::vector<double>& arr, const std::string& name)
{
    this->add_dof(
        [&arr]() { return reinterpret_cast<std::uintptr_t>(&arr); },
        [&arr]() { return arr.data(); }, 
        [&arr]() { return (int32_t)(arr.size()); }, 
        name);
}
