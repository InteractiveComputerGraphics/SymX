#pragma once
#include <vector>
#include <array>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <set>

#include <symx>

using namespace symx;

namespace symx::detail {
struct PairHash {
    size_t operator()(const std::pair<int,int>& p) const noexcept {
        // 64-bit mix; fine for 32-bit ints too
        uint64_t a = static_cast<uint64_t>(static_cast<uint32_t>(p.first));
        uint64_t b = static_cast<uint64_t>(static_cast<uint32_t>(p.second));
        // splitmix64-inspired
        auto mix = [](uint64_t x) {
            x += 0x9e3779b97f4a7c15ull;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
            x = x ^ (x >> 31);
            return x;
        };
        return static_cast<size_t>(mix(a) ^ (mix(b) + 0x9e3779b97f4a7c15ull + (mix(a) << 6) + (mix(a) >> 2)));
    }
};
} // namespace detail

// Build dihedral (hinge) connectivity for interior edges.
// For each interior edge {i,j} shared by triangles (i,j,k) and (j,i,l),
// returns a record {hinge_idx, v0=i, v1=j, v2=k, v3=l}.
// Edge direction is canonicalized as (v0=min(i,j), v1=max(i,j)) for consistency.
inline symx::LabelledConnectivity<5>
build_dihedral_connectivity(const std::vector<std::array<int,3>>& triangles)
{
    struct EdgeInfo { int opp = -1; int tri = -1; }; // first incident triangle's opposite vertex + id

    std::unordered_map<std::pair<int,int>, EdgeInfo, symx::detail::PairHash> edge_map;
    edge_map.reserve(triangles.size() * 3);

    auto canon = [](int a, int b) -> std::pair<int,int> {
        return (a < b) ? std::make_pair(a,b) : std::make_pair(b,a);
    };

    // helper: record an edge (u,v) with opposite vertex w from triangle t
    auto add_edge = [&](int u, int v, int w, int t,
                        symx::LabelledConnectivity<5>& hinges)
    {
        auto key = canon(u,v);
        auto it = edge_map.find(key);
        if (it == edge_map.end()) {
            edge_map.emplace(key, EdgeInfo{w, t});
        } 
        else {
            // Found second triangle sharing this edge -> create hinge
            const int i = key.first;
            const int j = key.second;
            const int k = it->second.opp; // opposite vertex from first tri
            const int l = w;              // opposite vertex from second tri
            hinges.numbered_push_back({ i, j, k, l });
        }
    };

    symx::LabelledConnectivity<5> hinges{{"idx", "v0", "v1", "v2", "v3"}};
    for (int t = 0; t < static_cast<int>(triangles.size()); ++t) {
        const auto& tri = triangles[t];
        const int a = tri[0], b = tri[1], c = tri[2];

        // edges with their opposite vertex
        add_edge(a, b, c, t, hinges);
        add_edge(b, c, a, t, hinges);
        add_edge(c, a, b, t, hinges);
    }

    return hinges;
}

inline LabelledConnectivity<3> get_unique_edges_conn(const std::vector<std::array<int, 3>>& triangles)
{
    std::set<std::array<int, 2>> edge_set;
    for (const auto& tri : triangles) {
        for (int i = 0; i < 3; i++) {
            int v1 = tri[i];
            int v2 = tri[(i + 1) % 3];
            if (v1 > v2) std::swap(v1, v2); // Ensure v1 < v2 for consistent edge representation
            edge_set.insert({v1, v2});
        }
    }

    LabelledConnectivity<3> edge_conn({"idx", "a", "b"});
    for (const auto& e : edge_set) {
        edge_conn.numbered_push_back({e[0], e[1]});
    }
    return edge_conn;
}
inline LabelledConnectivity<4> get_triangle_conn(const std::vector<std::array<int, 3>>& triangles)
{
    LabelledConnectivity<4> triangle_conn({"idx", "a", "b", "c"});
    for (const auto& tri : triangles) {
        triangle_conn.numbered_push_back({tri[0], tri[1], tri[2]});
    }
    return triangle_conn;
}

inline vtkio::CellType fem_element_to_vtk_cell_type(const FEM_Element element)
{
    switch (element) {
    case FEM_Element::Tet4: return vtkio::CellType::Tetra;
    case FEM_Element::Tet10: return vtkio::CellType::Tetra10;
    case FEM_Element::Hex8: return vtkio::CellType::Hexahedron;
    case FEM_Element::Hex27: return vtkio::CellType::Hexahedron27;
    default:
        throw std::runtime_error("fem_element_to_vtk_cell_type: Unsupported element type");
    }
}
inline void prepare_for_export(std::vector<int>& conn, const FEM_Element element)
{
    // vtkio::CellType::Hexahedron27 has one node flipped compared to our integration convention
    if (element == FEM_Element::Hex27) {
        for (int e = 0; e < conn.size() / 27; ++e) {
            std::swap(conn[e * 27 + 22], conn[e * 27 + 23]);  // Swap (22) and (23)
        }
    }
}
