#include "substitute.h"

#include <iostream>

using namespace symx;

namespace
{
	// Structural recursion mirroring diff_impl (diff.cpp). `cache` memoizes by expr_id so
	// shared subexpressions (DAG) are visited once, keeping this linear in graph size.
	Scalar subst_rec(const Scalar& s, const std::unordered_map<int32_t, Scalar>& map,
	                 std::unordered_map<int32_t, Scalar>& cache)
	{
		const auto cached = cache.find(s.expr_id);
		if (cached != cache.end()) {
			return cached->second;
		}

		auto rebuild = [&]() -> Scalar {
			const ExprType type = s.expr.type;

			// Leaves
			if (type == ExprType::Symbol) {
				const auto it = map.find(s.expr.a);  // expr.a holds the symbol id
				return (it != map.end()) ? it->second : s;
			}
			if (type == ExprType::Zero || type == ExprType::One || type == ExprType::ConstantFloat) {
				return s;
			}

			// Conditional
			if (type == ExprType::Branch) {
				return branch(subst_rec(s.get_condition(), map, cache),
				              subst_rec(s.left(), map, cache),
				              subst_rec(s.right(), map, cache));
			}

			// Operations: substitute children, then rebuild via the matching operator.
			const Scalar u = subst_rec(s.left(), map, cache);
			switch (type) {
			case ExprType::Add:        return u + subst_rec(s.right(), map, cache);
			case ExprType::Sub:        return u - subst_rec(s.right(), map, cache);
			case ExprType::Mul:        return u * subst_rec(s.right(), map, cache);
			case ExprType::Reciprocal: return u.inv();
			case ExprType::PowN:       return u.powN(s.expr.b);                       // expr.b = exponent
			case ExprType::PowF:       return u.powF(subst_rec(s.right(), map, cache));
			case ExprType::Sqrt:       return sqrt(u);
			case ExprType::Ln:         return ln(u);
			case ExprType::Log10:      return log10(u);
			case ExprType::Exp:        return exp(u);
			case ExprType::Sin:        return sin(u);
			case ExprType::Cos:        return cos(u);
			case ExprType::Tan:        return tan(u);
			case ExprType::ArcSin:     return asin(u);
			case ExprType::ArcCos:     return acos(u);
			case ExprType::ArcTan:     return atan(u);
			case ExprType::Print:      return u.print();
			default:
				std::cout << "symx error: substitute() found an unhandled ExprType: "
				          << get_label(type) << std::endl;
				exit(-1);
			}
		};

		const Scalar result = rebuild();
		cache.emplace(s.expr_id, result);
		return result;
	}

	std::unordered_map<int32_t, Scalar> build_map(const Scalar& e,
	                                               const std::vector<Scalar>& from,
	                                               const std::vector<Scalar>& to)
	{
		if (from.size() != to.size()) {
			std::cout << "symx error: substitute() got 'from' and 'to' of different sizes ("
			          << from.size() << " vs " << to.size() << ")." << std::endl;
			exit(-1);
		}
		std::unordered_map<int32_t, Scalar> map;
		map.reserve(from.size());
		for (std::size_t i = 0; i < from.size(); i++) {
			if (!from[i].is_symbol()) {
				std::cout << "symx error: substitute() 'from' entries must all be symbols." << std::endl;
				exit(-1);
			}
			if (from[i].expressions != e.expressions || to[i].expressions != e.expressions) {
				std::cout << "symx error: substitute() requires 'from'/'to' to live in the same "
				             "expression graph as the substituted expression." << std::endl;
				exit(-1);
			}
			map.emplace(from[i].get_symbol_idx(), to[i]);
		}
		return map;
	}
}


// ===================================== Public API =====================================
Scalar symx::substitute(const Scalar& e, const std::unordered_map<int32_t, Scalar>& sym_to_expr)
{
	std::unordered_map<int32_t, Scalar> cache;
	return subst_rec(e, sym_to_expr, cache);
}

Vector symx::substitute(const Vector& v, const std::unordered_map<int32_t, Scalar>& sym_to_expr)
{
	std::unordered_map<int32_t, Scalar> cache;  // shared across entries for reuse
	std::vector<Scalar> out;
	out.reserve(v.size());
	for (const Scalar& s : v.values()) {
		out.push_back(subst_rec(s, sym_to_expr, cache));
	}
	return Vector(out);
}

Matrix symx::substitute(const Matrix& m, const std::unordered_map<int32_t, Scalar>& sym_to_expr)
{
	std::unordered_map<int32_t, Scalar> cache;  // shared across entries for reuse
	std::vector<Scalar> out;
	out.reserve(m.values().size());
	for (const Scalar& s : m.values()) {
		out.push_back(subst_rec(s, sym_to_expr, cache));
	}
	return Matrix(out, m.shape());
}

Scalar symx::substitute(const Scalar& e, const std::vector<Scalar>& from, const std::vector<Scalar>& to)
{
	return substitute(e, build_map(e, from, to));
}

Vector symx::substitute(const Vector& v, const std::vector<Scalar>& from, const std::vector<Scalar>& to)
{
	const Scalar& any = v.values().at(0);
	return substitute(v, build_map(any, from, to));
}

Matrix symx::substitute(const Matrix& m, const std::vector<Scalar>& from, const std::vector<Scalar>& to)
{
	const Scalar& any = m.values().at(0);
	return substitute(m, build_map(any, from, to));
}
