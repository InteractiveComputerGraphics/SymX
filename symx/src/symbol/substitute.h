#pragma once
#include <unordered_map>
#include <vector>
#include <cstdint>

#include "Scalar.h"
#include "Vector.h"
#include "Matrix.h"

namespace symx
{
	/*
	* Symbolic substitution: replace symbol leaves with arbitrary expressions, rebuilding
	* the expression graph. A `Symbol` node whose id appears in the map is replaced by
	* its expression; every other node is reconstructed from its substituted children via
	* the ordinary operators (so CSE applies automatically through `_find_or_add`).
	*
	* The replacement expressions must live in the same `Expressions` graph as `e`. The
	* substitution is simultaneous: replacements are inserted as-is and are not themselves
	* re-substituted (so e.g. x -> x + 1 substitutes exactly once).
	*/

	// Map form: symbol id (see Scalar::get_symbol_idx) -> replacement expression.
	Scalar substitute(const Scalar& e, const std::unordered_map<int32_t, Scalar>& sym_to_expr);
	Vector substitute(const Vector& v, const std::unordered_map<int32_t, Scalar>& sym_to_expr);
	Matrix substitute(const Matrix& m, const std::unordered_map<int32_t, Scalar>& sym_to_expr);

	// Convenience form: `from` must all be symbols; replaced element-wise by `to`.
	Scalar substitute(const Scalar& e, const std::vector<Scalar>& from, const std::vector<Scalar>& to);
	Vector substitute(const Vector& v, const std::vector<Scalar>& from, const std::vector<Scalar>& to);
	Matrix substitute(const Matrix& m, const std::vector<Scalar>& from, const std::vector<Scalar>& to);
}
