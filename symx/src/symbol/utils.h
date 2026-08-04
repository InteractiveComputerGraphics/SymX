#pragma once
#include "Scalar.h"
#include "Vector.h"
#include "Matrix.h"

namespace symx
{
	std::vector<Scalar> collect_scalars(const std::vector<Vector>& vectors);
	std::vector<Scalar> collect_scalars(const std::vector<std::vector<Scalar>>& vectors);
	std::vector<Scalar> collect_scalars(const std::vector<Scalar>& lst1, const std::vector<Scalar>& lst2);
	std::string get_checksum(const std::vector<Scalar>& exprs);
}
