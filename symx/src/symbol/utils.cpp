#include "utils.h"

using namespace symx;

std::vector<Scalar> symx::collect_scalars(const std::vector<Vector>& vectors)
{
	std::vector<Scalar> scalars;
	for (const Vector& v : vectors) {
		for (const Scalar& s : v.values()) {
			scalars.push_back(s);
		}
	}
	return scalars;
}

std::vector<Scalar> symx::collect_scalars(const std::vector<std::vector<Scalar>> &vectors)
{
	std::vector<Scalar> scalars;
	for (const std::vector<Scalar>& v : vectors) {
		for (const Scalar& s : v) {
			scalars.push_back(s);
		}
	}
	return scalars;
}

std::vector<Scalar> symx::collect_scalars(const std::vector<Scalar> &lst1, const std::vector<Scalar> &lst2)
{
    std::vector<Scalar> scalars;
    scalars.reserve(lst1.size() + lst2.size());
    scalars.insert(scalars.end(), lst1.begin(), lst1.end());
    scalars.insert(scalars.end(), lst2.begin(), lst2.end());
    return scalars;
}

std::string symx::get_checksum(const std::vector<Scalar> &exprs)
{
	std::string cache_id;
	for (const Scalar& s : exprs) {
		cache_id += s.get_checksum();
	} 
	return cache_id;
}
