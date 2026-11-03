#include "Expressions.h"

#include <limits>
#include <cmath>
#include <picoSHA2/picosha2.h>

using namespace symx;

// CSE Mode
CSE symx::_cse_mode = CSE::Safe; // Default to false (no NaN checking)

void symx::set_cse_mode(CSE mode)
{
	_cse_mode = mode;
}

CSE symx::get_cse_mode()
{
	return _cse_mode;
}

Expressions::Expressions()
{
	// Zero and One are always at indices 0 and 1.
	this->_find_or_add(Expr(ExprType::Zero, -1, -1));
	this->_find_or_add(Expr(ExprType::One, -1, -1));
}
int32_t Expressions::declare_symbol()
{
	this->symbol_values_set.push_back(false);
	this->symbol_values.push_back(0.0);
	this->symbol_locations.push_back((int32_t)this->expressions.size());

	// Add it to the expression graph
	const int32_t symbol_id = (int32_t)this->symbol_locations.size() - 1;
	const int32_t expr_id = (int32_t)this->expressions.size();
	this->expressions.push_back(Expr(ExprType::Symbol, symbol_id, -1));

	return expr_id;
}
int32_t Expressions::declare_constant_float(double constant)
{
	// Check if it is zero
	if (std::abs(constant) < 10.0 * std::numeric_limits<double>::epsilon()) {
		return this->get_zero_idx();
	}

	// Check if it is one
	else if (std::abs(constant - 1.0) < 10.0 * std::numeric_limits<double>::epsilon()) {
		return this->get_one_idx();
	}

	// Check if it is already declared
	else {
		// Create the Expr
		Expr expr(ExprType::ConstantFloat, 0, 0, 0);
		expr.pack_double(constant);
		return this->_find_or_add(expr);
	}
}
int32_t Expressions::add_operation(ExprType type, int32_t a, int32_t b)
{
	if (!is_operation(type)) {
		std::cout << "symx error: cannot use Expressions::add_operation() for non-operation types." << std::endl;
		exit(-1);
	}

	return this->_find_or_add(Expr(type, a, b));
}
int32_t Expressions::add_branch(int32_t cond_expr_id, int32_t positive_branch_expr_id, int32_t negative_branch_expr_id)
{
	int32_t expr_id = (int32_t)this->expressions.size();
	this->expressions.push_back(Expr(ExprType::Branch, positive_branch_expr_id, negative_branch_expr_id, cond_expr_id));
	return expr_id;
}
int32_t Expressions::size() const
{
	return (int32_t)this->expressions.size();
}
int32_t Expressions::get_n_symbols() const
{
	return (int32_t)this->symbol_locations.size();
}
int32_t symx::Expressions::get_symbol_location(int32_t symbol_id) const
{
    return this->symbol_locations[symbol_id];
}
const std::vector<Expr> &Expressions::get_expressions() const
{
	return this->expressions;
}
int32_t Expressions::get_zero_idx()
{
	return 0;
}
int32_t Expressions::get_one_idx()
{
	return 1;
}
void Expressions::set_value(int32_t expr_id, double val)
{
	const Expr& expr = this->expressions[expr_id];
	if (expr.type != ExprType::Symbol) {
		std::cout << "symx error: Can't set a non-symbol value." << std::endl;
		exit(-1);
	}
	const int symbol_id = expr.a;
	this->symbol_values[symbol_id] = val;
	this->symbol_values_set[symbol_id] = static_cast<uint8_t>(true);
}

double Expressions::get_value(int32_t expr_id) const
{
	const Expr& expr = this->expressions[expr_id];
	if (expr.type != ExprType::Symbol) {
		std::cout << "symx error: Can't get a value from a non-symbol." << std::endl;
		exit(-1);
	}
	if (!this->symbol_values_set[expr.a]) {
		std::cout << "symx error: Can't read a value not set. Missing symbol [idx=" << expr.a << "]" << std::endl;
	}

	const int symbol_id = expr.a;
	return this->symbol_values[symbol_id];
}

double Expressions::eval(int32_t expr_id) const
{
	// Memoize over the (append-only, topologically-ordered) graph so shared subexpressions
	// are evaluated once. Without this, a shared DAG is walked as its exponentially-sized
	// equivalent tree. The cache is per-call because symbol values change between calls.
	std::vector<double> memo(expr_id + 1);
	std::vector<uint8_t> done(expr_id + 1, 0);
	return this->_eval_rec(expr_id, memo, done);
}

double Expressions::_eval_rec(int32_t expr_id, std::vector<double>& memo, std::vector<uint8_t>& done) const
{
	if (done[expr_id]) {
		return memo[expr_id];
	}

	// Operands are always created before their parent, so every reachable node has a strictly
	// smaller index and fits in `memo` (sized expr_id+1). Guard that invariant in debug builds.
	auto rec = [&](int32_t child) -> double {
		assert(child >= 0 && child < expr_id &&
			"Expressions::_eval_rec: operand index must be < node index (broken graph ordering)");
		return this->_eval_rec(child, memo, done);
	};

	const Expr& expr = this->expressions[expr_id];
	double result;
	switch (expr.type)
	{
	case ExprType::Zero:
		result = 0.0; break;
	case ExprType::One:
		result = 1.0; break;
	case ExprType::ConstantFloat:
		result = expr.unpack_double(); break;
	case ExprType::Symbol:
		if (!this->symbol_values_set[expr.a]) {
			std::cout << "symx error: Cannot evaluate an expression if all symbols are not set.\nMissing symbol [idx=" << expr.a << "]" << std::endl;
		}
		result = this->symbol_values[expr.a]; break;
	case ExprType::Add:
		result = rec(expr.a) + rec(expr.b); break;
	case ExprType::Sub:
		result = rec(expr.a) - rec(expr.b); break;
	case ExprType::Mul:
		result = rec(expr.a) * rec(expr.b); break;
	case ExprType::Reciprocal:
		result = 1.0 / rec(expr.a); break;
	case ExprType::PowN:
		result = std::pow(rec(expr.a), (int)expr.b); break;  // expr.b is the exponent, not an index
	case ExprType::PowF:
		result = std::pow(rec(expr.a), rec(expr.b)); break;
	case ExprType::Sqrt:
		result = std::sqrt(rec(expr.a)); break;
	case ExprType::Ln:
	{
		double arg = rec(expr.a);
		result = (arg <= 0.0) ? -std::numeric_limits<double>::infinity() : std::log(arg);
	}
	break;
	case ExprType::Log10:
	{
		double arg = rec(expr.a);
		result = (arg <= 0.0) ? -std::numeric_limits<double>::infinity() : std::log10(arg);
	}
	break;
	case ExprType::Exp:
		result = std::exp(rec(expr.a)); break;
	case ExprType::Sin:
		result = std::sin(rec(expr.a)); break;
	case ExprType::Cos:
		result = std::cos(rec(expr.a)); break;
	case ExprType::Tan:
		result = std::tan(rec(expr.a)); break;
	case ExprType::ArcSin:
		result = std::asin(rec(expr.a)); break;
	case ExprType::ArcCos:
		result = std::acos(rec(expr.a)); break;
	case ExprType::ArcTan:
		result = std::atan(rec(expr.a)); break;

	// Branch keeps lazy semantics: only the taken side is evaluated (and memoized).
	case ExprType::Branch:
		result = (rec(expr.cond) > 0) ? rec(expr.a) : rec(expr.b);
		break;

	default:
		std::cout << "symx error: Unhandled ExprType in Expressions.eval()" << std::endl;
		exit(-1);
	}

	done[expr_id] = 1;
	memo[expr_id] = result;
	return result;
}
int64_t symx::Expressions::get_n_bytes() const
{
    return sizeof(Expr) * this->expressions.size();
}
std::string symx::Expressions::get_checksum(const std::string &in) const
{
	// Initialize hasher
	picosha2::hash256_one_by_one hasher;
    hasher.init();

	// Process input string
	hasher.process(in.begin(), in.end());

	// Process all expressions (flat)
	hasher.process(
		reinterpret_cast<const char*>(this->expressions.data()),
		reinterpret_cast<const char*>(this->expressions.data() + this->expressions.size())
	);

	// Finalize the hash
    hasher.finish();
    std::vector<unsigned char> out_hash(picosha2::k_digest_size);
    hasher.get_hash_bytes(out_hash.begin(), out_hash.end());
    std::string hex_str = picosha2::bytes_to_hex_string(out_hash.begin(), out_hash.end());

    return hex_str;
}
int32_t Expressions::_find_or_add(const Expr &expr)
{
	/*
		Based on the Common Subexpression Elimination strategy set, expressions are found (because they already existed) or added to the list.
	*/

	// No CSE
	if (get_cse_mode() == CSE::No) {
		int32_t expr_id = (int32_t)this->expressions.size();
		this->expressions.push_back(expr);
		return expr_id;
	}

	// Regardless of the CSE mode, check if the expression we are looking for is exactly the next one from the last found
	if (this->expressions.size() > this->last_cse_idx_checked + 1 && this->_are_same(expr, this->expressions[this->last_cse_idx_checked + 1])) {
		this->last_cse_idx_checked += 1;
		return this->last_cse_idx_checked;
	}

	// Pick a CSE strategy
	if (get_cse_mode() == CSE::Safe) {

		// Look for the expression in the hash map
		const std::array<int32_t, 4> key = { (int32_t)expr.type, expr.a, expr.b, expr.cond };
		auto it = this->cse_map.find(key);
		if (it == this->cse_map.end()) {
			// New expression
			int32_t expr_id = (int32_t)this->expressions.size();
			this->cse_map[key] = expr_id;
			this->expressions.push_back(expr);
			return expr_id;
		}
		else {
			// Already exists
			const int existing_idx = it->second;
			this->last_cse_idx_checked = existing_idx;
			return existing_idx;
		}
	}
	else {
		std::cout << "symx error: Unhandled CSE mode in Expressions._find_or_add()" << std::endl;
		exit(-1);
	}
}

bool Expressions::_are_same(const Expr& a, const Expr& b)
{
	return a.type == b.type && a.a == b.a && a.b == b.b && a.cond == b.cond;
}
