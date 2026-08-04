#pragma once
#include <stdexcept>
#include <vector>

namespace bsm
{
	template<typename FLOAT>
	class ParallelNumber
	{
		const static int PADDING_BYTES = 64;
		std::vector<FLOAT> values;
		int padding = 1;
		bool reduced = false;
	public:
		// Start a new accumulation. local(thread_id) may be written until reduce().
		void begin_accumulation(const int n_threads);
		FLOAT& local(const int thread_id);
		// Sum every thread-local value into a single result. Must be called once
		// before result(); the returned reference is the reduced result.
		FLOAT& reduce();
		const FLOAT& result() const;
		// Read a non-mutating sum while accumulation is still in progress.
		FLOAT sum_locals() const;
		int get_n_threads() const;
	};

	// ============================================================================
	template<typename FLOAT>
	inline void ParallelNumber<FLOAT>::begin_accumulation(const int n_threads)
	{
		if (n_threads <= 0) {
			throw std::invalid_argument("ParallelNumber requires at least one thread.");
		}
		this->padding = PADDING_BYTES / sizeof(FLOAT);
		if (this->padding < 1) this->padding = 1;
		this->values.resize(n_threads*this->padding);
		for (int i = 0; i < n_threads; i++) {
			this->values[i * this->padding] = 0.0;
		}
		this->reduced = false;
	}
	template<typename FLOAT>
	inline FLOAT& ParallelNumber<FLOAT>::local(const int thread_id)
	{
		if (this->reduced) {
			throw std::logic_error("ParallelNumber::local() cannot be used after reduce().");
		}
		if (thread_id < 0 || thread_id >= this->get_n_threads()) {
			throw std::out_of_range("ParallelNumber thread index is out of range.");
		}
		return this->values[thread_id*this->padding];
	}
	template<typename FLOAT>
	inline FLOAT& ParallelNumber<FLOAT>::reduce()
	{
		if (this->reduced) {
			throw std::logic_error("ParallelNumber::reduce() may only be called once per accumulation.");
		}
		const int n = (int)this->values.size()/this->padding;
		for (int i = 1; i < n; i++) {
			this->values[0] += this->values[i * this->padding];
			this->values[i * this->padding] = 0.0;
		}
		this->reduced = true;
		return this->values[0];
	}
	template<typename FLOAT>
	inline const FLOAT& ParallelNumber<FLOAT>::result() const
	{
		if (!this->reduced) {
			throw std::logic_error("ParallelNumber::result() requires reduce() first.");
		}
		return this->values[0];
	}
	template<typename FLOAT>
	inline FLOAT ParallelNumber<FLOAT>::sum_locals() const
	{
		FLOAT sum = 0.0;
		for (int i = 0; i < this->get_n_threads(); ++i) {
			sum += this->values[i * this->padding];
		}
		return sum;
	}
    template <typename FLOAT>
    inline int ParallelNumber<FLOAT>::get_n_threads() const
    {
        return (int)this->values.size() / this->padding;
    }
}
