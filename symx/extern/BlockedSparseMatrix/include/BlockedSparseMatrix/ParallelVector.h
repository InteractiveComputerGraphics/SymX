#pragma once
#include <stdexcept>
#include <vector>

namespace bsm
{
	template<typename VECTOR>
	class ParallelVector
	{
		std::vector<VECTOR> vectors;
		bool reduced = false;
	public:
		// Start a new accumulation. local(thread_id) may be written until reduce().
		void begin_accumulation(const int n_threads, const int size);
		VECTOR& local(const int thread_id);
		// Sum thread-local vectors into one result before result() is read.
		VECTOR& reduce();
		const VECTOR& result() const;
		int get_n_threads() const;
	};

	template<typename T>
	inline void ParallelVector<T>::begin_accumulation(const int n_threads, const int size)
	{
		if (n_threads <= 0 || size < 0) {
			throw std::invalid_argument("ParallelVector requires a positive thread count and non-negative vector size.");
		}
		this->vectors.resize(n_threads);
		for (auto& vector : this->vectors) {
			vector.resize(size);
			for (int i = 0; i < size; i++) {
				vector[i] = 0.0;
			}
		}
		this->reduced = false;
	}
	template<typename VECTOR>
	inline VECTOR& ParallelVector<VECTOR>::reduce()
	{
		if (this->vectors.empty()) {
			throw std::logic_error("ParallelVector::reduce() requires begin_accumulation() first.");
		}
		if (this->reduced) {
			throw std::logic_error("ParallelVector::reduce() may only be called once per accumulation.");
		}
		const int n = (int)this->vectors[0].size();
		const int n_vectors = (int)this->vectors.size();
		#pragma omp parallel for schedule(static) num_threads(n_vectors)
		for (int i = 0; i < n; i++) {
			for (int j = 1; j < n_vectors; j++) {
				this->vectors[0][i] += this->vectors[j][i];
			}
		}
		this->reduced = true;
		return this->vectors[0];
	}
	template<typename VECTOR>
	inline const VECTOR& ParallelVector<VECTOR>::result() const
	{
		if (!this->reduced) {
			throw std::logic_error("ParallelVector::result() requires reduce() first.");
		}
		return this->vectors[0];
	}
	template<typename VECTOR>
	inline VECTOR& ParallelVector<VECTOR>::local(const int thread_id)
	{
		if (this->reduced) {
			throw std::logic_error("ParallelVector::local() cannot be used after reduce().");
		}
		if (thread_id < 0 || thread_id >= this->get_n_threads()) {
			throw std::out_of_range("ParallelVector thread index is out of range.");
		}
		return this->vectors[thread_id];
	}
    template <typename VECTOR>
    inline int ParallelVector<VECTOR>::get_n_threads() const
    {
        return (int)this->vectors.size();
    }
}
