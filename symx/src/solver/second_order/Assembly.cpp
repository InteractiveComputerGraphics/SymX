#include "Assembly.h"

void symx::Assembly::start(const std::vector<int>& dof_set_offsets, int n_threads, bool hess, bool grad)
{
	this->dof_set_offsets = dof_set_offsets;
	this->ndofs = dof_set_offsets.back();
	this->n_threads = n_threads;
	if (hess) {
		this->element_hessians->start(this->n_threads);
	}
	if (grad) {
		this->grad.begin_accumulation(this->n_threads, this->ndofs);
	}
	this->E.begin_accumulation(this->n_threads);
}

void symx::Assembly::stop(bool hess, bool grad)
{
	if (hess) {
		this->element_hessians->stop();
	}
	if (grad) {
		this->grad.reduce();
	}
	this->E.reduce();
}
