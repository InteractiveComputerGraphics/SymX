#include "CompiledCondition.h"

symx::CompiledCondition::CompiledCondition(const Potential &potential, const std::string& checksum, const std::string &compilation_directory, DeferredParallelTasks &tasks)
    : mws(potential.get_mws())
{
    this->_has_condition = potential.has_conditional();

    if (this->_has_condition) {
        // Try to load
        const std::string name = potential.get_name();
        this->cached = this->C.load_if_cached(this->mws, name + "_C", compilation_directory, checksum);

        // If not cached, compile
        if (!this->cached) {
            tasks.add([this, cond = potential.get_condition(), name, compilation_directory, checksum]() {
                this->C.compile(this->mws, { cond }, name + "_C", compilation_directory, checksum);
            });
        }
    } else {
        // No condition to compile – trivially consider it cached
        this->cached = true;
    }
}

bool symx::CompiledCondition::was_cached() const
{
    return this->cached;
}

bool symx::CompiledCondition::has_condition() const
{
    return this->_has_condition;
}

std::vector<uint8_t> &symx::CompiledCondition::get_element_condition()
{
    return this->has_element_positive_condition;
}

void symx::CompiledCondition::evaluate(int32_t n_threads)
{
    // Evaluate condition
    if (this->_has_condition) {
        const int n_elements = this->mws->conn.n_elements();
        this->has_element_positive_condition.resize(n_elements);
        this->C.run(n_threads,
            [this](const View<double> cond_val, int32_t element_idx, int32_t thread_id, const View<int32_t> conn)
            {
                this->has_element_positive_condition[element_idx] = static_cast<uint8_t>(cond_val[0] > 0.0);
            }
        );
    } 

    // Leave empty -> no condition
    else {
        this->has_element_positive_condition.clear();
        return;
    }
}
