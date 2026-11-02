#pragma once
#include "Potential.h"
#include "../compile/CompiledInLoop.h"
#include "DeferredParallelTasks.h"

namespace symx
{
    class CompiledCondition
    {
	private:
        /* Fields */
        const spMWS<double> mws = nullptr;
        CompiledInLoop<double> C;
        std::vector<uint8_t> has_element_positive_condition;
        bool _has_condition = false;
        bool cached = false;

    public:
        /* Methods */
        CompiledCondition(const Potential& potential, const std::string& checksum, const std::string &compilation_directory, DeferredParallelTasks& tasks);
        CompiledCondition(const Potential& potential, const std::string& checksum, const std::string &compilation_directory, DeferredParallelTasks& tasks, const std::string& artifact_suffix);
        bool was_cached() const;
        bool has_condition() const;
        std::vector<uint8_t>& get_element_condition();
        void evaluate(int32_t n_threads);
    };
    using spCompiledCondition = std::shared_ptr<CompiledCondition>;
} // namespace symx
