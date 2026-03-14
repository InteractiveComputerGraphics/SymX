# Symbol-Data Overview
Building compiled functions from differentiated expressions already covers a lot of ground, but in practice you rarely need just one function on one input.
SymX provides a higher-level entry point for loop evaluation over many data instances:
- `MappedWorkspace`: Creates SymX symbols bound to user data, automating value passing to compiled functions.
- `CompiledInLoop`: Evaluates expressions in stencil-based loops. Handles parallelism, vectorization, different compilation types, and more.

Together, these let you evaluate arbitrary expressions over entire discretizations without writing the evaluation loop yourself.
