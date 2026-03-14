#include "examples_main.h"

#include <symx>
using namespace symx;


void hello_world()
{
    std::cout << "\n=================== hello_world() ===================" << std::endl;

    // Create a symbol environment
    Workspace sws;

    // Declare symbols
    Scalar x = sws.make_scalar();

    // Do math
    Scalar dsinx_dx = diff(sin(x), x);

    // Compile
    Compiled<double> compiled({ dsinx_dx }, "hello_world", "../codegen");

    // Set numeric data before evaluation
    compiled.set(x, 0.0);

    // Evaluate
    View<double> result = compiled.run();
	
    // Print
	std::cout << "dsinx_dx(0.0) = " << result[0] << std::endl;  // = 1.0
}


