#include "catch.hpp"
#include <symx>
#include "utils.h"

using namespace symx;

TEST_CASE("Compiler diagnostics", "[diagnostics]")
{
    SECTION("Valid compiler diagnostics")
    {
        // Test current compiler validation
        symx::CompilerInfo info = symx::get_compiler_info(symx::resolve_compiler_path());
        
        // Should be available since tests are working
        REQUIRE(info.is_available == true);
        REQUIRE(!info.path.empty());
        REQUIRE(!info.version.empty());
        REQUIRE(!info.architecture.empty());
        
        std::cout << "Compiler validation successful:" << std::endl;
        std::cout << "  Path: " << info.path << std::endl;
        std::cout << "  Version: " << info.version << std::endl;
        std::cout << "  Architecture: " << info.architecture << std::endl;
        std::cout << "  AVX Support: " << (info.supports_avx ? "YES" : "NO") << std::endl;
    }
    
    SECTION("Invalid compiler detection")
    {
        // Test with a non-existent compiler
        symx::CompilerInfo bad_info = symx::get_compiler_info("/nonexistent/compiler");
        
        REQUIRE(bad_info.is_available == false);
        REQUIRE(!bad_info.error_message.empty());
        
        std::cout << "Invalid compiler correctly detected:" << std::endl;
        std::cout << "  Error: " << bad_info.error_message << std::endl;
    }
    
    SECTION("Print diagnostics (visual test)")
    {
        // This will print detailed diagnostics for manual verification
        std::cout << "\n--- Full Diagnostics Output ---" << std::endl;
        symx::validate_auto_search_compiler_paths();
        std::cout << "--- End Diagnostics Output ---\n" << std::endl;
    }
}
