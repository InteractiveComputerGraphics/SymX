#include "catch.hpp"
#include <symx>
#include "utils.h"

using namespace symx;

TEST_CASE("User-friendly compiler API", "[user_api]")
{
    SECTION("Easy compiler checking")
    {
        // Simple check if compiler is working
        CompilerInfo info = get_compiler_info(resolve_compiler_path());
        REQUIRE(info.is_available);
        REQUIRE(info.could_compile);
        REQUIRE(info.could_load);
        bool working = info.verified_result;
        REQUIRE(working == true);  // Should be true since tests are passing
        
        std::cout << "Compiler working: " << (working ? "YES" : "NO") << std::endl;
    }
    
    SECTION("Resolve compiler")
    {
        std::string recommended = resolve_compiler_path();
        REQUIRE(!recommended.empty());
        
        std::cout << "Recommended compiler: " << recommended << std::endl;
    }
    
    SECTION("Test specific compiler")
    {
        // Test the current compiler
        CompilerInfo info = get_compiler_info(symx::resolve_compiler_path());
        REQUIRE(info.is_available == true);
        
        // Test a bad compiler
        CompilerInfo bad_info = get_compiler_info("nonexistent_compiler");
        REQUIRE(bad_info.is_available == false);
        
        std::cout << "Current compiler test: " << (info.is_available ? "PASS" : "FAIL") << std::endl;
        std::cout << "Bad compiler test: " << (bad_info.is_available ? "PASS (unexpected)" : "FAIL (expected)") << std::endl;
    }
    
    SECTION("Comprehensive check (visual test)")
    {
        std::cout << "\n=== User-Friendly Compiler Check ===" << std::endl;
        print_compiler_diagnostics();
        std::cout << "===================================" << std::endl;
    }
}
