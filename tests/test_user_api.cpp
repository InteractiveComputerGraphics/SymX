#include "catch.hpp"
#include <symx>
#include "utils.h"

using namespace symx;

TEST_CASE("User-friendly compiler API", "[user_api]")
{
    SECTION("Easy compiler checking")
    {
        // Simple check if compiler is working
        bool working = is_compiler_working();
        REQUIRE(working == true);  // Should be true since tests are passing
        
        std::cout << "Compiler working: " << (working ? "YES" : "NO") << std::endl;
    }
    
    SECTION("Get recommended compiler")
    {
        std::string recommended = get_recommended_compiler();
        REQUIRE(!recommended.empty());
        
        std::cout << "Recommended compiler: " << recommended << std::endl;
    }
    
    SECTION("Test specific compiler")
    {
        // Test the current compiler
        CompilerInfo info = test_compiler(symx::resolve_compiler_path());
        REQUIRE(info.is_available == true);
        
        // Test a bad compiler
        CompilerInfo bad_info = test_compiler("nonexistent_compiler");
        REQUIRE(bad_info.is_available == false);
        
        std::cout << "Current compiler test: " << (info.is_available ? "PASS" : "FAIL") << std::endl;
        std::cout << "Bad compiler test: " << (bad_info.is_available ? "PASS (unexpected)" : "FAIL (expected)") << std::endl;
    }
    
    SECTION("Comprehensive check (visual test)")
    {
        std::cout << "\n=== User-Friendly Compiler Check ===" << std::endl;
        check_compiler_setup();
        std::cout << "===================================" << std::endl;
    }
}
