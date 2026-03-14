cmake_minimum_required(VERSION 3.15)

set(SOURCE_FILES
    examples_main.cpp
    hello_world.cpp
    NeoHookean.cpp
    triangle_mesh_area.cpp
    rubber_extrusion.cpp
    xpbd_cloth.cpp
    dynamic_elasticity_with_contact.cpp
	)

add_executable(examples ${SOURCE_FILES})
target_link_libraries(examples symx)
target_include_directories(examples PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extern/vtkio/include)

# Define directory paths
target_compile_definitions(examples PRIVATE
    SYMX_EXAMPLES_RES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res"
    SYMX_EXAMPLES_OUTPUT_DIR="${CMAKE_BINARY_DIR}/output"
)

