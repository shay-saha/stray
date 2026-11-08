cmake_minimum_required(VERSION 3.24)

project(WildlifeAtlas VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

include(FetchContent)
find_package(OpenGL REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)

# Build only the dependencies used by the application.
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "Build GLFW Wayland support")

FetchContent_Declare(glfw
    URL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw-3.4.tar.gz"
    URL_HASH SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_Declare(imgui
    URL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui-v1.92.5.tar.gz"
    URL_HASH SHA256=0eb50fe9aeba1a51f96b5843c7f630a32ed2e9362d693c61b87e4fa870cf826d
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(glfw imgui)

add_library(imgui_native STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)
target_include_directories(imgui_native SYSTEM PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
)
target_link_libraries(imgui_native PUBLIC glfw PRIVATE ${CMAKE_DL_LIBS})

add_executable(wildlife_atlas
    src/main.cpp
    src/app.cpp
    src/earth_texture.cpp
    src/earth_tiles.cpp
    src/globe.cpp
    src/habitat_view.cpp
    src/population_view.cpp
    src/trail_view.cpp
    src/simulation.cpp
    src/forecast.cpp
)
target_compile_features(wildlife_atlas PRIVATE cxx_std_20)
set_target_properties(wildlife_atlas PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(wildlife_atlas PRIVATE imgui_native OpenGL::GL CURL::libcurl Threads::Threads nlohmann_json::nlohmann_json)
target_include_directories(wildlife_atlas SYSTEM PRIVATE third_party)

set(atlas_texture "${CMAKE_CURRENT_SOURCE_DIR}/assets/earth-topography.jpg")
set(atlas_ocean "${CMAKE_CURRENT_SOURCE_DIR}/assets/ocean-relief.png")
set(atlas_elevation "${CMAKE_CURRENT_SOURCE_DIR}/assets/earth-elevation.png")
set(atlas_cities "${CMAKE_CURRENT_SOURCE_DIR}/assets/city-lights.jpg")
set(atlas_detail "${CMAKE_CURRENT_SOURCE_DIR}/assets/earth-detail")
set(atlas_font "${CMAKE_CURRENT_SOURCE_DIR}/assets/Urbanist-Variable.ttf")
set(atlas_font_license "${CMAKE_CURRENT_SOURCE_DIR}/assets/Urbanist-OFL.txt")
add_custom_target(wildlife_atlas_data ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${atlas_texture}" "$<TARGET_FILE_DIR:wildlife_atlas>/earth-topography.jpg"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${atlas_ocean}" "$<TARGET_FILE_DIR:wildlife_atlas>/ocean-relief.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${atlas_elevation}" "$<TARGET_FILE_DIR:wildlife_atlas>/earth-elevation.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${atlas_cities}" "$<TARGET_FILE_DIR:wildlife_atlas>/city-lights.jpg"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${atlas_detail}" "$<TARGET_FILE_DIR:wildlife_atlas>/earth-detail"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${atlas_font}" "${atlas_font_license}" "$<TARGET_FILE_DIR:wildlife_atlas>"
    DEPENDS wildlife_atlas
    VERBATIM
)

if(MSVC)
    target_compile_options(wildlife_atlas PRIVATE /W4 /utf-8)
else()
    target_compile_options(wildlife_atlas PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
    add_executable(simulation_forecast_test tests/simulation_forecast.cpp src/simulation.cpp)
    target_include_directories(simulation_forecast_test PRIVATE src)
    target_compile_features(simulation_forecast_test PRIVATE cxx_std_20)
    add_test(NAME simulation_forecast COMMAND simulation_forecast_test)
endif()

if(APPLE)
    configure_file(cmake/Info.plist.in "${CMAKE_CURRENT_BINARY_DIR}/WildlifeAtlas-Info.plist" @ONLY)
    set(atlas_bundle "$<TARGET_FILE_DIR:wildlife_atlas>/Wildlife Atlas.app/Contents")
    add_custom_target(wildlife_atlas_app ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${atlas_bundle}/MacOS"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${atlas_bundle}/Resources"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_BINARY_DIR}/WildlifeAtlas-Info.plist" "${atlas_bundle}/Info.plist"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:wildlife_atlas>" "${atlas_bundle}/MacOS/wildlife_atlas"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${atlas_texture}" "${atlas_bundle}/Resources/earth-topography.jpg"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${atlas_ocean}" "${atlas_bundle}/Resources/ocean-relief.png"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${atlas_elevation}" "${atlas_bundle}/Resources/earth-elevation.png"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${atlas_cities}" "${atlas_bundle}/Resources/city-lights.jpg"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${atlas_detail}" "${atlas_bundle}/Resources/earth-detail"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${atlas_font}" "${atlas_font_license}" "${atlas_bundle}/Resources"
        DEPENDS wildlife_atlas
        VERBATIM
    )
endif()
