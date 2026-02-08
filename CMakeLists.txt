cmake_minimum_required(VERSION 3.22)

project(variational-cluster-toolkit
    VERSION 0.1.0
    DESCRIPTION "Modern C++20 toolkit for variational Bayesian clustering"
    LANGUAGES CXX)

# ─── compiler rules ───────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CPM_DOWNLOAD_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")
if(NOT EXISTS "${CPM_DOWNLOAD_LOCATION}")
    message(STATUS "Downloading CPM.cmake …")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
        "${CPM_DOWNLOAD_LOCATION}" TLS_VERIFY ON)
endif()
include("${CPM_DOWNLOAD_LOCATION}")

# ─── Project‑wide options -----------------------------------------------------
option(VCTK_BUILD_TESTS "Build unit tests" ON)
option(VCTK_BUILD_EXAMPLES "Build examples" OFF)
option(VCTK_BUILD_BENCHMARKS "Build benchmark executables" OFF)
option(VCTK_BUILD_DOCS "Build Doxygen/Sphinx docs" OFF)
option(VCTK_USE_SYSTEM_EIGEN "Prefer system-installed Eigen3 before CPM fetch" ON)
option(VCTK_USE_OPENMP "Enable OpenMP parallel kernels" ON)
option(VCTK_FETCH_OPENMP "Fetch LLVM OpenMP with CPM if system OpenMP is unavailable" ON)
set(VCTK_OPENMP_LLVM_TAG "llvmorg-21.1.8" CACHE STRING "LLVM tag to use when fetching OpenMP runtime")

if(VCTK_USE_SYSTEM_EIGEN)
    find_package(Eigen3 3.4 CONFIG QUIET NO_MODULE)
endif()

if(NOT TARGET Eigen3::Eigen)
    if(DEFINED Eigen3_SOURCE_DIR AND EXISTS "${Eigen3_SOURCE_DIR}/Eigen/Core")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        target_include_directories(Eigen3::Eigen INTERFACE "${Eigen3_SOURCE_DIR}")
    endif()
endif()

if(NOT TARGET Eigen3::Eigen)
    CPMAddPackage(
        NAME Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        DOWNLOAD_ONLY TRUE
        OPTIONS "EIGEN_BUILD_DOC OFF" "EIGEN_BUILD_PKGCONFIG OFF"
    )
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${Eigen3_SOURCE_DIR}")
endif()

file(GLOB_RECURSE VCTK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(vctk STATIC ${VCTK_SOURCES})
add_library(vctk::vctk ALIAS vctk)

target_include_directories(vctk
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(vctk
    PUBLIC
    Eigen3::Eigen)

target_compile_features(vctk PUBLIC cxx_std_20)

if(VCTK_USE_OPENMP)
    if(APPLE)
        # AppleClang often needs explicit libomp hints.
        foreach(_omp_prefix "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
            if(EXISTS "${_omp_prefix}/include/omp.h")
                if(NOT DEFINED OpenMP_ROOT)
                    set(OpenMP_ROOT "${_omp_prefix}" CACHE PATH "OpenMP root path")
                endif()
                list(APPEND CMAKE_PREFIX_PATH "${_omp_prefix}")
                break()
            endif()
        endforeach()
    endif()

    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "VCTK: OpenMP enabled")
        target_link_libraries(vctk PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(vctk PUBLIC VCTK_HAS_OPENMP=1)
    elseif(VCTK_FETCH_OPENMP)
        message(STATUS "VCTK: OpenMP not found; fetching LLVM OpenMP via CPM (${VCTK_OPENMP_LLVM_TAG})")
        CPMAddPackage(
            NAME llvm_openmp
            GIT_REPOSITORY https://github.com/llvm/llvm-project.git
            GIT_TAG ${VCTK_OPENMP_LLVM_TAG}
            SOURCE_SUBDIR openmp
            OPTIONS
                "OPENMP_ENABLE_LIBOMPTARGET OFF"
                "OPENMP_BUILD_TOOLS OFF"
                "OPENMP_ENABLE_WERROR OFF"
        )
        if(TARGET omp)
            if(NOT TARGET OpenMP::OpenMP_CXX)
                add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
                target_link_libraries(OpenMP::OpenMP_CXX INTERFACE omp)
                target_compile_options(OpenMP::OpenMP_CXX INTERFACE
                    "$<$<CXX_COMPILER_ID:AppleClang>:-Xclang;-fopenmp>"
                    "$<$<NOT:$<CXX_COMPILER_ID:AppleClang>>:-fopenmp>"
                )
            endif()
            message(STATUS "VCTK: OpenMP enabled (CPM-fetched LLVM runtime)")
            target_link_libraries(vctk PUBLIC OpenMP::OpenMP_CXX)
            target_compile_definitions(vctk PUBLIC VCTK_HAS_OPENMP=1)
        else()
            message(STATUS "VCTK: CPM OpenMP fetch succeeded but target 'omp' was not found; building without OpenMP")
        endif()
    else()
        message(STATUS "VCTK: OpenMP requested but not found; building without it")
    endif()
endif()

if(VCTK_BUILD_TESTS)
    enable_testing()
    add_executable(vctk_comutils_test test/test_comutils.cpp)
    target_link_libraries(vctk_comutils_test PRIVATE vctk::vctk Eigen3::Eigen)
    add_test(NAME comutils_sanity COMMAND vctk_comutils_test)
    add_executable(vctk_probutils_test test/test_probutils.cpp)
    target_link_libraries(vctk_probutils_test PRIVATE vctk::vctk Eigen3::Eigen)
    add_test(NAME probutils_sanity COMMAND vctk_probutils_test)
    add_executable(vctk_distributions_test test/test_distributions.cpp)
    target_link_libraries(vctk_distributions_test PRIVATE vctk::vctk Eigen3::Eigen)
    add_test(NAME distributions_sanity COMMAND vctk_distributions_test)
    add_executable(vctk_algorithms_test test/test_vctk.cpp)
    target_link_libraries(vctk_algorithms_test PRIVATE vctk::vctk Eigen3::Eigen)
    add_test(NAME algorithms_sanity COMMAND vctk_algorithms_test)
    add_executable(vctk_merge_test test/test_merge.cpp)
    target_link_libraries(vctk_merge_test PRIVATE vctk::vctk Eigen3::Eigen)
    add_test(NAME merge_sanity COMMAND vctk_merge_test)
    add_executable(vctk_c_api_test test/test_vctk_c.cpp)
    target_link_libraries(vctk_c_api_test PRIVATE vctk::vctk Eigen3::Eigen)
    add_test(NAME c_api_sanity COMMAND vctk_c_api_test)
endif()

if(VCTK_BUILD_EXAMPLES)
    add_executable(vctk_vdp_example examples/vdp_example.cpp)
    target_link_libraries(vctk_vdp_example PRIVATE vctk::vctk Eigen3::Eigen)
    add_executable(vctk_merge_proof_example examples/merge_proof_example.cpp)
    target_link_libraries(vctk_merge_proof_example PRIVATE vctk::vctk Eigen3::Eigen)
endif()

if(VCTK_BUILD_BENCHMARKS)
    add_executable(vctk_bench_vdp bench/vdp_benchmark.cpp)
    target_link_libraries(vctk_bench_vdp PRIVATE vctk::vctk Eigen3::Eigen)
endif()

# ─── Installation & export ----------------------------------------------------
include(GNUInstallDirs)

install(TARGETS vctk
    EXPORT vctkTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT vctkTargets
    FILE vctkConfig.cmake
    NAMESPACE vctk::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vctk)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vctkConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vctkConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vctk)
