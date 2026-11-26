add_hal_lib(cuda LINK_LIBRARIES cuxtra)
add_shim_lib(cuda
    SOFTLINKS_LINUX libcuda.so.1 libcuda.so
    TARGET_NAME_WINDOWS "nvcuda"
)
add_platform_test()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    # DEF-10: privileged trap-pulse proxy so Level-3 runs without root (no CUDA dependency).
    add_executable(xsched-bar0d "${CMAKE_CURRENT_LIST_DIR}/bar0d/xsched_bar0d.cpp")
    target_include_directories(xsched-bar0d PRIVATE ${CMAKE_CURRENT_LIST_DIR}/hal/include)
    target_compile_options(xsched-bar0d PRIVATE -Wall -Wextra -O2)
    set_target_properties(xsched-bar0d PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    install(TARGETS xsched-bar0d RUNTIME DESTINATION bin)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "120")
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        if(EXISTS "$ENV{HOME}/.local/cuda-12.9/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "$ENV{HOME}/.local/cuda-12.9/bin/nvcc")
        elseif(EXISTS "/home/wy/cuda-12.9/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "/home/wy/cuda-12.9/bin/nvcc")
        endif()
    endif()
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-13")
        set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
    endif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    function(add_sm120_l3_bin target_name src_rel_path)
        if(EXISTS "${CMAKE_SOURCE_DIR}/${src_rel_path}")
            add_executable(${target_name} "${CMAKE_SOURCE_DIR}/${src_rel_path}")
            target_include_directories(${target_name} PRIVATE
                ${XSCHED_INCLUDE_DIR}
                ${CMAKE_CURRENT_LIST_DIR}/hal/include
                ${CMAKE_SOURCE_DIR}/preempt/include
                ${CMAKE_SOURCE_DIR}/utils/include
                ${CMAKE_SOURCE_DIR}/protocol/include
                ${CMAKE_SOURCE_DIR}/3rdparty/cuxtra/include
            )
            target_link_options(${target_name} PRIVATE -Wl,--no-as-needed)
            target_link_libraries(${target_name} PRIVATE
                halcuda
                preempt
                utils
                cuda
                cudart
                dl
                pthread
            )
            set_target_properties(${target_name} PROPERTIES
                CUDA_ARCHITECTURES "120"
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                BUILD_RPATH "${CMAKE_BINARY_DIR}/platforms/cuda;${CMAKE_BINARY_DIR}/preempt;${CMAKE_BINARY_DIR}/utils;${CMAKE_BINARY_DIR}/protocol"
                INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
            )
            install(TARGETS ${target_name} RUNTIME DESTINATION bin)
        endif()
    endfunction()

    add_sm120_l3_bin(sm120_cave_probe "test/sm120_v2/cave_probe.cu")
    add_sm120_l3_bin(sm120_v2_e2e_suite "test/sm120_v2/sm120_v2_e2e_suite.cu")
    add_sm120_l3_bin(sm120_v2_bench "test/sm120_v2/sm120_v2_bench.cu")
    add_sm120_l3_bin(sm120_l3_resumable_bench "test/sm120_l3_resumable/sm120_l3_resumable_bench.cu")

    if(EXISTS "${CMAKE_SOURCE_DIR}/test/sm120_v2/test_sass_disasm.cpp")
        add_executable(sm120_sass_disasm "${CMAKE_SOURCE_DIR}/test/sm120_v2/test_sass_disasm.cpp")
        target_include_directories(sm120_sass_disasm PRIVATE
            ${XSCHED_INCLUDE_DIR}
            ${CMAKE_CURRENT_LIST_DIR}/hal/include
        )
        set_target_properties(sm120_sass_disasm PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    endif()
endif()

