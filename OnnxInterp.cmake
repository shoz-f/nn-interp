set(ONNXRUNTIME_ROOTDIR ${THIRD_PARTY}/onnxruntime)


if(MSVC)
        set(URL_ONNXRUNTIME https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-win-x64-1.12.0.zip)
        string(APPEND CMAKE_CXX_FLAGS " /W4")
        set(ONNXRUNTIME_LIBRARIES
    	        ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib
    	        ${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime_providers_shared.lib
    	)
elseif(UNIX AND NOT APPLE)
        set(URL_ONNXRUNTIME https://github.com/microsoft/onnxruntime/releases/download/v1.12.0/onnxruntime-linux-x64-1.12.0.tgz)
        string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
        string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
	set(ONNXRUNTIME_LIBRARIES
                ${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so
                )
	set(CMAKE_INSTALL_RPATH "\$ORIGIN")
else()
	message(FATAL_ERROR "unknown target")
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
if(onnxruntime_USE_CUDA)
        add_definitions(-DUSE_CUDA)
endif()

# add ONNX Runtime
if(NOT EXISTS ${ONNXRUNTIME_ROOTDIR})
        message("** Download onnxruntime.")
        FetchContent_Declare(onnxruntime
        	URL ${URL_ONNXRUNTIME}
        	SOURCE_DIR ${ONNXRUNTIME_ROOTDIR}
        	)
        FetchContent_MakeAvailable(onnxruntime)
endif()

set(X_INTERP
	src/onnx/onnx_interp.cpp
	)
include_directories(
	${ONNXRUNTIME_ROOTDIR}/include
	${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session
	)
link_libraries(
	${ONNXRUNTIME_LIBRARIES}
	)

if(MSVC)
	file(GLOB NNFW_DLLS "${ONNXRUNTIME_ROOTDIR}/lib/*.dll")
	message("${NNFW_DLLS}")
elseif(UNIX AND NOT APPLE)
	file(GLOB NNFW_DLLS "${ONNXRUNTIME_ROOTDIR}/lib/*.so*")
	message("${NNFW_DLLS}")
endif()
