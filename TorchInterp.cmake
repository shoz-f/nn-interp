set(LIBTORCH_ROOTDIR ${THIRD_PARTY}/libtorch)

if("${NN_TARGET}" STREQUAL "windows-x86_64")
	if(${NN_CONFIG} STREQUAL "gpu")
		set(URL_LIBTORCH "https://download.pytorch.org/libtorch/cu117/libtorch-win-shared-with-deps-2.0.1%2Bcu117.zip")
	else()
		set(URL_LIBTORCH "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.0.1%2Bcpu.zip")
	endif()
	string(APPEND CMAKE_CXX_FLAGS " /wd4624 /wd4819 /wd4067 /wd4251 /wd4244")
elseif("${NN_TARGET}" STREQUAL "linux-x86_64")
	if(${NN_CONFIG} STREQUAL "gpu")
		set(URL_LIBTORCH "https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip")
	else()
		set(URL_LIBTORCH "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip")
	endif()
	set(CMAKE_INSTALL_RPATH "\$ORIGIN")
else()
	message(FATAL_ERROR "unknown target")
endif()

# add Libtorch
if(NOT EXISTS ${LIBTORCH_ROOTDIR})
	message("** Download LibTorch.")
	FetchContent_Declare(Torch
		URL ${URL_LIBTORCH}
		SOURCE_DIR ${LIBTORCH_ROOTDIR}
		)
	FetchContent_MakeAvailable(Torch)
endif()

find_package(Torch REQUIRED
	PATHS ${LIBTORCH_ROOTDIR}
	)

set(X_INTERP
	src/torch/torch_interp.cpp
	)
include_directories(
	${TORCH_INCLUDE_DIRS}
	)
link_libraries(
	${TORCH_LIBRARIES}
	)

if(MSVC)
	file(GLOB NNFW_DLLS "${LIBTORCH_ROOTDIR}/lib/*.dll")
elseif(UNIX)
	file(GLOB NNFW_DLLS "${LIBTORCH_ROOTDIR}/lib/*.so*")
endif()
