set(LIBTORCH_ROOTDIR ${THIRD_PARTY}/libtorch)

if(MSVC)
	set(URL_LIBTORCH "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.12.1%2Bcpu.zip")
	string(APPEND CMAKE_CXX_FLAGS " /wd4624 /wd4819 /wd4067 /wd4251 /wd4244")
	set(NNFW_DLLS_DIR "${LIBTORCH_ROOTDIR}/lib/*.dll")
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
include_directories(
	${TORCH_INCLUDE_DIRS}
	)
link_libraries(
	${TORCH_LIBRARIES}
	)

if(MSVC)
	file(GLOB NNFW_DLLS "${THIRD_PARTY}/libtorch/lib/*.dll")
endif()
