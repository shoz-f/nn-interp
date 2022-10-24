set(TFLITE_ROOTDIR ${THIRD_PARTY}/tensorflow_src)

# target-dependent preparation
set(NERVES_ARMV6 rpi rpi0)
set(NERVES_ARMV7NEON rpi2 rpi3 rpi3a bbb osd32mp1)
set(NERVES_AARCH64 rpi4)

if("$ENV{MIX_TARGET}" IN_LIST NERVES_ARMV6)
	message("Target is ARMv6")
	message("...donwload toolchain")
	include(toolchain_armv6.cmake)
elseif("$ENV{MIX_TARGET}" IN_LIST NERVES_ARMV7NEON)
	message("Target is ARMv7NEON")
	message("...donwload toolchain")
	include(toolchain_armv7neon.cmake)
elseif("$ENV{MIX_TARGET}" IN_LIST NERVES_AARCH64)
	message("AArch64 has not been testes yet!!!\n...donwload toolchain")
	include(toolchain_aarch64.cmake)
endif()

# check requirements
find_package(Patch)
if(NOT Patch_FOUND)
	message(FATAL_ERROR "Patch not found patch command")
endif()

#Set(FETCHCONTENT_QUIET FALSE)

# add Tensorflow sources
if(NOT EXISTS ${TFLITE_ROOTDIR})
	message("** Download Tensorflow lite etc.")
	FetchContent_Declare(tflite
		GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
		GIT_TAG        v2.9.2
		GIT_PROGRESS   TRUE
		SOURCE_DIR ${TFLITE_ROOTDIR}
		SOURCE_SUBDIR tensorflow/lite
		)
else()
	FetchContent_Declare(tflite
		SOURCE_DIR ${TFLITE_ROOTDIR}
		SOURCE_SUBDIR tensorflow/lite
		)
endif()
FetchContent_MakeAvailable(tflite)

# apply the patch to TensorFlow file set when target is Windows.
IF((${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows") AND (NOT EXISTS ${TFLITE_ROOTDIR}/patched))
  	execute_process(
  		COMMAND patch --verbos -p1 -i ${CMAKE_SOURCE_DIR}/msc.patch
  		WORKING_DIRECTORY ${TFLITE_ROOTDIR}
  		)
  	file(TOUCH ${TFLITE_ROOTDIR}/patched)
endif()

set(X_INTERP
	src/tflite/tfl_interp.cpp
	)
link_libraries(
	tensorflow-lite
	)
