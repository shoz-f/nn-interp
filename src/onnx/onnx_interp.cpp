/***  File Header  ************************************************************/
/**
* onnx_interp.cpp
*
* Tiny ML interpreter on ONNX runtime
* @author   Shozo Fukuda
* @date     create Fri Apr 15 12:55:02 JST 2022
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
**/
/**************************************************************************{{{*/

#include <stdio.h>
#include "../tensor_spec.h"
#include "onnx_interp.h"

/***  Module Header  ******************************************************}}}*/
/**
* initialize interpreter
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
void init_interp(SysInfo& sys, std::string& model, std::string& inputs, std::string& outputs)
{
    sys.mInterp = new OnnxInterp(model);
}

/***  Module Header  ******************************************************}}}*/
/**
* query dimension of onnx tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
size_t
get_tensor_size(
Ort::Value& value)
{
    size_t size;

    auto tensor_info = value.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();

    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 0;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        size = 8;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        size = 4;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        size = 2;
        break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        size = 1;
        break;

    default:
        return 0;
    }

    size_t shape_len = tensor_info.GetDimensionsCount();
    int64_t* shape = new int64_t[shape_len];
    tensor_info.GetDimensions(shape, shape_len);
    for (int j = 0; j < shape_len; j++) {
        if (shape[j] == -1) { shape[j] = 1; }

        size *= shape[j];
    }

    delete [] shape;

    return size;
}

/***  Method Header  ******************************************************}}}*/
/**
* constructor
* @par DESCRIPTION
*   construct an instance.
**/
/**************************************************************************{{{*/
OnnxInterp::OnnxInterp(std::string onnx_model)
{
    Ort::AllocatorWithDefaultOptions _ort_alloc;
    Ort::SessionOptions session_options;

#if _MSC_VER >=1900
    std::wstring widestr = std::wstring(onnx_model.begin(), onnx_model.end());
    mSession = Ort::Session(mEnv, widestr.c_str(), session_options);
#elif __GNUC__
    mSession = Ort::Session(mEnv, onnx_model.c_str(), session_options);
#endif

    mInputCount = mSession.GetInputCount();
    mInputNames = new char*[mInputCount];
    for (int i = 0; i < mInputCount; i++) {
        mInputNames[i] = mSession.GetInputName(i, _ort_alloc);

        Ort::TypeInfo type_info = mSession.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();

        size_t shape_len = tensor_info.GetDimensionsCount();
        int64_t* shape = new int64_t[shape_len];
        tensor_info.GetDimensions(shape, shape_len);
        for (int j = 0; j < shape_len; j++) {
            if (shape[j] == -1) { shape[j] = 1; }
        }

        mInput.emplace_back(Ort::Value::CreateTensor(_ort_alloc, shape, shape_len, type));

        delete [] shape;
    }

    mOutputCount = mSession.GetOutputCount();
    mOutputNames = new char*[mOutputCount];
    for (int i = 0; i < mOutputCount; i++) {
        mOutputNames[i] = mSession.GetOutputName(i, _ort_alloc);

        Ort::TypeInfo type_info = mSession.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();

        size_t shape_len = tensor_info.GetDimensionsCount();
        int64_t* shape = new int64_t[shape_len];
        tensor_info.GetDimensions(shape, shape_len);
        for (int j = 0; j < shape_len; j++) {
            if (shape[j] == -1) { shape[j] = 1; }
        }

        mOutput.push_back(Ort::Value::CreateTensor(_ort_alloc, shape, shape_len, type));

        delete [] shape;
    }
}

/***  Method Header  ******************************************************}}}*/
/**
* destructor
* @par DESCRIPTION
*   delate an instance.
**/
/**************************************************************************{{{*/
OnnxInterp::~OnnxInterp()
{
    Ort::AllocatorWithDefaultOptions _ort_alloc;

    for (int i = 0; i < mInputCount; i++) {
        _ort_alloc.Free(mInputNames[i]);
    }
    delete [] mInputNames;

    for (int i = 0; i < mOutputCount; i++) {
        _ort_alloc.Free(mOutputNames[i]);
    }
    delete [] mOutputNames;
}

/***  Module Header  ******************************************************}}}*/
/**
* query dimension of input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
void
OnnxInterp::info(json& res)
{
    const std::string _dtype[] = {
        "UNDEFINED",
        "FLOAT",        // maps to c type float
        "UINT8",        // maps to c type uint8_t
        "INT8",         // maps to c type int8_t
        "UINT16",       // maps to c type uint16_t
        "INT16",        // maps to c type int16_t
        "INT32",        // maps to c type int32_t
        "INT64",        // maps to c type int64_t
        "STRING",       // maps to c++ type std::string
        "BOOL",
        "FLOAT16",
        "DOUBLE",       // maps to c type double
        "UINT32",       // maps to c type uint32_t
        "UINT64",       // maps to c type uint64_t
        "COMPLEX64",    // complex with float32 real and imaginary components
        "COMPLEX128",   // complex with float64 real and imaginary components
        "BFLOAT16"      // Non-IEEE floating-point format based on IEEE754 single-precision
    };

    for (int index = 0; index < mInputCount; index++) {
        json onnx_tensor;

        onnx_tensor["index"] = index;
        onnx_tensor["name"]  = mInputNames[index];

        Ort::TypeInfo type_info = mSession.GetInputTypeInfo(index);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        onnx_tensor["type"] = _dtype[tensor_info.GetElementType()];

        for (const auto& n : tensor_info.GetShape()) {
            if (n == -1) {
                onnx_tensor["dims"].push_back("none");
            }
            else {
                onnx_tensor["dims"].push_back(n);
            }
        }

        res["inputs"].push_back(onnx_tensor);
    }

    for (int index = 0; index < mOutputCount; index++) {
        json onnx_tensor;

        onnx_tensor["index"] = index;
        onnx_tensor["name"]  = mOutputNames[index];

        Ort::TypeInfo type_info = mSession.GetOutputTypeInfo(index);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        onnx_tensor["type"] = _dtype[tensor_info.GetElementType()];

        for (const auto& n : tensor_info.GetShape()) {
            if (n == -1) {
                onnx_tensor["dims"].push_back("none");
            }
            else {
                onnx_tensor["dims"].push_back(n);
            }
        }

        res["outputs"].push_back(onnx_tensor);
    }
}

/***  Module Header  ******************************************************}}}*/
/**
* set input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
int
OnnxInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size)
{
 //   if (size != mInput[index].GetStringTensorDataLength()) {
 //       return -2;
 //   }

    memcpy(mInput[index].GetTensorMutableData<uint8_t>(), data, size);

    return size;
}

/***  Module Header  ******************************************************}}}*/
/**
* set input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
int
OnnxInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv)
{
//    if (size != mInput[index].GetStringTensorDataLength()/sizeof(float)) {
//        return -2;
//    }

    float* dst = mInput[index].GetTensorMutableData<float>();
    const uint8_t* src = data;
    for (int i = 0; i < size; i++) {
        *dst++ = conv(*src++);
    }

    return size;
}

/***  Module Header  ******************************************************}}}*/
/**
* execute inference
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
bool
OnnxInterp::invoke()
{
    mOutput = mSession.Run(Ort::RunOptions{nullptr}, mInputNames, mInput.data(), mInputCount, mOutputNames, mOutputCount);
    return true;
}

/***  Module Header  ******************************************************}}}*/
/**
* get result tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
std::string
OnnxInterp::get_output_tensor(unsigned int index)
{
    return std::string(mOutput[index].GetTensorData<char>(), get_tensor_size(mOutput[index]));
}

/*** onnx_interp.cpp ******************************************************}}}*/
