/***  File Header  ************************************************************/
/**
* torch_interp.cpp
*
* Tiny ML interpreter on LibTorch
* @author   Shozo Fukuda
* @date     create Mon Sep 12 18:31:08 2022
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
**/
/**************************************************************************{{{*/

#include <torch/torch.h>
#include <torch/nn/functional/activation.h>
#include "../tensor_spec.h"
#include "torch_interp.h"

/***  Module Header  ******************************************************}}}*/
/**
* initialize interpreter
* @par DESCRIPTION
*
*
**/
/**************************************************************************{{{*/
void init_interp(SysInfo& sys, std::string& model, std::string& inputs, std::string& outputs)
{
    sys.mInterp = new TorchInterp(model, inputs, outputs);
}

/***  Method Header  ******************************************************}}}*/
/**
* constructor
* @par DESCRIPTION
*   construct an instance.
**/
/**************************************************************************{{{*/
TorchInterp::TorchInterp(std::string& model, std::string& inputs, std::string& outputs)
{
	try {
	    mModule = torch::jit::load(model);
	}
	catch (const c10::Error& e) {
	    std::cerr << "Error loading model\n";
	    std::cerr << e.what_without_backtrace();
	    throw;
	}

	torch::NoGradGuard no_grad;
	mModule.eval();

	//std::cout << "Model loaded successfully\n";

    mInputSpec = parse_tensor_spec(inputs, true);
    mInputCount = mInputSpec.size();

    mOutputSpec = parse_tensor_spec(outputs);
    mOutputCount = mOutputSpec.size();
}

/***  Method Header  ******************************************************}}}*/
/**
* destructor
* @par DESCRIPTION
*   delate an instance.
**/
/**************************************************************************{{{*/
TorchInterp::~TorchInterp()
{
    for (auto item : mInputSpec) { delete item; }
    mInputSpec.clear();

    for (auto item : mOutputSpec) { delete item; }
    mOutputSpec.clear();
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
TorchInterp::info(json& res)
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

    res["framework"] = "LibTorch";

    for (int index = 0; index < mInputCount; index++) {
        json json_tensor;

        json_tensor["index"] = index;
        json_tensor["type"]  = _dtype[mInputSpec[index]->mDType];
        for (const auto& n : mInputSpec[index]->mShape) {
            json_tensor["dims"].push_back(n);
        }

        res["inputs"].push_back(json_tensor);
    }

    for (int index = 0; index < mOutputCount; index++) {
        json json_tensor;

        json_tensor["index"] = index;
        json_tensor["type"]  = _dtype[mOutputSpec[index]->mDType];
        for (const auto& n : mOutputSpec[index]->mShape) {
            json_tensor["dims"].push_back(n);
        }

        res["outputs"].push_back(json_tensor);
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
TorchInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size)
{
    memcpy(mInputSpec[index]->mBlob, data, size);
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
TorchInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv)
{
    float* dst = reinterpret_cast<float*>(mInputSpec[index]->mBlob);

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
* @retval
**/
/**************************************************************************{{{*/
bool
TorchInterp::invoke()
{
    std::vector<torch::jit::IValue> inputs;

    for (const auto blob : mInputSpec) {
        inputs.push_back(torch::from_blob(blob->mBlob, c10::IntArrayRef(blob->mShape)));
    }

    mOutput = mModule.forward(inputs);

    return true;
}

/***  Module Header  ******************************************************}}}*/
/**
* get result tensor
* @par DESCRIPTION
*
* @retval
**/
/**************************************************************************{{{*/
std::string
TorchInterp::get_output_tensor(unsigned int index)
{
    if (mOutput.isTuple()) {
        return "";
    }
    else if (mOutput.isTensor()) {
        at::Tensor t = mOutput.toTensor();
        return std::string(reinterpret_cast<char*>(t.data_ptr()), t.nbytes());
    }
    else {
        return "";
    }
}

/*** torch_interp.cpp *****************************************************}}}*/
