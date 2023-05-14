/***  File Header  ************************************************************/
/**
* tfl_interp.cc
*
* Elixir/Erlang Port ext. of tensor flow lite
* @author	   Shozo Fukuda
* @date	create Sat Sep 26 06:26:30 JST 2020
* System	   MINGW64/Windows 10<br>
*
**/
/**************************************************************************{{{*/

#include <stdio.h>
#include "../tensor_spec.h"
#include "tfl_interp.h"

#include "tensorflow/lite/kernels/register.h"

#define TFLITE_EXPERIMENTAL 1

void add_custom_operations(tflite::ops::builtin::BuiltinOpResolver& resolver);

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
    sys.mInterp = new TflInterp(model, sys.mNumThread);
}

/***  Method Header  ******************************************************}}}*/
/**
* constructor
* @par DESCRIPTION
*   construct an instance.
**/
/**************************************************************************{{{*/
TflInterp::TflInterp(std::string tfl_model, int thread)
{
    // load tensor flow lite model
    mModel = tflite::FlatBufferModel::BuildFromFile(tfl_model.c_str());

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*mModel, resolver);
    builder.SetNumThreads(thread);
    builder(&mInterpreter);

    if (mInterpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "error: AllocateTensors()\n";
        exit(1);
    }
    
    mInputCount  = mInterpreter->inputs().size();
    mOutputCount = mInterpreter->outputs().size();
}

/***  Method Header  ******************************************************}}}*/
/**
* destructor
* @par DESCRIPTION
*   delate an instance.
**/
/**************************************************************************{{{*/
TflInterp::~TflInterp() {}

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
TflInterp::info(json& res)
{
    for (int index = 0; index < mInterpreter->inputs().size(); index++) {
        json tflite_tensor;

        TfLiteTensor* itensor = mInterpreter->input_tensor(index);

        tflite_tensor["name"] = std::string(itensor->name);
        tflite_tensor["type"] = std::string(TfLiteTypeGetName(itensor->type));
        for (int i = 0; i < itensor->dims->size; i++) {
            tflite_tensor["dims"].push_back(itensor->dims->data[i]);
        }

        res["inputs"].push_back(tflite_tensor);
    }

    for (int index = 0; index < mInterpreter->outputs().size(); index++) {
        json tflite_tensor;

        TfLiteTensor* itensor = mInterpreter->output_tensor(index);

        tflite_tensor["name"] = std::string(itensor->name);
        tflite_tensor["type"] = std::string(TfLiteTypeGetName(itensor->type));
        for (int i = 0; i < itensor->dims->size; i++) {
            tflite_tensor["dims"].push_back(itensor->dims->data[i]);
        }

        res["outputs"].push_back(tflite_tensor);
    }

#if TFLITE_EXPERIMENTAL
    int first_node_id = mInterpreter->execution_plan()[0];
    const auto& first_node_reg =
        mInterpreter->node_and_registration(first_node_id)->second;
    res["XNNPack"] = (tflite::GetOpNameByRegistration(first_node_reg) == "DELEGATE TfLiteXNNPackDelegate");
#endif
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
TflInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size)
{
    TfLiteTensor* itensor = mInterpreter->input_tensor(index);
    memcpy(itensor->data.raw, data, size);

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
TflInterp::set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv)
{
    TfLiteTensor* itensor = mInterpreter->input_tensor(index);
    float* dst = itensor->data.f;
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
TflInterp::invoke()
{
    mInterpreter->Invoke();
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
TflInterp::get_output_tensor(unsigned int index)
{
    TfLiteTensor* otensor = mInterpreter->output_tensor(index);
    return std::string(otensor->data.raw, otensor->bytes);
}

/*** tfl_interp.cc ********************************************************}}}*/
