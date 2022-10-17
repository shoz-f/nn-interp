/***  File Header  ************************************************************/
/**
* @file torch_interp.h
*
* Tiny ML interpreter on LibTorch
* @author   Shozo Fukuda
* @date     create Fri Apr 15 12:55:02 JST 2022
* @date     update $Date:$
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
*******************************************************************************/
#ifndef _TORCH_INTERP_H
#define _TORCH_INTERP_H

/*--- INCLUDE ---*/
#include "../tiny_ml.h"
#include <torch/script.h>
#include <torch/nn/functional/activation.h>

/*--- CONSTANT ---*/

/*--- TYPE ---*/

/***  Class Header  *******************************************************}}}*/
/**
* Pytorch Interpreter
* @par DESCRIPTION
*   Tiny ML Interpreter on Pytorch
*
**/
/**************************************************************************{{{*/
class TorchInterp :public TinyMLInterp {
friend class TorchInterpTest;

//CONSTANT:
public:

//LIFECYCLE:
public:
    TorchInterp(std::string onnx_model);
    TorchInterp(std::string& torch_script, std::string& inputs, std::string& outputs);
    virtual ~TorchInterp();

//ACTION:
public:
    void info(json& res);
    int set_input_tensor(unsigned int index, const uint8_t* data, int size);
    int set_input_tensor(unsigned int index, const uint8_t* data, int size, std::function<float(uint8_t)> conv);
    bool invoke();
    std::string get_output_tensor(unsigned int index);

//ACCESSOR:
public:

//INQUIRY:
public:

//ATTRIBUTE:
private:
    torch::jit::script::Module mModule;

    std::vector<TensorSpec*> mInputSpec;
    std::vector<TensorSpec*> mOutputSpec;

    torch::jit::IValue mOutput;
};

/*INLINE METHOD:
--$-----------------------------------*/

/*--- MACRO ---*/

/*--- EXTERNAL MODULE ---*/

/*--- EXTERNAL VARIABLE ---*/

#endif /* _TORCH_INTERP_H */
