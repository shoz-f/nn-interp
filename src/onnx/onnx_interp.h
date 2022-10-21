/***  File Header  ************************************************************/
/**
* @file onnx_interp.h
*
* Tiny ML interpreter on ONNX runtime
* @author   Shozo Fukuda
* @date     create Fri Apr 15 12:55:02 JST 2022
* @date     update $Date:$
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
*******************************************************************************/
#ifndef _ONNX_INTERP_H
#define _ONNX_INTERP_H

/*--- INCLUDE ---*/
#include "../tiny_ml.h"
#include <onnxruntime_cxx_api.h>

/*--- CONSTANT ---*/

/*--- TYPE ---*/

/***  Class Header  *******************************************************}}}*/
/**
* Onnx Interpreter
* @par DESCRIPTION
*   Tiny ML Interpreter on ONNX Runtime
*
**/
/**************************************************************************{{{*/
class OnnxInterp :public TinyMLInterp {
friend class OnnxInterpTest;

//CONSTANT:
public:

//LIFECYCLE:
public:
  OnnxInterp(std::string onnx_model);
  virtual ~OnnxInterp();

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
    Ort::Env mEnv{ORT_LOGGING_LEVEL_WARNING, "onnx_interp"};
    Ort::Session mSession{nullptr};

    char** mInputNames{nullptr};
    std::vector<Ort::Value> mInput;

    char** mOutputNames{nullptr};
    std::vector<Ort::Value> mOutput;
};

/*INLINE METHOD:
--$-----------------------------------*/

/*--- MACRO ---*/

/*--- EXTERNAL MODULE ---*/

/*--- EXTERNAL VARIABLE ---*/

#endif /* _ONNX_INTERP_H */
