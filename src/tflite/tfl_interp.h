/***  File Header  ************************************************************/
/**
* @file tfl_interp.h
*
* Tiny ML interpreter on Tensorflow lite
* @author   Shozo Fukuda
* @date     create Fri Apr 15 12:55:02 JST 2022
* @date     update $Date:$
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
*******************************************************************************/
#ifndef _TFL_INTERP_H
#define _TFL_INTERP_H

/*--- INCLUDE ---*/
#include "../tiny_ml.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

/*--- CONSTANT ---*/

/*--- TYPE ---*/

/***  Class Header  *******************************************************}}}*/
/**
* Onnx Interpreter
* @par DESCRIPTION
*   Tiny ML Interpreter on Tensorflow lite
*
**/
/**************************************************************************{{{*/
class TflInterp :public TinyMLInterp {
friend class TflInterpTest;

//CONSTANT:
public:

//LIFECYCLE:
public:
  TflInterp(std::string tfl_model, int thread);
  virtual ~TflInterp();

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
    std::unique_ptr<tflite::Interpreter> mInterpreter;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
};

/*INLINE METHOD:
--$-----------------------------------*/

/*--- MACRO ---*/

/*--- EXTERNAL MODULE ---*/

/*--- EXTERNAL VARIABLE ---*/

#endif /* _TFL_INTERP_H */
