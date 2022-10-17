/***  File Header  ************************************************************/
/**
* tiny_ml.cpp
*
* Elixir/Erlang Port ext. of Tiny ML
* @author   Shozo Fukuda
* @date     create Mon Apr 18 08:52:19 JST 2022
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
**/
/**************************************************************************{{{*/

#include <stdio.h>
#include <fstream>

#include "tiny_ml.h"
#include "postprocess.h"

/***  Module Header  ******************************************************}}}*/
/**
* query dimension of input tensor
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
std::string
info(SysInfo& sys, const void*)
{
    json res;

    res["exe"  ]   = sys.mExe;
    res["model"]   = sys.mModelPath;
    res["label"]   = sys.mLabelPath;
    res["class"]   = sys.mNumClass;
    res["thread"]  = sys.mNumThread;

    sys.mInterp->info(res);

    json lap_time;
    lap_time["input"]  = sys.mLap[0].count();
    lap_time["exec"]   = sys.mLap[1].count();
    lap_time["output"] = sys.mLap[2].count();
    res["times"] = lap_time;

    return res.dump();
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
static int
set_input_tensor(TinyMLInterp* interp, const void* args)
{
    int res;

    PACK(
    struct Prms {
        unsigned int size;
        unsigned int index;
        unsigned int dtype;
        float        min;
        float        max;
        uint8_t      data[1];
    });
    const Prms*  prms = reinterpret_cast<const Prms*>(args);
    const int prms_size = sizeof(prms->size) + prms->size;
    const int data_size = prms_size - sizeof(Prms) + sizeof(uint8_t);

    if (prms->index >= interp->InputCount()) {
        return -1;
    }

    switch (prms->dtype) {
    case 0:
        res = interp->set_input_tensor(prms->index, prms->data, data_size);
        break;

    case 1:
        {
        double a = (prms->max - prms->min)/255.0;
        double b = prms->min;
        res = interp->set_input_tensor(prms->index, prms->data, data_size,
                                       [a,b](uint8_t x){ return static_cast<float>(a*x + b); });
        }
        break;

    default:
        return -3;
    }

    return (res < 0) ? res : prms_size;
}

std::string
set_input_tensor(SysInfo& sys, const void* args)
{
    json res;

    sys.start_watch();

    int status = set_input_tensor(sys.mInterp, args);
    res["status"] = (status >= 0) ? 0 : status;

    sys.LAP_INPUT();

    return res.dump();
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
std::string
invoke(SysInfo& sys, const void*)
{
    json res;

    sys.start_watch();

    res["status"] = sys.mInterp->invoke();

    sys.LAP_EXEC();

    return res.dump();
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
get_output_tensor(SysInfo& sys, const void* args)
{
    struct Prms {
        unsigned int index;
    };
    const Prms*  prms = reinterpret_cast<const Prms*>(args);

    if (prms->index >= sys.mInterp->OutputCount()) {
        return std::string("");
    }

    sys.start_watch();

    std::string&& res = sys.mInterp->get_output_tensor(prms->index);

    sys.LAP_OUTPUT();

    return res;
}

/***  Module Header  ******************************************************}}}*/
/**
* execute inference in session mode
* @par DESCRIPTION
*
*
* @retval
**/
/**************************************************************************{{{*/
std::string
run(SysInfo& sys, const void* args)
{
    // set input tensors
    PACK(
    struct Prms {
        unsigned int  count;
        unsigned char data[1];
    });
    const Prms* prms = reinterpret_cast<const Prms*>(args);

    sys.start_watch();

    const unsigned char* ptr = prms->data;
    for (unsigned int i = 0; i < prms->count; i++) {
        int next = set_input_tensor(sys.mInterp, ptr);
        if (next < 0) {
            // error about input tensors: error_code {-1..-3}
            return std::string(reinterpret_cast<char*>(&next), sizeof(next));
        }

        ptr += next;
    }

    sys.LAP_INPUT();

    // invoke
    if (!sys.mInterp->invoke()) {
        // error about invoke: error_code {-11..}
        int status = -11;
        return std::string(reinterpret_cast<char*>(&status), sizeof(status));
    }

    sys.LAP_EXEC();

    // get output tensors  <<count::little-integer-32, size::little-integer-32, bin::binary-size(size), ..>>
    uint32_t count = static_cast<uint32_t>(sys.mInterp->OutputCount());
    std::string output(reinterpret_cast<char*>(&count), sizeof(count));

    for (uint32_t index = 0; index < count; index++) {
        std::string&& otensor = sys.mInterp->get_output_tensor(index);
        uint32_t size = static_cast<uint32_t>(otensor.size());
        output += std::string(reinterpret_cast<char*>(&size), sizeof(size))
               +  otensor;
    }

    sys.LAP_OUTPUT();

    return output;
}

/**************************************************************************}}}**
* command dispatch table
***************************************************************************{{{*/
typedef std::string (TMLFunc)(SysInfo& sys, const void* args);

TMLFunc* gCmdTbl[] = {
    info,
    set_input_tensor,
    invoke,
    get_output_tensor,
    run,

    POST_PROCESS
};

const int gMaxCmd = sizeof(gCmdTbl)/sizeof(TMLFunc*);

/***  Module Header  ******************************************************}}}*/
/**
* tensor flow lite interpreter
* @par DESCRIPTION
*
**/
/**************************************************************************{{{*/
void
interp(std::string& model, std::string& labels, std::string& inputs, std::string& outputs)
{
    init_interp(gSys, model, inputs, outputs);

    // load labels
    if (labels != "none") {
        std::string   label;
        std::ifstream lb_file(labels);
        if (lb_file.fail()) {
            std::cerr << "error: Failed to open file\n";
            exit(1);
        }
        while (getline(lb_file, label)) {
            gSys.mLabel.emplace_back(label);
        }
        gSys.mNumClass = gSys.mLabel.size();
    }
    else {
        gSys.mLabel.clear();
        gSys.mNumClass = 0;
    }

    // REPL
    for (;;) {
        // receive command packet
        std::string cmd_line;
        int n = gSys.mRcv(cmd_line);
        if (n <= 0) {
            break;
        }

        // command branch
        PACK(
        struct Cmd {
            unsigned int cmd;
            uint8_t        args[1];
        });
        const Cmd& call = *reinterpret_cast<const Cmd*>(cmd_line.data());

        std::string&& result = (call.cmd < gMaxCmd) ? gCmdTbl[call.cmd](gSys, call.args)
                                                     : "unknown command";

        // send the result in JSON string
        n = gSys.mSnd(result);
        if (n <= 0) {
            break;
        }
    }

    delete gSys.mInterp;
}

/*** tiny_ml.cpp **********************************************************}}}*/
