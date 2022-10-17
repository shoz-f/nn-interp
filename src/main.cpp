/***  File Header  ************************************************************/
/**
* onnx_interp.cpp
*
* Tiny ML interpreter on ONNX runtime
* @author   Shozo Fukuda
* @date		create Fri Apr 15 12:55:02 JST 2022
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
**/
/**************************************************************************{{{*/

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <string>
#include "tiny_ml.h"
#include "getopt/getopt.h"

/***  Global **************************************************************}}}*/
/**
* system infomation
**/
/**************************************************************************{{{*/
SysInfo gSys;

/***  Module Header  ******************************************************}}}*/
/**
* prit usage
* @par DESCRIPTION
*   print usage to terminal
**/
/**************************************************************************{{{*/
void usage()
{
    std::cout
      << "onnx_interp [opts] <model.onnx> <class.label>\n"
      << "\toption:\n"
      << "\t  -i <spec> : input tensor spec - \"f4,1,3,224,224\"\n"
      << "\t  -o <spec> : output tensor spec - \"f4,1,1000\"\n"
      << "\t  -d <num> : diagnosis mode\n"
      << "\t             1 = save the formed image\n"
      << "\t             2 = save model's input/output tensors\n"
      << "\t             4 = save result of the prediction\n";
}

/***  Module Header  ******************************************************}}}*/
/**
* onnx runntime for Elixir/Erlang Port ext.
* @par DESCRIPTION
*   Elixir/Erlang Port extension (experimental)
*
* @return exit status
**/
/**************************************************************************{{{*/
int
main(int argc, char* argv[])
{
	int opt, longindex;
	const struct option longopts[] = {
	    {"inputs",   required_argument, NULL, 'i'},
	    {"outputs",  required_argument, NULL, 'o'},
		{"debug",    required_argument, NULL, 'd'},
        {"parallel", required_argument, NULL, 'j'},
		{0,0,0,0}
	};

    // initialize system environment
    gSys.mDiag      = 0;
    gSys.mNumThread = 4;
    gSys.reset_lap();
    
    std::string inputs;
    std::string outputs;

	for (;;) {
		opt = getopt_long(argc, argv, "i:o:d:j:", longopts, NULL);
		if (opt == -1) {
			break;
		}
		else switch (opt) {
		case 'i':
		    inputs = optarg;
		    break;
		case 'o':
		    outputs = optarg;
		    break;
		case 'd':
			break;
        case 'j':
            gSys.mNumThread = atoi(optarg);
            break;
		case '?':
		case ':':
			std::cerr << "error: unknown options\n\n";
			usage();
			return 1;
		}
	}
	if ((argc - optind) < 2) {
		// argument error
		std::cerr << "error: expect <model.onnx>\n\n";
		usage();
		return 1;
	}

    // save exe infomations
    gSys.mExe.assign(argv[0]);
    gSys.mModelPath.assign(argv[optind]);
    gSys.mLabelPath.assign(argv[optind+1]);

    // initialize i/o
    std::cin.exceptions(std::ios_base::badbit|std::ios_base::failbit|std::ios_base::eofbit);
    std::cout.exceptions(std::ios_base::badbit|std::ios_base::failbit|std::ios_base::eofbit);

#ifdef _WIN32
	_setmode(_fileno(stdin),  O_BINARY);
	_setmode(_fileno(stdout), O_BINARY);
#endif
	gSys.mRcv = rcv_packet_port;
	gSys.mSnd = snd_packet_port;

    // run interpreter
    interp(gSys.mModelPath, gSys.mLabelPath, inputs, outputs);

    return 0;
}

/*** main.cpp *************************************************************}}}*/
