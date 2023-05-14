/***  File Header  ************************************************************/
/**
* io_port.cc
*
* I/O driver for Elixir/Erlang Port
* @author      Shozo Fukuda
* @date create Fri Jun 17 04:59:08 JST 2022
* System       Windows10, WSL2/Ubuntu20.04.2, Linux Mint<br>
*
**/
/**************************************************************************{{{*/

#include <iostream>
#include <string>
#include <memory>

/***  Type ****************************************************************}}}*/
/**
* convert "unsigned int(32bit)" <-> "char[2]"
**/
/**************************************************************************{{{*/
union Magic {
    unsigned int ui32;
    char C[4];
};

/***  Module Header  ******************************************************}}}*/
/**
* receive command packet from Elixir/Erlang
* @par DESCRIPTION
*   receive command packet and store it to "cmd_line"
*
* @retval res >  0  success
* @retval res == 0  termination
* @retval res <  0  error
**/
/**************************************************************************{{{*/
int
rcv_packet_port(std::string& cmd_line)
{
    try {
        // receive packet size
        Magic len;
        /*+KNOWLEDGE:shoz:20/11/24:can't work "cin.get(len.C[3]).get(len.C[2]).." in WSL */
        len.C[3] = (char)std::cin.get();
        len.C[2] = (char)std::cin.get();
        len.C[1] = (char)std::cin.get();
        len.C[0] = (char)std::cin.get();

        // receive packet payload
        std::unique_ptr<char[]> buff(new char[len.ui32]);
        std::cin.read(buff.get(), len.ui32);

        // return received command line
        cmd_line.assign(buff.get(), len.ui32);
        return len.ui32;
    }
    catch(std::ios_base::failure) {
        return (std::cout.eof()) ? 0 : -1;
    }
    catch (std::bad_alloc& e) {
        std::cerr << e.what() << "@rcv_packet_port" << std::endl;
        return -1;
    }
}

/***  Module Header  ******************************************************}}}*/
/**
* send result packet to Elixir/Erlang
* @par DESCRIPTION
*   construct message packet and send it to stdout
*
* @return count of sent byte or error code
**/
/**************************************************************************{{{*/
int
snd_packet_port(std::string result)
{
    try {
        Magic len = { static_cast<unsigned int>(result.size()) };
        (std::cout.put(len.C[3]).put(len.C[2]).put(len.C[1]).put(len.C[0]) << result).flush();
        return len.ui32;
    }
    catch(std::ios_base::failure) {
        return (std::cout.eof()) ? 0 : -1;
    }
}

/*** io_port.cc ********************************************************}}}*/
