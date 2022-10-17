/***  File Header  ************************************************************/
/**
* postprocess.h
*
* system setting - used throughout the system
* @author      Shozo Fukuda
* @date create Tue Jul 13 14:32:28 JST 2021
* System    Windows10, WSL2/Ubuntu 20.04.2<br>
*
*******************************************************************************/
#ifndef _POSTPROCESS_H
#define _POSTPROCESS_H

/**************************************************************************}}}**
*
***************************************************************************{{{*/
std::string non_max_suppression_multi_class(SysInfo& sys, const void* args);

#define POST_PROCESS \
    non_max_suppression_multi_class

#endif /* _POSTPROCESS_H */
