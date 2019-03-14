/**
 * The default mode for an ifstream is std::ios_base::in, which opens the file for input. 
 * The default mode for ofstream is std::ios_base::out | std::ios_base::trunc
 * The ate mode (short for at-end), which sets the stream’s initial position to the end of the existing file contents
 * Another useful mode for output is app (short for append), which causes every write to append to the file. 
 * That is, app affects every write, whereas ate affects only the starting position. The app mode is useful when writing to a log file.  
 *
 */


// Write a debug() function that takes a single string as an argument and writes that string to a file named “debug.txt”.
#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <string>

/** @brief Write a debug message to the file @c "debug.txt"
 * @param msg The message to write
 */
void debug (std::string const& msg);

#endif
