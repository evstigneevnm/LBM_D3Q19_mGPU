#pragma once

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <iostream>
#include <mpi.h>

#include "Macro.h"



void dispatch_info_data(const char *property_tree_file_name, communication_variables *COM, std::string control_file_2_read, std::string control_file_2_write);
void read_info_data(const char *property_tree_file_name, communication_variables *COM, std::string &control_file_2_read, std::string &control_file_2_write);