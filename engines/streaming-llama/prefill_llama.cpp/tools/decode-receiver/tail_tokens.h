#pragma once

#include "llama.h"

#include <string>
#include <vector>

bool load_tail_tokens_file(const std::string & path, std::vector<llama_token> & tokens_out, std::string & error);
