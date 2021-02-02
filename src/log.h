#pragma once

#include <cstdio>
#include <cstdlib>

#ifndef DNDEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##args)
#else
#define DEBUG_PRINT(fmt, args...)
#endif
