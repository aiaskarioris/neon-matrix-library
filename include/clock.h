#pragma once
#include <time.h>

// === Timing =====================================================================================
// Starts the clock; Calling this function again will reset it
void startClock();

// Returns the number of clock ticks since the last `startClock()`
clock_t readClock();

// Converts clock ticks to milliseconds
float clockToMS(clock_t timeDifference);

// Macro for ease of use
#define GET_CLOCK ( clockToMS(readClock()) )

