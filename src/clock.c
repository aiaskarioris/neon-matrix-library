#include "clock.h"
#include <stdio.h>

clock_t __clock;

// Starts the clock; Calling this function again will reset it
void startClock() {
  __clock = clock();
}

// Returns the number of clock ticks since the last `startClock()`
clock_t readClock() {
  return (clock() - __clock);
}

float clockToMS(clock_t timeDifference) {
  return (float)timeDifference / (float)CLOCKS_PER_SEC * 1000.0;
}

