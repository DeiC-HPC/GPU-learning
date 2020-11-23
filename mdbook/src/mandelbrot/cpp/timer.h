#include <stdio.h>
#include <time.h>
#include <unistd.h>

class timer {
private:
  static void timespec_diff(struct timespec *start, struct timespec *stop,
                            struct timespec *result) {
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
      result->tv_sec = stop->tv_sec - start->tv_sec - 1;
      result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
      result->tv_sec = stop->tv_sec - start->tv_sec;
      result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }

    return;
  }

  timespec start;
  timespec end;
  bool is_stopped;

public:
  timer() : is_stopped(false) { clock_gettime(CLOCK_MONOTONIC, &start); }

  bool stop() {
    bool old = is_stopped;
    if (!is_stopped) {
      clock_gettime(CLOCK_MONOTONIC, &end);
      is_stopped = true;
    }
    return old;
  }

  double getTime() {
    timespec diff;

    if (!is_stopped) {
      clock_gettime(CLOCK_MONOTONIC, &end);
    }

    timespec_diff(&start, &end, &diff);

    return diff.tv_sec + diff.tv_nsec / 1000000000.0;
  }
};