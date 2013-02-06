#ifndef _TIMER_H_
#define _TIMER_H_

/**
 * You must link with the real-time library. Adding -lrt to your linker line
 * should do it.
 */

#include <time.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#ifdef CLOCK_MONOTONIC_RAW
#define TIMER_CLOCK CLOCK_MONOTONIC_RAW
#else
#define TIMER_CLOCK CLOCK_MONOTONIC
#endif

#define ONE_BILLION 1000000000

class Timer {
   public:
      void start() {
         clock_gettime(TIMER_CLOCK, &time_begin);
      }

      void stop() {
         clock_gettime(TIMER_CLOCK, &time_end);
      }

      uint64_t result() {
         uint64_t duration = 0;

         if (time_end.tv_nsec < time_begin.tv_nsec) {
            duration += (time_end.tv_sec - time_begin.tv_sec - 1) * ONE_BILLION;
            duration += (time_end.tv_nsec + ONE_BILLION) - time_begin.tv_nsec;
         } else {
            duration += (time_end.tv_sec - time_begin.tv_sec) * ONE_BILLION;
            duration += time_end.tv_nsec - time_begin.tv_nsec;
         }

         return duration;
      }

   private:
      struct timespec time_begin;
      struct timespec time_end;
}

#undef ONE_BILLION
#undef TIMER_CLOCK

#endif
