#ifndef MODULES_COMMON_INC_STANDARD_CONSTANTS_H
#define MODULES_COMMON_INC_STANDARD_CONSTANTS_H

static constexpr MyType2I COHERENCY_DRAW_RATIO = { 1, 1 }; // m / n
static constexpr int RANDOM_DRAWS = COHERENCY_DRAW_RATIO.x * COHERENCY_DRAW_RATIO.y;

static constexpr MyType MYTYPE_1 = 1;
static constexpr MyType MYTYPE_2 = 2;

static constexpr int RAND_THREADS = 64;

#endif