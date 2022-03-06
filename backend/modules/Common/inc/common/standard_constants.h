#ifndef MODULES_COMMON_INC_STANDARD_CONSTANTS_H
#define MODULES_COMMON_INC_STANDARD_CONSTANTS_H

static constexpr Vector2<int> COHERENCY_DRAW_RATIO = {1, 1 }; // m / n
static constexpr int RANDOM_DRAWS = COHERENCY_DRAW_RATIO.x * COHERENCY_DRAW_RATIO.y;
static constexpr int RAND_THREADS = 64;

#endif