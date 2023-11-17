#pragma once

/// Macro to switch execution based on a boolean condition at compile time.
/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...        - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```
#define BOOL_SWITCH(COND, CONST_NAME, ...)            \
    [&] {                                             \
        if (COND) {                                   \
            constexpr static bool CONST_NAME = true;  \
            return __VA_ARGS__();                     \
        } else {                                      \
            constexpr static bool CONST_NAME = false; \
            return __VA_ARGS__();                     \
        }                                             \
    }()

/// Macro to switch execution paths based on the size of a head dimension.
/// The head dimension size determines which templated function to call.
/// @param HEADDIM - an integer representing the head dimension size.
/// @param ...     - code to execute with the selected head dimension size.
///
/// Usage:
/// ```
/// FWD_HEADDIM_SWITCH(tensorHeadDim, [&] {
///     OptimizedFunction<kHeadDim>(...);
/// });
/// ```
#define FWD_HEADDIM_SWITCH(HEADDIM, ...)         \
    [&] {                                        \
        if (HEADDIM <= 32) {                     \
            constexpr static int kHeadDim = 32;  \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 64) {              \
            constexpr static int kHeadDim = 64;  \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 96) {              \
            constexpr static int kHeadDim = 96;  \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 128) {             \
            constexpr static int kHeadDim = 128; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 160) {             \
            constexpr static int kHeadDim = 160; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 192) {             \
            constexpr static int kHeadDim = 192; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 224) {             \
            constexpr static int kHeadDim = 224; \
            return __VA_ARGS__();                \
        } else if (HEADDIM <= 256) {             \
            constexpr static int kHeadDim = 256; \
            return __VA_ARGS__();                \
        }                                        \
    }()
