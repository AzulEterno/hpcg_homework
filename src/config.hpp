#ifndef CONFIG_HPP
#define CONFIG_HPP

typedef enum {
    OPTIM_TYPE_REF,
    OPTIM_TYPE_ZCY,
    OPTIM_TYPE_LMB
} OptimizationType;

extern OptimizationType g_optimization_type;

#endif // CONFIG_HPP
