#include "config.hpp"

OptimizationType g_optimization_type = OPTIM_TYPE_REF;
// OptimizationType g_optimization_type = OPTIM_TYPE_ZCY;
// OptimizationType g_optimization_type = OPTIM_TYPE_LMB;
const char* optimizationTypeToString(OptimizationType type) {
    switch (type) {
    case OPTIM_TYPE_REF:  return "OPTIMIZATION_NONE";
    case OPTIM_TYPE_ZCY: return "OPTIMIZATION_ZCY";
    case OPTIM_TYPE_LMB:  return "OPTIMIZATION_LMB";
    default: return "UNKNOWN";
    }
}