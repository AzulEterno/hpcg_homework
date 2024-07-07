#include "config.hpp"

OptimizationType g_optimization_type = OPTIM_TYPE_REF;
DotProductType g_dot_product_type = DOT_PRODUCT_TYPE_REF;
WAXPYType g_waxpy_type = WAXPY_TYPE_REF;

const char* optimizationTypeToString(OptimizationType type) {
    switch (type) {
    case OPTIM_TYPE_REF:  return "OPTIMIZATION_NONE";
    case OPTIM_TYPE_ZCY: return "OPTIMIZATION_ZCY";
    case OPTIM_TYPE_LMB:  return "OPTIMIZATION_LMB";
    default: return "UNKNOWN";
    }
}

const char* dotproductTypeToString(DotProductType type) {
    switch (type) {
        case DOT_PRODUCT_TYPE_REF: return "DOT_PRODUCT_TYPE_REF";
        case DOT_PRODUCT_TYPE_ZCY: return "DOT_PRODUCT_TYPE_ZCY";
        default: return "UNKNOWN";
    }
}

const char* waxpyTypeToString(WAXPYType type) {
    switch (type) {
        case WAXPY_TYPE_REF: return "WAXPY_TYPE_REF";
        case WAXPY_TYPE_ZCY: return "WAXPY_TYPE_ZCY";
        default: return "UNKNOWN";
    }
}
