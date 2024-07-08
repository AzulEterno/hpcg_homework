
#ifndef CONFIG_HPP
#define CONFIG_HPP

typedef enum :int {
    OPTIM_TYPE_REF,
    OPTIM_TYPE_ZCY,
    OPTIM_TYPE_LMB
} OptimizationType;

typedef enum {
    DOT_PRODUCT_TYPE_REF,
    DOT_PRODUCT_TYPE_ZCY
} DotProductType;

typedef enum {
    WAXPY_TYPE_REF,
    WAXPY_TYPE_ZCY
} WAXPYType;

const char* optimizationTypeToString(OptimizationType type);
const char* dotproductTypeToString(DotProductType type);
const char* waxpyTypeToString(WAXPYType type);

extern OptimizationType g_optimization_type; // g_mg_type
extern DotProductType g_dot_product_type;
extern WAXPYType g_waxpy_type;

#endif // CONFIG_HPP
