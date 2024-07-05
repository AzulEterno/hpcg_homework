#define HPCG_USE_ELL  //（1）连续存储ELL格式

#define HPCG_USE_MULTICOLORING  //（2）染色

#define HPCG_USE_POINT_MULTICOLORING  //（3）分点染色
//#define HPCG_USE_MULTICOLORING_RRARRANGE  //（4）分点染色重排

// #define HPCG_USE_BLOCK_MULTICOLORING  //（5）分块染色


//可用优化组合

// 全部关闭
// 1
// 1、2、3
// 1、2、3、4  //重排尚存在问题
// 1、2、5


