
#ifndef MLLM_NETPARAMETER_H
#define MLLM_NETPARAMETER_H

#include "Types.hpp"
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using std::string;
using std::vector;
using std::map;

namespace mllm {

typedef struct TNetTensor NetTensor;
typedef struct TNetParameter NetParameter;

typedef struct TNetOp {
    OpType type;
    vector<NetTensor *> in;   // Operation input tensors
    vector<NetTensor *> out;
    vector<string> in_op; // input ops' names;
    string name;
    OpParam param;
    int out_size = 1; // number of output tensors ...

} NetOp;

typedef struct TNetParameter { // this may be a compute graph
    string weights_path;              // no usage ??
    string model_path;                // no usage ??
    vector<NetOp *> net_ops;          // all Ops in compute graph
    vector<NetTensor *> net_tensors;  // all tensors in compute graph (maybe)
    std::set<NetTensor *> net_inputs; // 计算图的叶子节点(大概)
    std::set<NetTensor *> net_outputs; // output node of compute graph (maybe) no usage ???
    void topologySort();   // sort net_ops, so that we could execute op in order

} NetParameter;

// 前置声明
struct Context {
    vector<NetParameter> sub_param_;  // each sub_param is a subgraph
    vector<NetOp *> net_ops;
    std::set<NetTensor *> net_tensors; // all the tensors
    int idx = 0;  // it seems that every time we add a tensor, the idx increase by 1. maybe as a unique id
    int active_sub = 0;   // we may create multiple compute graph, this is the current compute graph (index of sub_param_)
};
inline NetParameter *get_active_subgraph(Context *ctx) {
    if (ctx->active_sub >= ctx->sub_param_.size()) {
        ctx->sub_param_.emplace_back();
    }
    return &ctx->sub_param_[ctx->active_sub];
}

struct intTensor_pair {
    int start_i;
    NetTensor* end_i;
};
struct Tensor_pair {
    NetTensor* end_i;
};
#define ANYDIM -198098
typedef struct TNetTensor {
    string name;
    vector<int> shape_;   // only input has shape ......
    DataType type;
    NetOp *in;     // 一个tensor可以由别的一些tensor计算得到 这个计算包括计算的类型和计算输入，都在NetOp里了
    vector<NetOp *> out;  // 这个tensor可以是别的一些计算的输入 这个out就是以该tensor为输入的NetOp
    NetParameter *subgraph;  // which subgraph this tensor is in
    Context *ctx;

    /**
     * \brief Clips the tensor along the specified dimensions.
     * \param b A vector specifying the start and end indices for the batch dimension.
     * \param h A vector specifying the start and end indices for the head dimension.
     * \param s A vector specifying the start and end indices for the sequence dimension.
     * \param d A vector specifying the start and end indices for the dimension dimension.
     * \return A pointer to the resulting tensor after clipping.
     *
     * e.g. clip({}, {}, {0, 1}, {}) will clip the tensor along the sequence dimension from 0 to 2;
     *      clip({}, {}, {0}, {}) will clip the tensor along the sequence dimension from 0 to 1;
     *      clip({}, {}, {-1}, {}) will clip the tensor along the sequence dimension from dim_len-1 to dim_len;
     *      _clip({}, {}, {0, i->shape(SEQUENCE)}, {}); will clip the tensor along the sequence dimension from 0 to i->shape(SEQUENCE);
     *      _clip({}, {}, {in_len}, {}); will clip the tensor along the sequence dimension from in_len to in_len+1;
     */
    NetTensor *clip(vector<int> b, vector<int> h, vector<int> s, vector<int> d);
    NetTensor *_clip(intTensor_pair b, intTensor_pair h, intTensor_pair s, intTensor_pair d);
    NetTensor *_clip(Tensor_pair b, Tensor_pair h, Tensor_pair s, Tensor_pair d);
    NetTensor *shape(Chl axis);
    NetTensor *view(int b, int h, int s, int d);
    NetTensor *flatten(Chl axis_start, Chl axis_end);
    NetTensor *transpose(Chl axis1, Chl axis2);
    NetTensor *norm(int L_n);
    NetTensor *mean(Chl axis);
    NetTensor *where(float data);
    NetTensor *where(float data, Chl axis);

    /* Overload the operators.*/
    NetTensor *operator+(NetTensor* in_1) ;
    NetTensor *operator*(NetTensor* in_1) ;
    NetTensor *operator*(float muln) ;
    NetTensor *operator/(NetTensor* in_1) ;
    NetTensor *operator/(float devr) ;
} NetTensor;

} // namespace mllm

#endif // MLLM_NETPARAMETER_H