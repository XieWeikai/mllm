#ifndef REQ_H
#define REQ_H

#include <string>
#include <vector>

#include "kvcache/PoolManager.hpp"

// currently there will be no batching on edge side
// so a simple Req as follows is enough
class Req {
public:
    Req(int context_size, PoolManagerBase *pool_manager);

    static int reqCnt;
    int id;

    std::string prompt; // original prompt

    // all tokens, including the prompt and generated tokens
    std::vector<int> tokens;

    std::vector<int> kv_indices;

    PoolManagerBase *pool_manager;
};

#endif //REQ_H