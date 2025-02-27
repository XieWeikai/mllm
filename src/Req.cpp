//
// Created by xwk on 25-2-27.
//
#include "Req.hpp"

int Req::reqCnt = 0;

Req::Req(int context_size, PoolManagerBase *pool_manager) {
    id = reqCnt++;
    tokens.reserve(context_size);
    kv_indices.reserve(context_size);
    this->pool_manager = pool_manager;
}

