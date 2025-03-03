//
// Created by xwk on 25-3-2.
//
#include <iostream>

#include "kvcache/MemCache.hpp"
#include "kvcache/PoolManager.hpp"
#include "Req.hpp"

#include "cmdline.h"
#include "models/llama3/modeling_llama3.hpp"
#include "models/llama3/tokenization_llama3.hpp"
#include "processor/PostProcess.hpp"

void put_tokens(vector<int> &tokens, vector<token_id_t> &input_ids){
    tokens.clear();
    for (auto &id : input_ids){
        tokens.push_back((int)id);
    }
}

void put_values(Tensor &t, vector<int> values) {
    t.reshape(1, 1, values.size(), 1);
    for (int i = 0; i < values.size(); i++) {
        t.setDataAt<int>(0, 0, i, 0, values[i]);
    }
}

int main(int argc, char **argv){
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/llama3_tokenizer.model");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/llama-3.2-1b-instruct_q4_k.mllm");
    cmdParser.add<string>("billion", 'b', "[1B | 3B |]", false, "1B");
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 400);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string model_billion = cmdParser.get<string>("billion");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    Llama3Config config(400, model_billion);
    auto tokenizer = LLama3Tokenizer(vocab_path);
    config.cache_limit = tokens_limit;
    auto model = Llama3Model(config, true); // use_paged_attn = true
    model.load(model_path);

    SimplePoolManager pool(tokens_limit); // kv cache pool
    cache::RadixCache radix_cache;

    Tensor out_loc(1, 1, tokens_limit, 1, Backend::global_backends[MLLM_CPU], true);
    out_loc.setName("out_loc");
    out_loc.setTtype(INPUT_TENSOR);
    out_loc.setDtype(MLLM_TYPE_I32);
    out_loc.setModule(&model);

    Tensor kv_indices(1, 1, tokens_limit, 1, Backend::global_backends[MLLM_CPU], true);
    kv_indices.setName("kv_indices");
    kv_indices.setTtype(INPUT_TENSOR);
    kv_indices.setDtype(MLLM_TYPE_I32);
    kv_indices.setModule(&model);

    vector<Tensor> all_inputs;
    all_inputs.reserve(3 + config.block_num);  // input_ids, out_loc, kv_indices, kv_cache_1, kv_cache_2, ..., kv_cache_n
    all_inputs.resize(3);
    for(int i = 1; i <= config.block_num; i++){ // set kv_cache for each layer. shape: [max_token, 2, num_kv_heads, head_dim]
        Tensor kv_cache(tokens_limit, config.num_key_value_heads, 2, config.hidden_dim / config.head_size, Backend::global_backends[MLLM_CPU], true);
        kv_cache.setName("kv_cache_" + std::to_string(i));
        kv_cache.setTtype(INPUT_TENSOR);
        kv_cache.setDtype(MLLM_TYPE_F32);
        kv_cache.setModule(&model);
        all_inputs.push_back(kv_cache);
    }


    vector<string> user_prompts = {
        "Hello, who are you?",
        "Hello, who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."
    };

    for (auto &user_prompt : user_prompts) {
        std::cout << "User: " << user_prompt << std::endl;
        std::cout << "Bot: " << std::flush;
        Req req(tokens_limit, &pool);
        string prompt = tokenizer.apply_chat_template(user_prompt);

        auto tokens = tokenizer.tokenize_and_get_ids(prompt);
        put_tokens(req.tokens, tokens);
        // NOTE: here we should ensure that at least one token not in the shared prefix
        // otherwise the model has no input.
        // so we pop the last token and find the prefix match in radix cache and then append it back
        auto last_token = req.tokens.back();
        req.tokens.pop_back();
        radix_cache.matchPrefix(req.tokens, req.kv_indices); // find prefix match in radix cache
        model.set_position(req.kv_indices.size());  // NOTE: set rope position
        req.tokens.push_back(last_token);

        vector<token_id_t> input_ids;
        input_ids.reserve(tokens.size() - req.kv_indices.size());
        for (int i = req.kv_indices.size(); i < tokens.size(); i++) {
            input_ids.push_back(tokens[i]);
        }
        auto input_tensor = Tokenizer::tokens2Input(input_ids);

        for (int step = 0; step < 100; step++) {
            int num_slots_needed = req.tokens.size() - req.kv_indices.size();
            if (!pool.canAllocate(num_slots_needed)) {
                if (num_slots_needed > tokens_limit)
                    throw std::runtime_error("num_slots_needed > tokens_limit");
                radix_cache.evict(num_slots_needed - pool.availableSlots(), &pool);
            }
            auto out_loc_vec = pool.allocate(num_slots_needed);
            req.kv_indices.insert(req.kv_indices.end(), out_loc_vec.begin(),
                                  out_loc_vec.end());
            put_values(out_loc, out_loc_vec);
            put_values(kv_indices, req.kv_indices);

            // run model
            all_inputs[0] = input_tensor;
            all_inputs[1] = out_loc;
            all_inputs[2] = kv_indices;
            auto result = model(all_inputs);

            // post process
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { std::cout << "\n"; break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});

            // cache
            radix_cache.cacheReq(req);
            // append new token
            req.tokens.push_back(out_token);
        }
        printf("\n");
        model.profiling();
    }

    return 0;
}
