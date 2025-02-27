//
// Created by xwk on 25-2-27.
//

#ifndef MLLM_MEMCACHE_HPP
#define MLLM_MEMCACHE_HPP

#include <vector>
#include <unordered_map>
#include <memory>

#include "Req.hpp"

namespace cache {
// radix tree node
typedef struct TreeNode {
    // token ids of a prefix
    std::vector<int> key;
    // kv indices of a prefix
    std::vector<int> value;

    // how many unfinished requests are using this node
    // if > 0 then the node should not be deleted
    int lock_ref;

    // children of the node
    std::unordered_map<int, std::unique_ptr<TreeNode>> children;

    // parent
    struct TreeNode *parent;

    struct LRUNode *lru_node;
} TreeNode;

void pretty_print(TreeNode *node);

std::string toString(TreeNode *node);

void tree_to_string(TreeNode *node, std::ostringstream &oss, const std::string &prefix = "", bool is_last = true);

int _add_lock_ref(TreeNode *node, int delta);

int _insert(TreeNode *node, const std::vector<int> &key, const std::vector<int> &value);

TreeNode *_match_prefix(TreeNode *node, const std::vector<int> &key,
                        std::vector<int> &result);

TreeNode *_split_node(TreeNode *child, int split_len);

typedef struct LRUNode {
    void *data;

    bool evictable;

    // global double linked list
    // all nodes are linked in this list
    struct LRUNode *prev;
    struct LRUNode *next;

    // evictable double linked list
    // only evictable nodes are linked in this list
    struct LRUNode *evictable_prev;
    struct LRUNode *evictable_next;

    // only non-evictable nodes have this field
    // points to the next evictable node
    // so that we know how to change non-evictable nodes to evictable nodes
    struct LRUNode *next_evictable;
} LRUNode;

class LRU {
public:
    explicit LRU();

    ~LRU();

    static LRUNode *newNode(void *data, bool evictable);

    void globalPutAfter(LRUNode *node, LRUNode *new_node);

    void globalRemove(LRUNode *node);

    void evictablePutAfter(LRUNode *node, LRUNode *new_node);

    void evictableRemove(LRUNode *node);

    void globalPutBefore(LRUNode *node, LRUNode *new_node);

    void evictablePutBefore(LRUNode *node, LRUNode *new_node);

    void saveDot(const char *filename, const std::string &direction = "TB");

    void putFront(LRUNode *node);

    void touch(LRUNode *node); // access a node

    void setEvictable(LRUNode *node, bool evictable);

    void *evict();

    friend class RadixCache;
private:
    // head and tail of the global double linked list and evictable double linked list
    // two lists share the same head and tail
    LRUNode *head, *tail;
};

class RadixCache {
public:
    explicit RadixCache();

    TreeNode *matchPrefix(const std::vector<int> &key, std::vector<int> &result);

    int insert(const std::vector<int> &key, const std::vector<int> &value);

    int addLockRef(TreeNode *node, int delta);

    void deleteLeaf(TreeNode *node);

    void cacheReq(Req &req);

    void cacheUnfinishedReq(Req &req);

    void cacheFinishedReq(Req &req);

    void evict(int num);

    void prettyPrint();

    void saveDot(const char* filename);
private:
    TreeNode *_split_node(TreeNode *child, int split_len);

    TreeNode *_matchPrefix(TreeNode *node, const int *key_l, const int *key_r, std::vector<int> &result);

    int _insert(TreeNode *node, const int *key_l, const int *key_r, const int *value_l, const int *value_r);

    // root node
    std::unique_ptr<TreeNode> root; // key and value are empty

    // LRU cache to manage Radix tree nodes
    std::unique_ptr<LRU> lru;
};
}

#endif // MLLM_MEMCACHE_HPP
