//
// Created by xwk on 25-2-27.
//

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <assert.h>

#include "MemCache.hpp"

namespace cache {

TreeNode *_split_node(TreeNode *child, int split_len) {
    auto new_node = std::make_unique<TreeNode>();

    new_node->key = std::vector<int>(child->key.begin(), child->key.begin() + split_len);
    new_node->value = std::vector<int>(child->value.begin(), child->value.begin() + split_len);
    new_node->lock_ref = child->lock_ref;

    child->key = std::vector<int>(child->key.begin() + split_len, child->key.end());
    child->value = std::vector<int>(child->value.begin() + split_len, child->value.end());

    auto parent = child->parent;
    new_node->parent = parent;

    auto new_node_key = new_node->key[0];
    auto child_key = child->key[0];

    new_node->children[child_key] = std::move(parent->children[new_node_key]);
    child->parent = new_node.get();

    parent->children[new_node_key] = std::move(new_node);

    return parent->children[new_node_key].get();
}

TreeNode *RadixCache::_split_node(TreeNode *child, int split_len) {
    auto new_node = std::make_unique<TreeNode>();

    new_node->key = std::vector<int>(child->key.begin(), child->key.begin() + split_len);
    new_node->value = std::vector<int>(child->value.begin(), child->value.begin() + split_len);
    new_node->lock_ref = child->lock_ref;

    auto lru_node = LRU::newNode(new_node.get(), false); // not leaf node, so not evictable
    new_node->lru_node = lru_node;
    lru->putFront(lru_node);

    child->key = std::vector<int>(child->key.begin() + split_len, child->key.end());
    child->value = std::vector<int>(child->value.begin() + split_len, child->value.end());

    auto parent = child->parent;
    new_node->parent = parent;

    auto new_node_key = new_node->key[0];
    auto child_key = child->key[0];

    new_node->children[child_key] = std::move(parent->children[new_node_key]);
    child->parent = new_node.get();

    parent->children[new_node_key] = std::move(new_node);

    return parent->children[new_node_key].get();
}

static int _key_match(const int *k1_l, const int *k1_r, const int *k2_l, const int *k2_r) {
    int i = 0;
    while (k1_l < k1_r && k2_l < k2_r && *k1_l == *k2_l) {
        i++;
        k1_l++;
        k2_l++;
    }
    return i;
}

TreeNode *_match_prefix(TreeNode *node, const int *key_l, const int *key_r, std::vector<int> &result) {
    if (key_l == key_r) {
        return node;
    }

    int id = *key_l;
    auto it = node->children.find(id);

    if (it == node->children.end()) {
        return node;
    }

    TreeNode *child = it->second.get();

    auto prefix_len = _key_match(key_l, key_r, child->key.data(), child->key.data() + child->key.size());

    if (prefix_len < child->key.size()) {
        auto new_node = _split_node(child, prefix_len);
        result.insert(result.end(), new_node->value.begin(), new_node->value.end());
        return new_node;
    } else {
        result.insert(result.end(), child->value.begin(), child->value.end());
        return _match_prefix(child, key_l + prefix_len, key_r, result);
    }
}

TreeNode *RadixCache::_matchPrefix(TreeNode *node, const int *key_l, const int *key_r, std::vector<int> &result) {
    lru->touch(node->lru_node);

    if (key_l == key_r) {
        return node;
    }

    int id = *key_l;
    auto it = node->children.find(id);

    if (it == node->children.end()) {
        return node;
    }

    TreeNode *child = it->second.get();

    auto prefix_len = _key_match(key_l, key_r, child->key.data(), child->key.data() + child->key.size());

    if (prefix_len < child->key.size()) {
        auto new_node = this->_split_node(child, prefix_len);
        result.insert(result.end(), new_node->value.begin(), new_node->value.end());
        return new_node;
    } else {
        result.insert(result.end(), child->value.begin(), child->value.end());
        return _matchPrefix(child, key_l + prefix_len, key_r, result);
    }
}

TreeNode *_match_prefix(TreeNode *node, const std::vector<int> &key,
                        std::vector<int> &result) {
    return _match_prefix(node, key.data(), key.data() + key.size(), result);
}

TreeNode *RadixCache::matchPrefix(const std::vector<int> &key, std::vector<int> &result) {
    return _matchPrefix(root.get(), key.data(), key.data() + key.size(), result);
}

int _insert(TreeNode *node, const int *key_l, const int *key_r, const int *value_l, const int *value_r) {
    // no key to insert
    if (key_l == key_r) {
        return 0;
    }

    int id = *key_l;
    auto it = node->children.find(id);

    if (it == node->children.end()) {
        auto new_node = std::make_unique<TreeNode>();
        new_node->key = std::vector<int>(key_l, key_r);
        new_node->value = std::vector<int>(value_l, value_r);
        new_node->parent = node;
        new_node->lock_ref = 0;

        node->children[id] = std::move(new_node);
        return 0;
    }

    TreeNode *child = it->second.get();

    auto prefix_len = _key_match(key_l, key_r, child->key.data(), child->key.data() + child->key.size());

    if (prefix_len < child->key.size()) {
        //        auto new_node = _split_node(child, prefix_len);
        //        return prefix_len + _insert(new_node, key_l + prefix_len, key_r, value_l + prefix_len, value_r);
        auto new_node = _split_node(child, prefix_len);

        auto remaining_key_len = (key_r - key_l) - prefix_len;
        if (remaining_key_len > 0) {
            auto new_child = std::make_unique<TreeNode>();
            new_child->key = std::vector<int>(key_l + prefix_len, key_r);
            new_child->value = std::vector<int>(value_l + prefix_len, value_r);
            new_child->parent = new_node;
            new_child->lock_ref = 0;

            int new_child_id = new_child->key[0];
            new_node->children[new_child_id] = std::move(new_child);
        }

        return prefix_len;
    } else {
        if (prefix_len == (key_r - key_l)) {
            return prefix_len;
        }
        return prefix_len + _insert(child, key_l + prefix_len, key_r, value_l + prefix_len, value_r);
    }
}

int RadixCache::_insert(TreeNode *node, const int *key_l, const int *key_r, const int *value_l, const int *value_r) {
    lru->touch(node->lru_node);

    // no key to insert
    if (key_l == key_r) {
        return 0;
    }

    int id = *key_l;
    auto it = node->children.find(id);

    if (it == node->children.end()) {
        auto new_node = std::make_unique<TreeNode>();
        new_node->key = std::vector<int>(key_l, key_r);
        new_node->value = std::vector<int>(value_l, value_r);
        new_node->parent = node;
        new_node->lock_ref = 0;

        auto lru_node = LRU::newNode(new_node.get(), true);
        new_node->lru_node = lru_node;
        lru->putFront(lru_node);

        node->children[id] = std::move(new_node);
        lru->setEvictable(node->lru_node, false); // not leaf anymore not evictable
        return 0;
    }

    TreeNode *child = it->second.get();

    auto prefix_len = _key_match(key_l, key_r, child->key.data(), child->key.data() + child->key.size());

    if (prefix_len < child->key.size()) {
        //        auto new_node = _split_node(child, prefix_len);
        //        return prefix_len + _insert(new_node, key_l + prefix_len, key_r, value_l + prefix_len, value_r);
        auto new_node = _split_node(child, prefix_len);

        auto remaining_key_len = (key_r - key_l) - prefix_len;
        if (remaining_key_len > 0) {
            auto new_child = std::make_unique<TreeNode>();
            new_child->key = std::vector<int>(key_l + prefix_len, key_r);
            new_child->value = std::vector<int>(value_l + prefix_len, value_r);
            new_child->parent = new_node;
            new_child->lock_ref = 0;

            auto lru_node = LRU::newNode(new_child.get(), true);
            new_child->lru_node = lru_node;
            lru->putFront(lru_node);

            int new_child_id = new_child->key[0];
            new_node->children[new_child_id] = std::move(new_child);
            lru->setEvictable(new_node->lru_node, false); // not leaf anymore not evictable
        }

        return prefix_len;
    } else {
        if (prefix_len == (key_r - key_l)) {
            return prefix_len;
        }
        return prefix_len + _insert(child, key_l + prefix_len, key_r, value_l + prefix_len, value_r);
    }
}

int _insert(TreeNode *node, const std::vector<int> &key, const std::vector<int> &value) {
    return _insert(node, key.data(), key.data() + key.size(), value.data(), value.data() + value.size());
}

int RadixCache::insert(const std::vector<int> &key, const std::vector<int> &value) {
    return _insert(root.get(), key.data(), key.data() + key.size(), value.data(), value.data() + value.size());
}

int _add_lock_ref(TreeNode *node, int delta) {
    int num_changed = 0;

    while (node->parent != nullptr) { // not root
        node->lock_ref += delta;
        node = node->parent;
        num_changed++;
    }
    return num_changed;
}

int RadixCache::addLockRef(TreeNode *node, int delta) {
    int num_changed = 0;

    if (delta > 0) {
        lru->setEvictable(node->lru_node, false);  // not evictable
    } else { // < 0, assume delta will not be 0
        // lock_ref is 0 and this node is leaf node
        // then set evictable
        if (node->lock_ref + delta == 0 && node->children.empty()) {
            lru->setEvictable(node->lru_node, true);  // evictable
        }
    }


    while (node->parent != nullptr) { // not root
        node->lock_ref += delta;
        node = node->parent;
        num_changed++;
    }
    return num_changed;
}

void _delete_leaf(TreeNode *node) {
    if (node->parent == nullptr) {
        return;
    }

    auto parent = node->parent;
    parent->children.erase(node->key[0]);
}

// this function will only be called when the node is evited by LRU
// means the node is not in LRU
// and caller should make sure the node is not in LRU and the node is leaf node and lock_ref is 0
// this should be called after lru.evict()
void RadixCache::deleteLeaf(TreeNode *node) {
    if (node->parent == nullptr) {
        return;
    }

    auto parent = node->parent;
    parent->children.erase(node->key[0]);

    if (parent->children.empty() && parent->lock_ref == 0) {
        lru->setEvictable(parent->lru_node, true);
    }
}

void RadixCache::cacheReq(Req &req) {
    assert(req.tokens.size() == req.kv_indices.size());
    insert(req.tokens, req.kv_indices);
}

void RadixCache::cacheUnfinishedReq(Req &req) {
    throw std::runtime_error("Not implemented yet");
}

void RadixCache::cacheFinishedReq(Req &req) {
    throw std::runtime_error("Not implemented yet");
}

void tree_to_string(TreeNode *node, std::ostringstream &oss, const std::string &prefix, bool is_last) {
    oss << prefix;
    oss << (is_last ? "└── " : "├── ");

    oss << "Key: [";
    for (size_t i = 0; i < node->key.size(); ++i) {
        oss << node->key[i];
        if (i < node->key.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";

    oss << "  Value: [";
    for (size_t i = 0; i < node->value.size(); ++i) {
        oss << node->value[i];
        if (i < node->value.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";

    oss << "  Lock Ref: " << node->lock_ref << "\n";

    for (auto it = node->children.begin(); it != node->children.end();) {
        auto next_it = std::next(it);
        bool is_last_child = (next_it == node->children.end());
        //        printf("id: %d node: %p\n", it->first, it->second.get());
        tree_to_string(it->second.get(), oss, prefix + (is_last ? "    " : "│   "), is_last_child);
        it = next_it;
    }
}

std::string toString(TreeNode *node) {
    std::ostringstream oss;
    tree_to_string(node, oss);
    return oss.str();
}

void pretty_print(TreeNode *node) {
    std::string tree_str = toString(node);
    std::cout << "Radix Tree Structure:\n";
    std::cout << tree_str;
}

void RadixCache::prettyPrint() {
    std::string tree_str = toString(root.get());
    std::cout << "Radix Tree Structure:\n";
    std::cout << tree_str;
}

RadixCache::RadixCache() {
    // init root node
    root = std::make_unique<TreeNode>();
    root->parent = nullptr;
    root->lock_ref = 1;  // root node should not be deleted

    // init LRU cache
    lru = std::make_unique<LRU>();
    auto lru_node = LRU::newNode(root.get(), false); // not evictable
    root->lru_node = lru_node;
    lru->putFront(lru_node);
}

LRU::~LRU() {
    auto node = head;
    while (node != nullptr) {
        auto next = node->next;
        delete node;
        node = next;
    }
}

LRU::LRU() {
    // null head and tail
    head = new LRUNode();
    tail = new LRUNode();
    head->next = tail;
    tail->prev = head;
    head->prev = tail->next = nullptr;
    head->evictable_next = tail;
    tail->evictable_prev = head;
    head->evictable_prev = tail->evictable_next = nullptr;
    head->next_evictable = nullptr;
    tail->next_evictable = nullptr;
    // in evictable list, head and tail are always there
    head->evictable = tail->evictable = true;
}

LRUNode *LRU::newNode(void *data, bool evictable) {
    auto node = new LRUNode();
    node->data = data;
    node->evictable = evictable;

    node->next = node->prev = nullptr;
    node->evictable_next = node->evictable_prev = nullptr;
    node->next_evictable = nullptr;

    return node;
}

// should before a node putting in evictable list
void LRU::globalPutAfter(LRUNode *node, LRUNode *new_node) {
    auto next_evictable = node->next->evictable ? node->next : node->next->next_evictable;
    new_node->next_evictable = next_evictable;

    new_node->prev = node;
    new_node->next = node->next;
    node->next->prev = new_node;
    node->next = new_node;
}

// should after a node removing in evictable list
void LRU::globalRemove(LRUNode *node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;

    node->prev = node->next = nullptr;
    node->next_evictable = nullptr;
}

// should after a node putting in global list
void LRU::evictablePutAfter(LRUNode *node, LRUNode *new_node) {
    auto p = node->next;
    while (p != new_node) {
        p->next_evictable = new_node;
        p = p->next;
    }

    new_node->evictable_prev = node;
    new_node->evictable_next = node->evictable_next;
    node->evictable_next->evictable_prev = new_node;
    node->evictable_next = new_node;

    new_node->next_evictable = nullptr;
}

// should before a node removing in global list
void LRU::evictableRemove(LRUNode *node) {
    auto p = node->evictable_prev->next;
    while (p != node) {
        p->next_evictable = node->evictable_next;
        p = p->next;
    }
    node->evictable_prev->evictable_next = node->evictable_next;
    node->evictable_next->evictable_prev = node->evictable_prev;

    node->next_evictable = node->evictable_next;
    node->evictable_prev = node->evictable_next = nullptr;
}

// should before a node putting in evictable list
void LRU::globalPutBefore(LRUNode *node, LRUNode *new_node) {
    auto p = node->prev;
    globalPutAfter(p, new_node);
}

// should after a node putting in global list
void LRU::evictablePutBefore(LRUNode *node, LRUNode *new_node) {
    auto p = node->evictable_prev;
    evictablePutAfter(p, new_node);
}

void LRU::putFront(LRUNode *node) {
    globalPutAfter(head, node);
    if (node->evictable) {
        evictablePutAfter(head, node);
    }
}

void LRU::touch(LRUNode *node) {
    if (node->evictable) {
        evictableRemove(node);
    }
    globalRemove(node);
    putFront(node);
}

void LRU::setEvictable(LRUNode *node, bool evictable) {
    if (node->evictable == evictable) { // no need to change
        return;
    }

    node->evictable = evictable;
    if (evictable) {
        evictablePutBefore(node->next_evictable, node);
    } else {
        evictableRemove(node);
    }
}

void *LRU::evict() {
    if (head->evictable_next == tail) { // nothing to evict
        return nullptr;
    }

    auto evict_node = tail->evictable_prev;
    void *data = evict_node->data;

    evictableRemove(evict_node);
    globalRemove(evict_node);

    delete evict_node;
    return data;
}

void LRU::saveDot(const char *filename, const std::string &direction) {
    std::ofstream out(filename);
    out << "digraph LRU {\n";
    out << "  rankdir=" << direction << ";\n";  // 方向参数控制布局
    out << "  node [shape=record, fontname=\"Arial\"];\n";
    out << "  edge [];\n\n";

    // 定义全局链表和evictable链表的关系
    LRUNode *current = head;
    while (current) {
        // 节点基本属性（不再显示evictable值）
        auto label = std::to_string((unsigned long) current->data);
        if (current == head) {
            label = "HEAD";
        } else if (current == tail) {
            label = "TAIL";
        }
        out << "  node" << current << " [label=\"{" << label << "}\"";
        if (current->evictable) {
            out << " color=\"red\" fontcolor=\"red\"";
        }
        out << "];\n";

        // 全局链表连接
        if (current->next) {
            out << "  node" << current << " -> node" << current->next
                << " [color=\"blue\", label=\"next\"];\n";
            out << "  node" << current->next << " -> node" << current
                << " [color=\"blue\", label=\"prev\"];\n";
        }

        // Evictable链表连接
        if (current->evictable) {
            if (current->evictable_next) {
                out << "  node" << current << " -> node" << current->evictable_next
                    << " [color=\"red\", style=dashed, label=\"e_next\"];\n";
                out << "  node" << current->evictable_next << " -> node" << current
                    << " [color=\"red\", style=dashed, label=\"e_prev\"];\n";
            }
        }

        // Next evictable指针
        if (!current->evictable && current->next_evictable) {
            out << "  node" << current << " -> node" << current->next_evictable
                << " [color=\"green\", style=dotted, label=\"next_e\"];\n";
        }

        current = current->next;
    }

    out << "}\n";
}

void RadixCache::saveDot(const char* filename) {
    std::ofstream out(filename);
    out << "digraph RadixCache {\n"
        << "  rankdir=TB;\n"
        << "  node [shape=Mrecord, fontname=\"Courier New\"];\n"
        << "  edge [];\n\n";

    // 生成树结构
    out << "  subgraph cluster_tree {\n"
        << "    label=\"Radix Tree Structure\";\n"
        << "    style=filled;\n"
        << "    color=lightgrey;\n";

    std::function<void(TreeNode*, TreeNode *)> traverseTree = [&](TreeNode *node, TreeNode *parent) {
        // 生成树节点
        out << "    tree_" << node << " [label=\"";
        out << "Key: [";
        for(size_t i=0; i<node->key.size(); ++i) {
            if(i>0) out << ", ";
            out << node->key[i];
        }
        out << "]";
        out << "\\nLock: " << node->lock_ref;
        out << "\"";
        if(node->children.empty()) out << " shape=box3d";
        out << "];\n";

        // 连接父节点
        if(parent) {
            out << "    tree_" << parent << " -> tree_" << node
                << ";\n";
        }

        // 递归处理子节点
        for(auto& child : node->children) {
            traverseTree(child.second.get(), node);
        }
    };

    traverseTree(root.get(), nullptr);
    out << "  }\n\n";

    // 生成LRU链表
    out << "  subgraph cluster_lru {\n"
        << "    label=\"LRU Structure\";\n"
        << "    style=filled;\n"
        << "    color=lightblue;\n";

    LRUNode *current = lru->head;
    while (current) {
        auto treeNode = static_cast<TreeNode*>(current->data);

        if (treeNode) {
            out << "    tree_" << treeNode << " -> node" << current
                << " [style=dashed, color=gray];\n";
        }

        // 节点基本属性（不再显示evictable值）
        auto label = "";
        if (current == lru->head) {
            label = "HEAD";
        } else if (current == lru->tail) {
            label = "TAIL";
        }
        out << "  node" << current << " [label=\"{" << label << "}\"";
        if (current->evictable) {
            out << " color=\"red\" fontcolor=\"red\"";
        }
        out << "];\n";

        // 全局链表连接
        if (current->next) {
            out << "  node" << current << " -> node" << current->next
                << " [color=\"blue\", label=\"next\"];\n";
            out << "  node" << current->next << " -> node" << current
                << " [color=\"blue\", label=\"prev\"];\n";
        }

        // Evictable链表连接
        if (current->evictable) {
            if (current->evictable_next) {
                out << "  node" << current << " -> node" << current->evictable_next
                    << " [color=\"red\", style=dashed, label=\"e_next\"];\n";
                out << "  node" << current->evictable_next << " -> node" << current
                    << " [color=\"red\", style=dashed, label=\"e_prev\"];\n";
            }
        }

        // Next evictable指针
        if (!current->evictable && current->next_evictable) {
            out << "  node" << current << " -> node" << current->next_evictable
                << " [color=\"green\", style=dotted, label=\"next_e\"];\n";
        }

        current = current->next;
    }
    out << "  }\n";

    out << "}\n";
}

int RadixCache::evict(int num, PoolManagerBase *pool) {
    int slots_freed = 0;
    while (slots_freed < num) {
        auto data = lru->evict();
        if (data == nullptr) { // no evictable node in LRU
            break;
        }
        auto node = static_cast<TreeNode *>(data);
        slots_freed += node->value.size();
        pool->deallocate(node->value); // release slots in pool
        deleteLeaf(node);
    }
    return slots_freed;
}
}
