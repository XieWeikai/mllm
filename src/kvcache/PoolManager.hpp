//
// Created by xwk on 25-2-27.
//

#ifndef MLLM_POOLMANAGER_HPP
#define MLLM_POOLMANAGER_HPP

#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <algorithm> // for std::copy
#include <list>

class PoolManagerBase {
public:
    using MoveFunction = std::function<void(int from, int to)>;

    explicit PoolManagerBase(int size, MoveFunction move_func = nullptr)
        : pool_size(size), move_slot_func(std::move(move_func)) {}

    virtual ~PoolManagerBase() = default;

    // allocate needed_size slots, return the indices of allocated slots
    virtual std::vector<int> allocate(int needed_size) = 0;

    // deallocate slots
    virtual void deallocate(const std::vector<int>& slots) = 0;

    // check if needed_size slots are available
    virtual bool canAllocate(int needed_size) = 0;

    // number of available slots
    virtual int availableSlots() = 0;

    [[nodiscard]] virtual int size() const {
        return pool_size;
    }

    // 可选的内存碎片整理功能
    virtual bool defragment() {
        if (!move_slot_func) {
            throw std::runtime_error("Defragmentation not supported: move function not provided");
        }
        return false;
    }

protected:
    int pool_size = 0;
    MoveFunction move_slot_func;
};


class SimplePoolManager : public PoolManagerBase {
private:
    std::vector<int> available_slots;
    int current_size;

public:
    explicit SimplePoolManager(int size, MoveFunction move_func = nullptr);

    std::vector<int> allocate(int needed_size) override;

    void deallocate(const std::vector<int>& slots) override;

    bool canAllocate(int needed_size) override;

    int availableSlots() override;

    bool defragment() override;
};



class ContiguousPoolManager : public PoolManagerBase {
private:
    std::list<std::pair<int, int>> free_blocks; // [start, end) pairs

public:
    explicit ContiguousPoolManager(int size, MoveFunction move_func = nullptr);

    std::vector<int> allocate(int needed_size) override;
    void deallocate(const std::vector<int>& slots) override;
    bool canAllocate(int needed_size) override;
    int availableSlots() override = 0;
    bool defragment() override;
};

#endif // MLLM_POOLMANAGER_HPP
