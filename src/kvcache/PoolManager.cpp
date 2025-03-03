//
// Created by xwk on 25-2-27.
//
#include <algorithm>

#include "PoolManager.hpp"

SimplePoolManager::SimplePoolManager(int size, MoveFunction move_func)
    : PoolManagerBase(size, std::move(move_func)), available_slots(size), current_size(size) {
    // Initialize available slots from 0 to size-1
    for (int i = 0; i < size; ++i) {
        available_slots[i] = size - i - 1;
    }
}

std::vector<int> SimplePoolManager::allocate(int needed_size) {
    if (needed_size < 0) {
        throw std::invalid_argument("allocate(): needed_size cannot be negative");
    }
    if (needed_size == 0) {
        return {};
    }
    if (current_size < needed_size) {
        throw std::runtime_error("allocate(): Not enough slots available");
    }

    // Calculate start position and create result array
    const int used_slots = pool_size - current_size;
    std::vector<int> result(
        available_slots.rbegin() + used_slots,
        available_slots.rbegin() + used_slots + needed_size
    );

    current_size -= needed_size;
    return result;
}

void SimplePoolManager::deallocate(const std::vector<int>& slots) {
    const int slots_count = static_cast<int>(slots.size());
    if (slots_count == 0) return;

    // Check for out-of-bounds (total capacity fixed to pool_size)
    if (current_size + slots_count > available_slots.size()) {
        throw std::runtime_error("deallocate(): Exceeding pool capacity");
    }

    // Batch copy released slots to the end of the available list
    std::copy(
        slots.rbegin(),
        slots.rend(),
        available_slots.begin() + current_size
    );
    current_size += slots_count;
}

bool SimplePoolManager::canAllocate(int needed_size) {
    return current_size >= needed_size;
}

int SimplePoolManager::availableSlots() {
    return current_size;
}

bool SimplePoolManager::defragment() {
    // Directly call base class implementation (throws unsupported exception)
    return PoolManagerBase::defragment();
}


ContiguousPoolManager::ContiguousPoolManager(int size, MoveFunction move_func)
    : PoolManagerBase(size, std::move(move_func)) {
    free_blocks.emplace_back(0, size);
}

std::vector<int> ContiguousPoolManager::allocate(int needed_size) {
    if (needed_size < 0) throw std::invalid_argument("Negative allocation size");
    if (needed_size == 0) return {};

    // first fit
    for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
        int available = it->second - it->first;
        if (available >= needed_size) {
            // 分配连续空间
            std::vector<int> result;
            result.reserve(needed_size);
            for (int i = 0; i < needed_size; ++i) {
                result.push_back(it->first + i);
            }

            // update free blocks
            if (available > needed_size) {
                *it = {it->first + needed_size, it->second};
            } else {
                free_blocks.erase(it);
            }
            return result;
        }
    }
    throw std::runtime_error("Not enough contiguous space");
}

void ContiguousPoolManager::deallocate(const std::vector<int>& slots) {
    if (slots.empty()) return;

    std::vector<int> sorted(slots);
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::pair<int, int>> new_blocks;
    int start = sorted[0];
    int end = start + 1;

    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] == sorted[i-1] + 1) {
            end++;
        } else {
            new_blocks.emplace_back(start, end);
            start = sorted[i];
            end = start + 1;
        }
    }
    new_blocks.emplace_back(start, end);

    for (auto& block : new_blocks) {
        auto it = free_blocks.begin();
        while (it != free_blocks.end()) {
            if (block.second == it->first) {
                block.second = it->second;
                it = free_blocks.erase(it);
            } else if (it->second == block.first) {
                block.first = it->first;
                it = free_blocks.erase(it);
            } else {
                ++it;
            }
        }
        free_blocks.push_back(block);
    }

    free_blocks.sort([](auto& a, auto& b) { return a.first < b.first; });
}

bool ContiguousPoolManager::canAllocate(int needed_size) {
    if (needed_size <= 0) return true;
    for (const auto& block : free_blocks) {
        if (block.second - block.first >= needed_size) return true;
    }
    return false;
}

bool ContiguousPoolManager::defragment() {
    if (!move_slot_func) {
        throw std::runtime_error("Defragmentation requires move function");
    }

    int used = pool_size;
    for (const auto& block : free_blocks) {
        used -= (block.second - block.first);
    }

    int target_pos = 0;
    for (int i = 0; i < pool_size && target_pos < used; ++i) {
        bool is_free = false;
        for (const auto& block : free_blocks) {
            if (i >= block.first && i < block.second) {
                is_free = true;
                break;
            }
        }

        if (!is_free && i != target_pos) {
            move_slot_func(i, target_pos);
            target_pos++;
        } else if (!is_free) {
            target_pos++;
        }
    }

    free_blocks.clear();
    if (used < pool_size) {
        free_blocks.emplace_back(used, pool_size);
    }
    return true;
}
