#ifndef _DYNAMICPOOLALLOCATOR_HPP
#define _DYNAMICPOOLALLOCATOR_HPP

#include <cstddef>
#include <cassert>

#include "umpire/tpl/simpool/StdAllocator.hpp"
#include "umpire/tpl/simpool/FixedPoolAllocator.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"

template <class IA = StdAllocator>
class DynamicPoolAllocator
{
protected:
  struct Block
  {
    char *data;
    std::size_t size;
    bool isHead;
    Block *next;
  };

  // Allocator for the underlying data
  typedef FixedPoolAllocator<struct Block, IA, (1<<6)> BlockAlloc;
  BlockAlloc blockAllocator;

  // Start of the nodes of used and free block lists
  struct Block *usedBlocks;
  struct Block *freeBlocks;

  // Total size allocated (bytes)
  std::size_t totalBytes;

  // Allocated size (bytes)
  std::size_t allocBytes;

  // Minimum size for allocations
  std::size_t minBytes;

  // Pointer to our allocator's allocation strategy
  std::shared_ptr<umpire::strategy::AllocationStrategy> allocator;

  // Search the list of free blocks and return a usable one if that exists, else NULL
  void findUsableBlock(struct Block *&best, struct Block *&prev, std::size_t size) {
    best = prev = NULL;
    for ( struct Block *iter = freeBlocks, *iterPrev = NULL ; iter ; iter = iter->next ) {
      if ( iter->size >= size && (!best || iter->size < best->size) ) {
        best = iter;
        prev = iterPrev;
      }
      iterPrev = iter;
    }
  }

  inline std::size_t alignmentAdjust(const std::size_t size) {
    const std::size_t AlignmentBoundary = 16;
    return std::size_t (size + (AlignmentBoundary-1)) & ~(AlignmentBoundary-1);
  }

  // Allocate a new block and add it to the list of free blocks
  void allocateBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
    const std::size_t sizeToAlloc = std::max(alignmentAdjust(size), minBytes);
    curr = prev = NULL;
    void *data = NULL;

    // Allocate data
    data = allocator->allocate(sizeToAlloc);
    totalBytes += sizeToAlloc;
    assert(data);

    // Find next and prev such that next->data is still smaller than data (keep ordered)
    struct Block *next;
    for ( next = freeBlocks; next && next->data < data; next = next->next ) {
      prev = next;
    }

    // Allocate the block
    curr = (struct Block *) blockAllocator.allocate();
    if (!curr) return;
    curr->data = static_cast<char *>(data);
    curr->size = sizeToAlloc;
    curr->isHead = true;
    curr->next = next;

    // Insert
    if (prev) prev->next = curr;
    else freeBlocks = curr;
  }

  void splitBlock(struct Block *&curr, struct Block *&prev, const std::size_t size) {
    struct Block *next;
    const std::size_t alignedsize = alignmentAdjust(size);

    if ( curr->size == size || curr->size == alignedsize ) {
      // Keep it
      next = curr->next;
    }
    else {
      // Split the block
      std::size_t remaining = curr->size - alignedsize;
      struct Block *newBlock = (struct Block *) blockAllocator.allocate();
      if (!newBlock) return;
      newBlock->data = curr->data + alignedsize;
      newBlock->size = remaining;
      newBlock->isHead = false;
      newBlock->next = curr->next;
      next = newBlock;
      curr->size = alignedsize;
    }

    if (prev) prev->next = next;
    else freeBlocks = next;
  }

  void releaseBlock(struct Block *curr, struct Block *prev) {
    assert(curr != NULL);

    if (prev) prev->next = curr->next;
    else usedBlocks = curr->next;

    // Find location to put this block in the freeBlocks list
    prev = NULL;
    for ( struct Block *temp = freeBlocks ; temp && temp->data < curr->data ; temp = temp->next ) {
      prev = temp;
    }

    // Keep track of the successor
    struct Block *next = prev ? prev->next : freeBlocks;

    // Check if prev and curr can be merged
    if ( prev && prev->data + prev->size == curr->data && !curr->isHead ) {
      prev->size = prev->size + curr->size;
      blockAllocator.deallocate(curr); // keep data
      curr = prev;
    }
    else if (prev) {
      prev->next = curr;
    }
    else {
      freeBlocks = curr;
    }

    // Check if curr and next can be merged
    if ( next && curr->data + curr->size == next->data && !next->isHead ) {
      curr->size = curr->size + next->size;
      curr->next = next->next;
      blockAllocator.deallocate(next); // keep data
    }
    else {
      curr->next = next;
    }
  }

  void freeAllBlocks() {
    // Release the used blocks
    while(usedBlocks) {
      releaseBlock(usedBlocks, NULL);
    }

    // Release the unused blocks
    while(freeBlocks) {
      assert(freeBlocks->isHead);
      allocator->deallocate(freeBlocks->data);
      totalBytes -= freeBlocks->size;
      struct Block *curr = freeBlocks;
      freeBlocks = freeBlocks->next;
      blockAllocator.deallocate(curr);
    }
  }

public:
  DynamicPoolAllocator(
      std::shared_ptr<umpire::strategy::AllocationStrategy> strat,
      const std::size_t _minBytes = (1 << 8))
    : blockAllocator(),
      usedBlocks(NULL),
      freeBlocks(NULL),
      totalBytes(0),
      allocBytes(0),
      minBytes(_minBytes),
      allocator(strat) { }

  ~DynamicPoolAllocator() { freeAllBlocks(); }

  void *allocate(std::size_t size) {
    struct Block *best, *prev;
    findUsableBlock(best, prev, size);

    // Allocate a block if needed
    if (!best) allocateBlock(best, prev, size);
    assert(best);

    // Split the free block
    splitBlock(best, prev, size);

    // Push node to the list of used nodes
    best->next = usedBlocks;
    usedBlocks = best;

    // Increment the allocated size
    allocBytes += size;

    // Return the new pointer
    return usedBlocks->data;
  }

  void deallocate(void *ptr) {
    assert(ptr);

    // Find the associated block
    struct Block *curr = usedBlocks, *prev = NULL;
    for ( ; curr && curr->data != ptr; curr = curr->next ) {
      prev = curr;
    }
    if (!curr) return;

    // Remove from allocBytes
    allocBytes -= curr->size;

    // Release it
    releaseBlock(curr, prev);
  }

  std::size_t allocatedSize() const { return allocBytes; }

  std::size_t totalSize() const {
    return totalBytes + blockAllocator.totalSize();
  }

  std::size_t numFreeBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = freeBlocks; temp; temp = temp->next) nb++;
    return nb;
  }

  std::size_t numUsedBlocks() const {
    std::size_t nb = 0;
    for (struct Block *temp = usedBlocks; temp; temp = temp->next) nb++;
    return nb;
  }
};

#endif
