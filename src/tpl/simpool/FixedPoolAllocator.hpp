#ifndef _FIXEDPOOLALLOCATOR_HPP
#define _FIXEDPOOLALLOCATOR_HPP

#include <strings.h>
#include <iostream>

template<class T, class MA, int NP=(1<<6)>
class FixedPoolAllocator
{
protected:
  struct Pool
  {
    unsigned char *data;
    unsigned int *avail;
    unsigned int numAvail;
    struct Pool* next;
  };

  struct Pool *pool;
  const std::size_t numPerPool;
  const std::size_t totalPoolSize;

  std::size_t numBlocks;

  void newPool(struct Pool **pnew) {
    struct Pool *p = static_cast<struct Pool *>(MA::allocate(totalPoolSize));
    p->numAvail = numPerPool;
    p->next = NULL;

    p->data  = reinterpret_cast<unsigned char *>(p) + sizeof(struct Pool);
    p->avail = reinterpret_cast<unsigned int *>(p->data + numPerPool * sizeof(T));
    for (int i = 0; i < NP; i++) p->avail[i] = (unsigned int)-1;

    *pnew = p;
  }

  T* allocInPool(struct Pool *p) {
    if (!p->numAvail) return NULL;

    for (int i = 0; i < NP; i++) {
      int bit = ffs(p->avail[i]) - 1;
      if (bit >= 0) {
        p->avail[i] ^= 1 << bit;
        p->numAvail--;
        int entry = i * sizeof(unsigned int) * 8 + bit;
        return reinterpret_cast<T*>(p->data) + entry;
      }
    }

    return NULL;
  }

public:
  static inline FixedPoolAllocator &getInstance() {
    static FixedPoolAllocator instance;
    return instance;
  }

  FixedPoolAllocator()
    : numPerPool(NP * sizeof(unsigned int) * 8),
      totalPoolSize(sizeof(struct Pool) +
                    numPerPool * sizeof(T) + NP * sizeof(unsigned int)),
      numBlocks(0)
  { newPool(&pool); }

  ~FixedPoolAllocator() {
    for (struct Pool *curr = pool; curr; ) {
      struct Pool *next = curr->next;
      MA::deallocate(curr);
      curr = next;
    }
  }

  T* allocate() {
    T* ptr = NULL;

    struct Pool *prev = NULL;
    struct Pool *curr = pool;
    while (!ptr && curr) {
      ptr = allocInPool(curr);
      prev = curr;
      curr = curr->next;
    }

    if (!ptr) {
      newPool(&prev->next);
      ptr = allocate();
      // TODO: In this case we should reverse the linked list for optimality
    }
    else {
      numBlocks++;
    }
    return ptr;
  }

  void deallocate(T* ptr) {
    int i = 0;
    for (struct Pool *curr = pool; curr; curr = curr->next) {
      const T* start = reinterpret_cast<T*>(curr->data);
      const T* end   = reinterpret_cast<T*>(curr->data) + numPerPool;
      if ( (ptr >= start) && (ptr < end) ) {
        // indexes bits 0 - numPerPool-1
        const int indexD = ptr - reinterpret_cast<T*>(curr->data);
        const int indexI = indexD / sizeof(unsigned int) / 8;
        const int indexB = indexD % ( sizeof(unsigned int) * 8 );
        // TODO - The check below appears to be buggy.  I believe that the
        // bit should be "0" if it was previously allocated.  The check seems
        // to be checking for "1".
//#ifdef DEBUG
        //if (!((curr->avail[indexI] >> indexB) & 1))
          //std::cerr << "Trying to deallocate an entry that was not marked as allocated"
                    //<< std::endl;
//#endif
        curr->avail[indexI] ^= 1 << indexB;
        curr->numAvail++;
        numBlocks--;
        return;
      }
      i++;
    }
    std::cerr << "Could not find pointer to deallocate" << std::endl;
    throw(std::bad_alloc());
  }

  /// Return allocated size to user.
  std::size_t allocatedSize() const { return numBlocks * sizeof(T); }

  /// Return total size with internal overhead.
  std::size_t totalSize() const {
    return numPools() * totalPoolSize;
  }

  /// Return the number of pools
  std::size_t numPools() const {
    std::size_t np = 0;
    for (struct Pool *curr = pool; curr; curr = curr->next) np++;
    return np;
  }

  /// Return the pool size
  std::size_t poolSize() const { return totalPoolSize; }
};


#endif // _FIXEDPOOLALLOCATOR_HPP
