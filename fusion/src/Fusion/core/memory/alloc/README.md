# Fusion Alloc Layer 
## Purpose 
The alloc layer provides the raw memory management foundation for Fusion. Its responsibilities are to:
- acquire raw memory regions from backend sub-allocators 
- manage reuse of those regions efficiently 
- support multiple allocation strategies for different workload lifetimes
- remain independent of tensor semantics, physics semantics, and kernel logic This layer should allocate bytes, not own layout meaning.

## Current design 

At the moment, the alloc stack is roughly: 
```
Storage / Tensor / Physics
        |
        v
   PoolAllocator
        |
        v
   CPUSubAllocator
        |
        v
 OS / heap allocation
``` 

### Current components 
- `ISubAllocator` - backend interface for acquiring raw memory regions 
- `CPUSubAllocator` - CPU implementation of the sub-allocator backend 
- `PoolAllocator` / `BFCPoolAllocator` - pooled split/coalesce allocator over acquired regions - manages chunks, free buckets, and coalescing 
- `ArenaAllocator` - intended scratch / temporary allocator path 

### Current behavior 
- requests are rounded to power-of-two chunk sizes 
- larger free chunks may be split 
- adjacent free chunks may be coalesced 
- free chunks are tracked in size buckets 
- backing regions are acquired from a sub-allocator 

### Current limitation The current growth strategy is conservative. 
Repeated small allocations may create multiple small root regions instead of being sourced from a larger reusable slab. That is simple, 
but not ideal for: 
- locality 
- reuse 
- fragmentation resistance 
- realistic mixed physics + DL workloads 

## Current pooled allocator invariants 
Useful working invariants for `PoolAllocator`: 
- `chunk_id` is expected to match the index in `chunks_` 
- `size == 0` means inactive / deleted chunk
- inactive chunks must satisfy: 
  - `ptr == nullptr` 
  - `requested_size == 0` 
  - `in_use == false` 
  - `prev == kInvalidChunkID` 
  - `next == kInvalidChunkID` 
  - non-deleted chunks must have `ptr != nullptr` 
- free chunks must satisfy: 
  - `in_use == false` 
  - `requested_size == 0` 
- allocated chunks must satisfy: 
  - `in_use == true` 
- `merge_chunks(left, right)` is directional: 
  - left survives 
  - right is tombstoned 
- coalescing is only valid for physically adjacent chunks 
- buckets are keyed by `round_down_pow2(chunk.size)` 
- `RegionManager` must map live chunk base pointers to the current owning `ChunkID` 

## Why one allocator is not enough
Fusion is aiming to support both: 
- deep learning workloads 
- physics / simulation workloads 

These workloads stress memory differently. 

### Deep learning tends to want 
- large contiguous buffers 
- predictable reuse 
- aligned storage 
- fast scratch allocation 
- dense tensor-friendly memory 

### Physics tends to want 
- irregular medium and small allocations 
- transient workspaces 
- SoA / AoSoA-friendly storage 
- long-lived and short-lived memory mixed together 

Because of that, the alloc layer should evolve into a **small family of allocators behind one interface**, 
rather than one allocator handling every case equally. 

## Intended architecture 
The intended direction is: 
```
          Allocation Request / Memory Plan
                      |
                      v
               Allocator Router
         /------------|-------------\
        v             v              v
PooledAllocator  ArenaAllocator  SmallObjectPool
 (persistent)     (scratch)       (metadata)
        \             |              /
         \------------|-------------/
                      |
                      v
               SubAllocator layer
          (CPU / pinned / GPU later)
                      |
                      v
               Raw memory regions
``` 

## Intended allocator roles 
### 1. Pooled allocator Use for: 
- persistent tensor buffers 
- physics field arrays 
- particle arrays 
- medium / large reusable allocations 

This is the natural evolution of the current `PoolAllocator`.

### 2. Arena allocator Use for: 
- op scratch buffers 
- solver temporaries 
- per-step temporary buffers 
- transient reduction / workspace memory 

- Arena allocation is usually a better fit than split/coalesce for short-lived memory. 

### 3. Small-object pool Possible future addition for: 
- graph metadata 
- descriptors 
- tiny bookkeeping objects 
- small simulation metadata structures 

- This is not the first priority, but may be useful later. 

## Relationship to storage and layout 
The alloc layer should **not** own layout semantics. 
It should care about: 
- size 
- alignment 
- lifetime 
- memory domain Layout meaning should live above this layer. 

Examples of higher-level layout/storage concerns: 
- dense contiguous tensor storage 
- strided tensor views 
- SoA particle storage 
- AoSoA blocked storage 
- tiled / blocked buffers 

These should live in storage abstractions such as:
- `TensorBuffer` 
- `DenseStorage` 
- `TensorView` 
- future `ParticleStorage` 
- future `FieldStorage` 

The allocator gives bytes. Storage gives those bytes meaning.

## Relationship to FUIR / IR 
The alloc layer should interact with IR **through a memory planning stage**, not directly. 

### IR should describe 
- operand shapes 
- layout kinds 
- access kinds 
- storage kind 
- update kind 
- item size 
- whether an operand is owned or a view 
- whether a buffer is likely persistent or temporary 

### Memory planning should translate that into 
- which buffers need actual allocation 
- which buffers are aliases / views 
- required byte sizes 
- alignment requirements 
- lifetime categories 
- allocator routing decisions 

### Alloc layer should then 
- receive allocation requests 
- choose the appropriate allocator 
- return raw memory buffers 
So the alloc layer is informed by IR metadata, but remains independent from IR lowering logic itself. 

## Planned request model 
The alloc layer is expected to evolve from simple `(size, alignment)` calls toward richer allocation requests. 
Conceptually, requests should eventually carry: 
- size 
- alignment 
- lifetime 
- usage kind
- layout hints? 
- memory domain

- Example lifetime categories:
- `Persistent`
- `Scratch`
- `StepTemporary`
- `Metadata`

This will allow a memory manager / router to choose the right allocator strategy.

## Planned growth policy changes 
The current pooled allocator growth policy is conservative and too tightly coupled to request size. 
The intended direction is to separate: 
- **chunk sizing policy** 
- **region growth policy**

### Chunk sizing policy 
Rounds user requests into allocator-friendly chunk sizes. 
### Region growth policy 
Should become more slab-like for small allocations: 
- acquire larger reusable regions 
- split internally 
- improve reuse and locality 
- reduce pathological tiny root-region behavior 

This would make the allocator more effective for mixed physics + DL workloads. 

## Design direction compared with slab allocators 
The pooled allocator is **not** intended to become a pure slab allocator. 
A pure slab allocator is usually: 
- fixed-size per class 
- rigid 
- ideal for repeated same-sized objects

Fusion’s pooled allocator should remain: 
- split/coalesce capable 
- flexible across size classes 
- region-based 
- closer in spirit to BFC-style allocators 

- That said, the growth policy for small allocations may become more slab-like. 
## Design direction 
The current pooled allocator is conceptually closer to a BFC-style allocator than to a classic slab allocator. 
Shared ideas: 
- acquire larger backing regions 
- split larger chunks 
- coalesce adjacent free chunks 
- bucket free chunks by size 
- reuse memory internally 

## Recommended layering 
The intended layered structure is: 
```
FUIR / lowering / execution planning
                |
                v
          Memory planning
                |
                v
         Allocator routing
                |
                v
             Alloc layer
                |
                v
        Sub-allocator backend
``` 

Responsibilities stay clean: 
- IR describes what memory is needed 
- memory planning decides how to request it 
- alloc layer fulfills the request 
- storage layer assigns semantic meaning to the allocated bytes 

## Near-term TODOs
- improve pooled allocator region growth strategy 
- separate chunk sizing from region growth sizing
- make `ArenaAllocator` a first-class scratch allocator 
- introduce richer `AllocationRequest` metadata
- add a `MemoryManager` / allocator router 
- add a debug invariant validation path for pooled allocator structure 
- decide whether `requested_size` should mean: 
  - original user request size 
  - or rounded allocator chunk size 
- expand tests around: 
  - fragmentation 
  - growth policy 
  - bucket correctness 
  - coalescing behavior 
  - fuzz / invariant validation  

## Long-term direction 
The alloc layer should become the basis of a unified memory system for Fusion / Nova: 
- backend-agnostic 
- physics-aware through memory planning 
- tensor-aware through storage abstractions 
- portable across CPU / pinned / GPU memory domains 
- capable of supporting both persistent and transient workloads efficiently 

The goal is not one allocator for everything. The goal is one coherent memory architecture with multiple allocator 
strategies under a common interface.