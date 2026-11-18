# Index-Space IR Semantics

# Part I: Affine Access semantics

## Scope

This document defines the mathematical semantics of Nova's index-space IR. It describes how a shared logical iteration space maps onto the physical axes and storage of each operand, and how those mappings lower into loop dimensions and operand-access plans.

The current document formalises dense affine access, including:

- Direct access
- Broadcasting
- Elementwise operations
- Reductions
- Contractions
- Loop-order lowering

Indexed and topological access will be added subsequently.

The central idea is:

> Each operand defines a fixed mapping from the shared logical iteration coordinates to its own physical tensor coordinates.

When this mapping contains no data-dependent coordinate lookup, its resulting memory address is affine in the logical coordinates.

## 1. Notation

For operand $o$, let:

- $R_o$ be its physical rank;
- $\vec{d}_o$ be its physical shape;
- $\vec{\sigma}_o$ be its element-stride vector;
- $w$ be its item size in bytes;
- $\vec{p}_o$ be a coordinate in its physical index space.

For the shared logical iteration space, let:

- $L$ be the number of logical axes;
- $E_m$ be the extent of logical axis $m$;
- $\vec{\ell}$ be a logical coordinate vector.

All coordinate domains are zero-based and half-open.

## 2. Physical tensor storage

The physical coordinate of operand $o$ is:

$$\vec{p}_o=(p_{o,0},\ldots,p_{o,R_o-1})$$

Its byte address is:

$$\mathrm{addr}_o(\vec{p}_o)=\mathrm{base}_o+w\sum_{a=0}^{R_o-1}\sigma_{o,a}p_{o,a}$$

For example, a contiguous tensor with shape $[2,3,4]$ normally has element strides:

$$[12,4,1]$$

The same address equation describes non-contiguous views because physical strides are supplied independently of the logical computation.

`OperandDescription` supplies the operand's physical shape, strides, item size and other storage properties. The index-space IR supplies the semantic mapping between the computation and that storage.

## 3. Logical iteration space

The IR defines $L$ logical axes with extents:

$$E_0,E_1,\ldots,E_{L-1}$$

Their Cartesian product defines the logical iteration domain:

$$\mathcal{D}=[0,E_0)\times[0,E_1)\times\cdots\times[0,E_{L-1})$$

A logical point is:

$$\vec{\ell}=(\ell_0,\ell_1,\ldots,\ell_{L-1})\in\mathcal{D}$$

Logical axes describe the coordinates of the computation. They are independent of the rank, shape and layout of any particular operand.

For an elementwise operation with output shape $[2,3,4]$, the logical coordinates may be written as $(i,j,k)$, where:

$$i\in[0,2),\qquad j\in[0,3),\qquad k\in[0,4)$$

## 4. Logical-to-physical axis mappings

Each `AxisUse` describes how one physical axis of an operand relates to one logical axis.

### 4.1 Direct access

If physical axis $a$ of operand $o$ directly uses logical axis $m$, then:

$$p_{o,a}=\ell_m$$

The physical and logical extents must agree:

$$d_{o,a}=E_m$$

### 4.2 Broadcast access

If physical axis $a$ is broadcast over logical axis $m$, then:

$$p_{o,a}=0$$

The logical coordinate changes while the operand repeatedly accesses the same physical coordinate.

The physical extent is therefore one:

$$d_{o,a}=1$$

### 4.3 Absent logical axes

An operand need not contain a physical axis corresponding to every logical axis. This occurs when a lower-rank operand is right-aligned with a higher-rank output.

If operand $o$ does not use logical axis $m$, its physical coordinates are invariant with respect to that logical coordinate. Consequently, its lowered byte stride for logical axis $m$ is zero.

An absent logical axis and an explicit size-one broadcast both lower to a zero byte stride.

They remain structurally distinct in the IR:

- An explicit broadcast has a physical axis.
- A rank-padded logical axis has no corresponding physical axis.

## 5. Matrix form of affine access

The complete coordinate mapping for operand $o$ can be represented by a matrix:

$$M_o\in\{0,1\}^{R_o\times L}$$

The physical coordinates are obtained from the logical coordinates using:

$$\vec{p}_o=M_o\vec{\ell}$$

A direct physical axis has one non-zero entry in its row.

A broadcast physical axis has a zero row.

A logical axis unused by an operand has a zero column in that operand's mapping matrix.

Substituting the coordinate mapping into the physical address equation gives:

$$\mathrm{addr}_o(\vec{\ell})=\mathrm{base}_o+w\vec{\sigma}_o^{\mathsf T}\vec{p}_o$$

Using $\mathbf{p}_o = M_o\mathbf{\ell}$:

$$\mathrm{addr}_o(\vec{\ell})=\mathrm{base}_o+w\vec{\sigma}_o^{\mathsf T}M_o\vec{\ell}$$

Define the byte-stride vector over logical axes as:

$$\vec{\beta}_o=wM_o^{\mathsf T}\vec{\sigma}_o$$

The address can then be written as:

$$\mathrm{addr}_o(\vec{\ell})=\mathrm{base}_o+\sum_{m=0}^{L-1}\beta_{o,m}\ell_m$$

This is the defining property of the affine access model:

> The address is a fixed linear function of the logical coordinates, plus a base address.

The IR's `AxisUse` entries are a sparse, typed representation of the mapping matrix $M_o$.

## 6. Elementwise broadcasting

Consider the elementwise operation:

$$Y[2,3,4]=A[2,1,4]+B[3,1]$$

The shared logical coordinates are $(i,j,k)$.

The output mapping is:

$$Y(i,j,k)=Y[i,j,k]$$

The first input mapping is:

$$A(i,j,k)=A[i,0,k]$$

The second input mapping is:

$$B(i,j,k)=B[j,0]$$

The leading logical axis $i$ is absent from $B$ because $B$ is right-aligned as a rank-two operand.

The output address is:

$$\mathrm{addr}_Y(i,j,k)=\mathrm{base}_Y+w\left(i\sigma_{Y,0}+j\sigma_{Y,1}+k\sigma_{Y,2}\right)$$

The address of $A$ is:

$$\mathrm{addr}_A(i,j,k)=\mathrm{base}_A+w\left(i\sigma_{A,0}+k\sigma_{A,2}\right)$$

The address of $B$ is:

$$\mathrm{addr}_B(i,j,k)=\mathrm{base}_B+wj\sigma_{B,0}$$

The corresponding logical byte strides are:

| Operand | `i` | `j` | `k` |
|---|---:|---:|---:|
| `Y` | `w * sigma[Y,0]` | `w * sigma[Y,1]` | `w * sigma[Y,2]` |
| `A` | `w * sigma[A,0]` | `0` | `w * sigma[A,2]` |
| `B` | `0` | `w * sigma[B,0]` | `0` |

This table is the mathematical form of the affine access information produced by lowering.

## 7. Reductions

Consider the reduction:

$$Y[i,k]=\sum_j X[i,j,k]$$

The logical axes are:

| Logical axis | Kind |
|---|---|
| `i` | `Independent` |
| `k` | `Independent` |
| `j` | `Reduction` |

The input maps all three logical coordinates directly:

$$X(i,k,j)=X[i,j,k]$$

The output is invariant along the reduction coordinate:

$$Y(i,k,j)=Y[i,k]$$

Its byte stride for logical axis $j$ is therefore zero.

If the selected loop order is `(i, k, j)`, the conceptual loop nest is:

```cpp
for (i)
   for (k) {
      acc = 0;

      for (j)
         acc += X[i, j, k];

      Y[i, k] = acc;
   }
```

`IndexKind::Reduction` supplies the reduction semantics. The address of each operand remains affine in the logical coordinates.

If the output retains the reduced dimension with extent one, that physical dimension is a storage-level representation of `keepdim`. It does not make the output vary over the reduction coordinate.

## 8. Contractions

Matrix multiplication can be written as:

$$C_{ij}=\sum_k A_{ik}B_{kj}$$

The logical axes are:

- $i$: independent;
- $j$: independent;
- $k$: reduction.

The output mapping is:

$$C(i,j,k)=C[i,j]$$

The first input mapping is:

$$A(i,j,k)=A[i,k]$$

The second input mapping is:

$$B(i,j,k)=B[k,j]$$

Their logical byte strides are:

| Operand | `i` | `j` | `k` |
|---|---:|---:|---:|
| `C` | `w * sigma[C,0]` | `w * sigma[C,1]` | `0` |
| `A` | `w * sigma[A,0]` | `0` | `w * sigma[A,1]` |
| `B` | `0` | `w * sigma[B,1]` | `w * sigma[B,0]` |

Labels such as $i$, $j$ and $k$ belong to the frontend notation.

Once the binding has been resolved, they are represented by logical-axis IDs, and downstream compiler stages operate only on the canonical IR.

Roles such as `M`, `N`, `K` and `Batch` are planning annotations derived from operand participation. They do not define the underlying contraction semantics.

## 9. Loop order and lowering

Logical axes have stable IDs, but the planner selects a permutation:

$$\pi=(\pi_0,\pi_1,\ldots,\pi_{L-1})$$

This permutation defines the physical loop order.

The loop coordinate at position $q$ corresponds to logical axis $\pi_q$.

Let $s_o$ denote the `byte_stride_per_loop` vector for operand $o$.

Lowering therefore produces:

$$s_o[q]=\beta_o[\pi(q)]$$

For a contraction, the loop order might be `(i, j, k)`, placing the output coordinates before the reduction coordinate.

A different legal permutation changes traversal and performance but does not change the mathematical operation.

`LoopDim` records the extent and semantic kind of the logical axis at each loop position.

`OperandAccess` records how the address of each operand changes when that loop coordinate advances.

## 10. Correspondence with IR types

| IR concept | Mathematical meaning |
|---|---|
| `LogicalAxis` | One coordinate `ell[m]` in the shared computation domain |
| `LogicalAxis::extent` | The logical domain size `E_m` |
| `IndexKind` | Whether a coordinate identifies an output point or is reduced |
| `PhysicalAxis` | One coordinate `p[o,a]` in an operand's physical index space |
| `AxisAccess::Direct` | `p[o,a] = ell[m]` |
| `AxisAccess::Broadcast` | `p[o,a] = 0` |
| `OperandUse` | The complete coordinate mapping `M_o` for one operand |
| `loop_order` | A permutation of the logical coordinates |
| `AffineAccess` | The byte-stride coefficients `beta_o` in the address equation |

## 11. Affine IR invariants

The validators establish the assumptions required by the affine mathematics:

1. Every logical-axis reference is valid.
2. Every physical-axis reference is valid.
3. Physical-axis IDs are unique within an operand.
4. A physical axis is not mapped more than once within an `OperandUse`.
5. Direct mappings have equal logical and physical extents.
6. Broadcast mappings refer to size-one physical axes.
7. Every required independent logical axis is represented by the output.
8. Reduction axes do not cause the output address to vary.
9. The loop order contains every logical axis exactly once.
10. Lowered access vectors contain one byte-stride coefficient per loop position.
11. Operand descriptions agree with the physical axes represented by the IR.

These checks act as proof obligations for lowering.

Once they hold, the compiler can safely replace the structural IR mapping with the affine byte-address equation.

## 12. Boundary of the affine model

The affine scheme requires:

$$\mathrm{addr}_o(\vec{\ell})=\mathrm{base}_o+\vec{\beta}_o^{\mathsf T}\vec{\ell}$$

The byte-stride vector $\vec{\beta}_o$ is fixed before execution.

Indexed access crosses this boundary.

In an indexed operand, at least one physical coordinate is produced by a lookup such as:

$$p_{o,a}=I[\ell_m]$$

rather than by a direct or constant mapping.

Substituting that lookup into the address equation gives:

$$\mathrm{addr}_o(\vec{\ell})=\mathrm{base}_o+w\left(\sigma_{o,a}I[\ell_m]+\sum_{b\ne a}\sigma_{o,b}p_{o,b}\right)$$

The value of $I[\ell_m]$ must be loaded at runtime. The address is therefore no longer a fixed affine function of the logical loop coordinates.

The indexed-access section of this specification will define:

- Coordinate maps
- Indexed gather semantics
- Indexed scatter and accumulation semantics
- Topology-controlled iteration domains
- CRS and blocked-CRS traversal
- The relationship between logical interaction axes and physical particle axes

# Part II: Indexed Access Semantics

## 13. Scope

This part defines the indexed-access semantics of Nova's index-space IR.

Indexed access extends the affine model to computations in which a logical coordinate does not directly identify a physical tensor coordinate. Instead, the physical coordinate is obtained through a runtime index map.

This model supports:

- Indexed gathers
- Indexed scatter and scatter-add
- Pairwise and graph-like interaction domains
- Edge-list topology
- CRS topology
- Blocked-CRS topology
- Dense, SoA and AoSoA storage
- Reverse-mode differentiation through indexed reads

Simulation-specific concepts are not part of the index-space IR. Particle interactions, neighbour lists and molecular forces are expressed using the more general indexed-access and topology abstractions defined here.

The central distinction is:

> Indexed access determines which physical entity is selected. Storage layout determines where that entity is stored.

---

## 14. General coordinate-map model

In the affine model, each physical coordinate is either copied from a logical coordinate or held constant by broadcasting.

Indexed access generalises this relationship.

For physical axis $a$ of operand $o$, define a coordinate map:

$$p_{o,a}=\phi_{o,a}(\vec{\ell})$$

Here:

- $\vec{\ell}$ is a point in the shared logical iteration domain;
- $p_{o,a}$ is the selected coordinate along physical axis $a$;
- $\phi_{o,a}$ defines how that physical coordinate is obtained.

The complete physical coordinate vector is:

$$\vec{p}_o=\phi_o(\vec{\ell})$$

The storage layout then maps the physical coordinate to an address:

$$\mathrm{addr}_o(\vec{\ell})=\Lambda_o\bigl(\phi_o(\vec{\ell})\bigr)$$

The function $\Lambda_o$ is the storage-layout mapping for operand $o$.

This separates the computation into three stages:

1. Select a logical iteration point.
2. Map that point to an operand's physical coordinates.
3. Map those physical coordinates into storage.

### 14.1 Direct coordinate maps

A direct mapping copies logical coordinate $\ell_m$ into physical coordinate $p_{o,a}$:

$$\phi_{o,a}(\vec{\ell})=\ell_m$$

The logical and physical extents must agree:

$$E_m=d_{o,a}$$

### 14.2 Broadcast coordinate maps

A broadcast mapping holds a physical coordinate constant while the corresponding logical coordinate changes:

$$\phi_{o,a}(\vec{\ell})=0$$

The physical extent is one:

$$d_{o,a}=1$$

### 14.3 Indexed coordinate maps

An indexed mapping obtains a physical coordinate from an index map:

$$\phi_{o,a}(\vec{\ell})=I_r(\ell_m)$$

Here:

- $r$ identifies the index map;
- $\ell_m$ is the logical coordinate used to query it;
- $I_r(\ell_m)$ is the resulting physical coordinate.

For a one-dimensional index map:

$$I_r:[0,E_m)\rightarrow[0,d_{o,a})$$

Unlike a direct mapping, the physical coordinate cannot be determined from $\ell_m$ using a fixed stride. The value of $I_r(\ell_m)$ must be loaded or otherwise resolved at runtime.

The corresponding operand address is:

$$\mathrm{addr}_o(\vec{\ell})=\Lambda_o\bigl(\phi_o(\vec{\ell})\bigr)$$

For dense strided storage, this becomes:

$$\mathrm{addr}_o(\vec{\ell})=\mathrm{base}_o+w\sum_{a=0}^{R_o-1}\sigma_{o,a}\phi_{o,a}(\vec{\ell})$$

Because at least one coordinate map contains a runtime lookup, the address is no longer a fixed affine function of the logical coordinates.

---

## 15. Index maps

An index map is a semantic mapping from a logical coordinate to a physical coordinate.

The canonical IR refers to an index map through a stable identifier. It does not contain a pointer to a concrete index buffer or topology object.

Conceptually, an index-map declaration describes:

- The map identifier
- The logical axis forming its domain
- The physical coordinate domain forming its codomain
- Any structural properties required by validation

The execution plan binds that declaration to a concrete representation, such as:

- An index array
- An edge-list endpoint array
- A CRS column-index array
- A CRS row owner
- A blocked-CRS accessor
- A generated or implicit mapping

Consequently, two different physical topology representations may implement the same index-space semantics.

### 15.1 Map identity

Two indexed mappings using the same logical and physical axes are not necessarily equivalent.

For example:

$$s(e)\neq t(e)$$

even though both functions map an interaction identifier $e$ to a particle identifier.

An indexed `AxisUse` must therefore identify the index map it uses. Marking the access only as `Indexed` is insufficient.

### 15.2 Map bounds

For every valid logical coordinate, the resulting physical coordinate must be in bounds:

$$0\leq I_r(\ell_m)<d_{o,a}$$

Some properties can be validated structurally when the IR is built. Bounds that depend on runtime index data must be checked when the topology is created, bound or executed.

### 15.3 Map immutability during execution

An index map is treated as immutable for the duration of an execution plan.

Changing the topology or index data may invalidate:

- Cached traversal plans
- Blocked topology representations
- Gather metadata
- Vectorisation decisions
- Bounds validation

Topology mutation therefore requires rebinding or rebuilding any plan whose assumptions are no longer valid.

---

## 16. Abstract topology relations

A topology describes a collection of interactions between entities.

Let the entity domain be:

$$\mathcal{V}=[0,N)$$

Let the interaction domain be:

$$\mathcal{E}=[0,E)$$

Each interaction identifier $e$ maps to an ordered pair of entities:

$$T(e)=\bigl(s(e),t(e)\bigr)$$

Here:

- $s(e)$ is the source entity;
- $t(e)$ is the target entity.

The functions $s$ and $t$ are ordinary index maps.

The index-space IR does not require these maps to be named source and target. Those names describe their role in a particular topology or operation. At the IR level, they are distinct index maps referenced by stable IDs.

### 16.1 Direction

The pair $\bigl(s(e),t(e)\bigr)$ is ordered.

Therefore:

$$\bigl(s(e),t(e)\bigr)\neq\bigl(t(e),s(e)\bigr)$$

unless the operation explicitly defines those interactions as equivalent.

An undirected relation may be represented by:

- One stored interaction with symmetric operation semantics; or
- Two directed interactions, one in each direction.

This choice belongs to the topology and operation contract.

### 16.2 Multiplicity

The interaction collection is an ordered multiset rather than necessarily a mathematical set.

This means:

- Duplicate interactions may exist.
- Multiple interactions may reference the same source.
- Multiple interactions may reference the same target.
- Self-interactions may exist unless explicitly rejected.
- Interaction order may be observable when outputs are indexed by interaction ID.

Operation-specific validators may impose stronger constraints.

### 16.3 Empty and isolated entities

A valid topology may contain:

- No interactions
- Entities with no outgoing interactions
- Entities with no incoming interactions
- Empty CRS rows

These cases must not require special semantic treatment, although a lowering implementation may select specialised execution paths for them.

---

## 17. Indexed reads

An indexed read is a gather.

Consider an input tensor $X$ whose first physical axis identifies an entity and whose second physical axis identifies a component.

Let $I_r$ map logical interaction identifiers to physical entity identifiers.

The gather is:

$$Y[e,c]=X[I_r(e),c]$$

The coordinate mappings for the input are:

$$p_{X,0}=I_r(e)$$

$$p_{X,1}=c$$

The entity axis uses indexed access, while the component axis uses direct access.

The output mappings are direct:

$$p_{Y,0}=e$$

$$p_{Y,1}=c$$

The gather operation does not imply any particular topology representation or storage layout.

---

## 18. Multiple accesses to the same storage

One storage object may be accessed through more than one coordinate map within the same operation.

For example:

$$\Delta[e,c]=X[t(e),c]-X[s(e),c]$$

Both terms read from $X$, but they use different index maps.

The semantic model must therefore distinguish:

- Storage identity
- Operand access occurrence
- Coordinate mapping

Two operand access occurrences may refer to the same underlying tensor while using different `OperandUse` mappings.

The compiler may represent this using either:

- Multiple operand slots bound to the same storage; or
- Multiple access sites referencing one operand identity.

This representation choice does not change the mathematical semantics.

---

## 19. Indexed writes

An indexed write is a scatter.

A scatter maps a logical coordinate to a physical destination coordinate:

$$p_{Y,0}=I_r(e)$$

The value written to that destination may use different update semantics.

### 19.1 Scatter assignment

A scatter assignment has the form:

$$Y[I_r(e),c]\gets X[e,c]$$

This operation is unambiguous only when:

- The index map is injective; or
- A conflict-resolution rule is defined for duplicate destinations.

Without one of these conditions, multiple logical iterations may assign different values to the same physical element.

### 19.2 Scatter-add

A scatter-add accumulates into the selected destination:

$$Y[I_r(e),c]\gets Y[I_r(e),c]+X[e,c]$$

Equivalent contributions are:

$$Y[p,c]=\sum_{e:I_r(e)=p}X[e,c]$$

Scatter-add is well-defined in the presence of repeated destination indices.

Its parallel implementation may require:

- Atomics
- Segmented reduction
- Thread-local accumulation
- Graph colouring
- Ownership partitioning
- Another conflict-free traversal

These are lowering and execution choices rather than index-space semantics.

### 19.3 Determinism

Floating-point addition is not associative. Different parallel accumulation orders may therefore produce slightly different results.

An execution policy may distinguish between:

- Deterministic accumulation
- Stable but representation-dependent accumulation
- Unordered parallel accumulation

The mathematical operation remains scatter-add, while determinism is an execution guarantee.

### 19.4 Access and update semantics are separate

Indexed access determines the destination coordinate.

The update kind determines what happens at that coordinate.

Therefore, the following are separate properties:

- `Indexed` coordinate selection
- Assignment, addition or another update operation

An indexed access must not implicitly mean scatter assignment.

---

## 20. Pair-delta example

Consider a position tensor:

$$X\in\mathbb{R}^{N\times C}$$

For three-dimensional coordinates:

$$C=3$$

Let the topology contain $E$ directed interactions.

The pair-delta operation is:

$$\Delta[e,c]=X[t(e),c]-X[s(e),c]$$

with:

$$e\in[0,E)$$

and:

$$c\in[0,C)$$

The logical iteration domain is:

$$\mathcal{D}=[0,E)\times[0,C)$$

The logical axes are:

| Logical coordinate | Meaning | Kind |
|---|---|---|
| `e` | Interaction identifier | Independent |
| `c` | Component identifier | Independent |

The access occurrences are:

| Access occurrence | Physical axis | Coordinate mapping |
|---|---|---|
| Output delta | Interaction | `e` |
| Output delta | Component | `c` |
| Source position | Particle | `s(e)` |
| Source position | Component | `c` |
| Target position | Particle | `t(e)` |
| Target position | Component | `c` |

The output uses only direct mappings.

The source and target particle axes use indexed mappings.

The component axes use direct mappings.

This example combines:

- A shared logical interaction axis
- A direct component axis
- Two distinct index maps
- Two accesses to the same underlying position storage
- A topology-independent scalar operation

---

## 21. Reverse-mode differentiation

The adjoint of an indexed gather is a scatter-add.

For:

$$Y[e,c]=X[I_r(e),c]$$

and upstream gradient $\overline{Y}$, the gradient with respect to $X$ is:

$$\overline{X}[p,c]=\sum_{e:I_r(e)=p}\overline{Y}[e,c]$$

Repeated indices cause multiple logical contributions to accumulate into the same physical gradient element.

For pair delta:

$$\Delta[e,c]=X[t(e),c]-X[s(e),c]$$

the gradient with respect to the position tensor is:

$$\overline{X}[p,c]=\sum_{e:t(e)=p}\overline{\Delta}[e,c]-\sum_{e:s(e)=p}\overline{\Delta}[e,c]$$

The target contribution is added.

The source contribution is subtracted.

Consequently, scatter-add support is required for reverse-mode differentiation through indexed reads, even when the forward operation performs only gathers.

---

## 22. Topology representations

The abstract topology relation does not prescribe a storage representation.

Different representations may encode the same collection of interactions.

### 22.1 Edge-list representation

An edge list stores source and target maps explicitly:

$$s(e)=\mathrm{source}[e]$$

$$t(e)=\mathrm{target}[e]$$

Both endpoint maps require an indexed load.

This representation provides simple random access by interaction ID.

### 22.2 CRS representation

CRS stores:

- A row-offset array
- A column-index array

For source entity $i$, its interactions occupy:

$$e\in[\mathrm{row}[i],\mathrm{row}[i+1])$$

Within that range:

$$s(e)=i$$

$$t(e)=\mathrm{column}[e]$$

The source map is implicit in the row traversal.

The target map is loaded from the column-index array.

Although CRS naturally lowers into a segmented row traversal, its abstract semantics remain a collection of indexed interactions.

### 22.3 Blocked-CRS representation

Blocked CRS groups interactions into blocks suitable for vector or cache-oriented traversal.

The blocked representation may introduce:

- Block identifiers
- Lanes within a block
- Padding entries
- Activity masks
- Reordered interactions

Padding entries are not logical interactions and must not contribute to observable outputs or reductions.

Blocked traversal changes execution order and storage access, but it does not change the underlying topology relation.

### 22.4 Representation equivalence

Two topology representations are semantically equivalent when they encode the same interaction multiset and preserve all operation-visible properties.

If interaction-indexed outputs are observable, a representation change must either:

- Preserve the interaction enumeration; or
- Provide the permutation relating the two enumerations.

---

## 23. Storage-layout independence

Index maps select physical entities. They do not define how those entities are arranged in memory.

After indexed coordinate selection, the storage layout maps the resulting physical coordinates to addresses.

### 23.1 Dense storage

For dense strided storage:

$$\Lambda_o(\vec{p}_o)=\mathrm{base}_o+w\sum_{a=0}^{R_o-1}\sigma_{o,a}p_{o,a}$$

### 23.2 SoA storage

In structure-of-arrays storage, component fields are stored in separate contiguous regions.

The selected entity coordinate and component coordinate are interpreted by the SoA layout mapping.

### 23.3 AoSoA storage

In array-of-structures-of-arrays storage, an entity coordinate is decomposed into a block and lane.

For block width $B$ and entity coordinate $p$:

$$b=\lfloor p/B\rfloor$$

$$\lambda=p\bmod B$$

The AoSoA layout mapping uses the block, component and lane coordinates to determine the address.

The topology index map still returns the logical physical entity coordinate $p$. It does not need to know the block width or lane arrangement.

This separation allows the same indexed IR to target Dense, SoA and AoSoA storage.

---

## 24. Indexed lowering

Indexed lowering converts semantic coordinate maps into executable access and traversal plans.

The lowering process must preserve:

- Logical-axis identity
- Index-map identity
- Physical-axis identity
- Storage-layout information
- Update semantics

For an indexed physical axis, a constant byte stride is insufficient. The lowered access must instead describe how to:

1. Resolve the relevant index map.
2. Load or derive the physical coordinate.
3. Apply the operand's storage-layout mapping.
4. Perform the required read or update.

The planner may select different traversals for the same semantic IR:

- Flat interaction traversal
- CRS row traversal
- Blocked-CRS traversal
- Vectorised lane traversal
- Conflict-free scatter traversal

These traversal decisions affect performance but do not change the mathematical operation.

---

## 25. Correspondence with compiler concepts

| Compiler concept | Mathematical meaning |
|---|---|
| `LogicalAxis` | One coordinate in the logical computation domain |
| `PhysicalAxis` | One coordinate in an operand's physical domain |
| `AxisAccess::Direct` | `p = ell` |
| `AxisAccess::Broadcast` | `p = 0` |
| `AxisAccess::Indexed` | `p = I(ell)` |
| `IndexMapId` | Stable reference to an index map |
| `OperandUse` | Coordinate mapping for one operand access occurrence |
| Topology binding | Concrete implementation of one or more index maps |
| Indexed access plan | Runtime procedure for resolving indexed addresses |
| Update kind | Assignment, accumulation or another write operation |
| Traversal plan | Physical order in which logical interactions are executed |

---

## 26. Indexed IR invariants

The indexed-access validators establish the following assumptions:

- Every logical-axis reference is valid.
- Every physical-axis reference is valid.
- Every indexed `AxisUse` references a valid index map.
- Direct and broadcast mappings do not reference index maps.
- The domain of an index map matches its logical axis.
- The index-map extent agrees with the logical-axis extent.
- Every index-map result lies within the referenced physical-axis extent.
- Each physical axis has one coordinate mapping per access occurrence.
- The topology interaction count agrees with the logical interaction extent.
- Concrete topology bindings satisfy the declared index-map interface.
- Indexed writes define an explicit update operation.
- Potentially colliding assignments are rejected or given a conflict policy.
- Padding entries in blocked representations are masked.
- Storage-layout bindings agree with the operand's physical axes.
- An execution plan is invalidated when a bound topology changes incompatibly.

Some invariants are structural and can be checked when the IR is built.

Others depend on runtime index data and must be checked when the topology is constructed, bound or executed.

---

## 27. Initial implementation boundary

The initial indexed-access implementation supports:

- One-dimensional index maps
- One logical interaction axis per index-map lookup
- One physical coordinate produced by each index map
- Indexed gathers
- Scatter-add for reverse-mode differentiation
- Edge-list topology
- CRS topology
- Blocked-CRS topology
- Dense, SoA and AoSoA storage
- Topology that remains immutable during plan execution

The following are deferred:

- Chained or recursive indexing
- Index maps consuming multiple logical coordinates
- Index maps producing multiple coordinates
- Arbitrary sparse tensor algebra
- Topology mutation during execution
- Distributed topology
- General user-defined scatter reductions
- Cross-device indexed access

These restrictions define the first implementation boundary without constraining later extensions to the coordinate-map model.