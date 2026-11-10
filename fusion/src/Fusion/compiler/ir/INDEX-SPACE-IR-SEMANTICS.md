# Index-Space IR Semantics

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

Using $\vec{p}_o=M_o\vec{\ell}$:

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

Lowering therefore produces:

$$\mathrm{byte\_stride\_per\_loop}_o[q]=\beta_{o,\pi_q}$$

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