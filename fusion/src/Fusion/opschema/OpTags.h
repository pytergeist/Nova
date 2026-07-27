#ifndef FUSION_CORE_OPSCHEMA_OPTAGS_H_
#define FUSION_CORE_OPSCHEMA_OPTAGS_H_

/// This file holds low level Op tags, these form the meta-data, along with
/// OpSchema, and OpTraits, for operations in the system to be used by all
/// higher level layers that execute Operations. Including subsystems: Ops, CPU,
/// Autodiff (poss FUIR?). This file is simple tag declarations, ordered in the
/// same fashion as the Fusion/ops layer. To add to this file, please the new
/// tag under one of the file headers in the order it appears in that file. If
/// you are adding a new file to the Fusion/Ops layer (denoting a new type of o
/// operation - or for refactoring purposes) please add a header for that file
/// in alphabetical order. Even though inplace ops are explicit in these files,
/// do not add a seperate tag for these, as this should be reflected in the
/// OpSchema.
///
/// The tags follow the naming convention OpTag, where
/// op = fn name from file, in camelcase.

// Comparison

struct GreaterTag {};
struct GreaterEqualTag {};
struct MaximumTag {};

/// Ewise

struct AddTag {};
struct SubTag {};
struct MulTag {};
struct DivTag {};
struct ReciprocalTag {};
struct PowTag {};

/// LinAlg
struct MatMulTag {};
struct SwapAxesTag {};

/// Reduce
struct SumTag {};
struct MeanTag {};

/// Transcendental
struct SqrtTag {};
struct LogTag {};
struct ExpTag {};

/// Topology
struct PairDelta3Tag {};

#endif // FUSION_CORE_OPSCHEMA_OPTAGS_H_