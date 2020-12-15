#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/passes/canonicalize_graph_fuser_ops.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/clear_undefinedness.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/cuda_graph_fuser.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/inline_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/inplace_check.h>
#include <torch/csrc/jit/passes/insert_guards.h>
#include <torch/csrc/jit/passes/loop_unrolling.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_expands.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/specialize_autogradzero.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include "ATen/core/interned_strings.h"
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

C10_DEFINE_bool(
    torch_jit_enable_new_executor,
    true,
    "If this flag is set to false TorchScript will be using the legacy/original executor");

namespace torch {
namespace jit {

#if defined(C10_MOBILE)
static std::atomic<bool> executor_mode{true};
static std::atomic<bool> profiling_mode{false};
#else
static std::atomic<bool> executor_mode{true};
static std::atomic<bool> profiling_mode{true};
#endif

static std::atomic<size_t> num_profiled_runs{1};
static std::atomic<size_t> bailout_depth{20}; // NOLINT

std::atomic<bool>& getProfilingMode() {
  return profiling_mode;
}
std::atomic<bool>& getExecutorMode() {
  return executor_mode;
}

std::atomic<size_t>& getNumProfiledRuns() {
  return num_profiled_runs;
}

std::atomic<size_t>& getBailoutDepth() {
  return bailout_depth;
}

static bool needsGradientInProfilingMode(Block* b) {
  for (auto n : b->nodes()) {
    if (n->kind() == prim::BailOut) {
      auto ptt = n->output()->type()->expect<TensorType>();
      if (ptt->requiresGrad() && *ptt->requiresGrad()) {
        return true;
      }
    }
    if (n->kind() == prim::profile) {
      auto type = n->ty(attr::profiled_type)->expect<TensorType>();
      if (type->requiresGrad() && *type->requiresGrad()) {
        return true;
      }
    }

    for (auto ib : n->blocks()) {
      if (needsGradientInProfilingMode(ib)) {
        return true;
      }
    }
  }
  return false;
}

void runNooptPassPipeline(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before LowerGradOf (beginning of runNooptPassPipeline)\n", *graph);
  LowerGradOf(*graph);
  GRAPH_DEBUG("After LowerGradOf, before RemoveExpands\n", *graph);
  RemoveExpands(graph);
  GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
  CanonicalizeOps(graph);
  GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
  EliminateDeadCode(graph);
  GRAPH_DEBUG(
      "After EliminateDeadCode (end of runNooptPassPipeline)\n", *graph);
}

void runPreAutodiffPassPipeline(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before InsertGuards (beginning of runPreAutodiffPassPipeline)\n",
      *graph);

  if (tensorExprFuserEnabled() || RegisterCudaFuseGraph::isRegistered()) {
    // With TE fuser or nvfuser, we don't generate bailouts
    LowerGradOf(*graph);
    GRAPH_DEBUG("After LowerGradOf, before specializeAutogradZero\n", *graph);
  } else {
    InsertGuards(graph);
    GRAPH_DEBUG("After InsertGuards, before LowerGradOf\n", *graph);
    LowerGradOf(*graph);
    GRAPH_DEBUG("After LowerGradOf, before EliminateRedundantGuards\n", *graph);
    EliminateRedundantGuards(graph);
    GRAPH_DEBUG(
        "After EliminateRedundantGuards, before InsertBailOuts\n", *graph);
    InsertBailOuts(graph);
    GRAPH_DEBUG(
        "After InsertBailOuts, before specializeAutogradZero\n", *graph);
  }

  specializeAutogradZero(graph);
  GRAPH_DEBUG("After specializeAutogradZero\n", *graph);
  // runRequiredPasses
  {
    RemoveExpands(graph);
    GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
    CanonicalizeOps(graph);
    GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
    EliminateDeadCode(graph);
    GRAPH_DEBUG("After EliminateDeadCode", *graph);
  }
  PeepholeOptimize(graph);
  GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
  ConstantPropagation(graph);

  // runOptimization:
  {
    EliminateDeadCode(graph);
    GRAPH_DEBUG(
        "After EliminateDeadCode, before EliminateCommonSubexpression\n",
        *graph);
    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before PeepholeOptimize\n",
        *graph);

    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG("After ConstantPropagation, before ConstantPooling\n", *graph);
    ConstantPooling(graph);
    GRAPH_DEBUG("After ConstantPooling, before UnrollLoops\n", *graph);

    UnrollLoops(graph);
    GRAPH_DEBUG("After UnrollLoops, before RemoveListMutation\n", *graph);
    // run again with unrolled loops
    RemoveListMutation(graph);
    GRAPH_DEBUG("After RemoveListMutation, before PeepholeOptimize\n", *graph);
    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG(
        "After ConstantPropagation, before EliminateCommonSubexpression\n",
        *graph);

    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before CheckInplace\n", *graph);

    CheckInplace(graph);
  }
  GRAPH_DEBUG(
      "After CheckInplace (end of runPreAutodiffPassPipeline)\n", *graph);
}


  void removeTensorTypeSpecialization2(Value* v) {
    if (!v->type()->cast<TensorType>()) {
      return;
    }
    // Constants & TensorExprGroup will always produce specialized tensor type,
    // TypeCheck are inserted by this pass and only used by fusion groups that
    // insert proper guards
    if (v->node()->kind() == prim::Constant ||
        v->node()->kind() == prim::TypeCheck ||
        v->node()->kind() == prim::TensorExprGroup) {
      return;
    }
    v->setType(TensorType::get());
  }

  void removeTensorTypeSpecializations2(Block* block) {
    for (Value* v : block->inputs()) {
      removeTensorTypeSpecialization2(v);
    }
    for (Node* n : block->nodes()) {
      for (Block* b : n->blocks()) {
        removeTensorTypeSpecializations2(b);
      }
      for (Value* v : n->outputs()) {
        removeTensorTypeSpecialization2(v);
      }
    }
  }

  static void guardDifferentiableGraph(Node* fusion_group) {
    GRAPH_DEBUG("Inserting a typecheck guard for a node", *fusion_group);
    auto subgraph = SubgraphUtils::getSubgraph(fusion_group);

    // Fixup types of the subgraph inputs
    std::vector<Value*> inputs_to_check;
    for (Value* input : fusion_group->inputs()) {
      // We only check inputs of the fusion group and expect NNC to infer
      // intermediates and outputs shapes
      if (!input->type()->cast<TensorType>()) {
        continue;
      }

      // fusion outputs are already guarded
      if (input->node()->kind() == prim::Constant ||
          input->node()->kind() == prim::DifferentiableGraph) {
        continue;
      }
      inputs_to_check.push_back(input);
    }

    if (!inputs_to_check.size()) {
      return;
    }

    // Add prim::TypeCheck node
    //
    // TypeCheck nodes  look like the following:
    //   %out1 : Float(2, 3), %out2 : Int(10, 30), %types_match : bool =
    //   prim::TypeCheck(%inp1 : Tensor, %inp2 : Tensor)
    //
    // They have N inputs whose types we are going to check and N+1 outputs. The
    // first N outputs specify expected types and N+1-th output holds the result
    // of the check (bool).
    Node* typecheck_node =
        fusion_group->owningGraph()
            ->create(
                prim::TypeCheck, inputs_to_check, inputs_to_check.size() + 1)
            ->insertBefore(fusion_group);
    Value* typecheck_result = typecheck_node->output(inputs_to_check.size());

    std::unordered_map<Value*, Value*> typechecked_inputs;
    for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
      typechecked_inputs[typecheck_node->input(i)] = typecheck_node->output(i);
    }

    // Fixup types of the typecheck node outputs, which are used by the op in
    // execution
    typecheck_node->output(inputs_to_check.size())->setType(BoolType::get());
    for (size_t i = 0; i < typecheck_node->inputs().size(); ++i) {
      typecheck_node->output(i)->setType(typecheck_node->input(i)->type());
    }

    // Insert if
    auto versioning_if =
        fusion_group->owningGraph()
            ->create(
                prim::If, {typecheck_result}, fusion_group->outputs().size())
            ->insertAfter(typecheck_node);
    for (size_t idx = 0; idx < fusion_group->outputs().size(); ++idx) {
      versioning_if->output(idx)->setType(fusion_group->output(idx)->type());
      fusion_group->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
    }
    auto true_block = versioning_if->addBlock();
    auto false_block = versioning_if->addBlock();

    // Fill in the false block. It should contain the unoptimized
    // copy of the fused subgraph.
    WithInsertPoint guard(false_block->return_node());
    const auto subgraph_outputs = insertGraph(
        *fusion_group->owningGraph(), *subgraph, fusion_group->inputs());
    for (Value* output : subgraph_outputs) {
      false_block->registerOutput(output);
    }

    // types get copied to the fallback graph, so remove specializations before
    // replacing
    removeTensorTypeSpecializations2(false_block);
    replaceBlockWithFallbackGraph(false_block, fusion_group->inputs());

    // Fill in the true block. It has all inputs type-checked and its
    // body should be the fusion group node.
    fusion_group->moveBefore(true_block->return_node());
    for (size_t idx = 0; idx < fusion_group->inputs().size(); ++idx) {
      if (typechecked_inputs.count(fusion_group->input(idx))) {
        fusion_group->replaceInput(
            idx, typechecked_inputs.at(fusion_group->input(idx)));
      }
    }
    for (Value* output : fusion_group->outputs()) {
      true_block->registerOutput(output);
    }
  }

void runDiffGraphPasses(std::shared_ptr<Graph>& graph) {

  if (tensorExprFuserEnabled()) {
    // Remove prim::profile nodes and embed the profile info directly in the
    // IR in value types. We're doing such transformation as optimizations
    // that try to merge/fuse nodes in the graph (e.g. BatchMM and GraphFuser)
    // work worse in the presence of intermittent prim::profile nodes.
    // Optimizations relying on the type info are also responsible for
    // inserting proper type checks. Once we're done with these optimizations
    // we will wipe the tensor type information from the IR, so that it's not
    // accidentally used by any other pass.
    RemoveProfileNodesAndSpecializeTypes(graph);
  }

  GRAPH_DEBUG(
      "Before EliminateDeadCode (beginning of runDiffGraphPasses)\n", *graph);
  // runOptimization:
  {
    // Basic graph preprocessing to eliminate noise.
    EliminateDeadCode(graph);
    GRAPH_DEBUG(
        "After EliminateDeadCode, before EliminateCommonSubexpression\n",
        *graph);
    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before PeepholeOptimize\n",
        *graph);

    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG("After ConstantPropagation, before ConstantPooling\n", *graph);
    ConstantPooling(graph);
    GRAPH_DEBUG("After ConstantPooling, before UnrollLoops\n", *graph);

    UnrollLoops(graph);
    GRAPH_DEBUG("After UnrollLoops, before RemoveListMutation\n", *graph);
    // run again with unrolled loops
    RemoveListMutation(graph);
    GRAPH_DEBUG("After RemoveListMutation, before PeepholeOptimize\n", *graph);
    PeepholeOptimize(graph);
    GRAPH_DEBUG("After PeepholeOptimize, before ConstantPropagation\n", *graph);
    ConstantPropagation(graph);
    GRAPH_DEBUG(
        "After ConstantPropagation, before EliminateCommonSubexpression\n",
        *graph);

    EliminateCommonSubexpression(graph);
    GRAPH_DEBUG(
        "After EliminateCommonSubexpression, before CheckInplace\n", *graph);

    CheckInplace(graph);
  }
  GRAPH_DEBUG("After CheckInplace, before customPrePasses\n", *graph);

  // runNondiffOptimization
  {
    // Run custom passes that different backends can register.
    for (const auto& passPair : getCustomPrePasses()) {
      passPair.first(graph);
    }
    GRAPH_DEBUG("After customPrePasses, before LowerSimpleTuples\n", *graph);

    // TupleConstruct / TupleUnpack pairs can still be present at this point
    // and must be removed for fusion.
    LowerSimpleTuples(graph);
    GRAPH_DEBUG("After LowerSimpleTuples\n", *graph);

    if (tensorExprFuserEnabled()) {
      GRAPH_DEBUG(
          "After RemoveProfileNodesAndSpecializeTypes, before BatchMM\n",
          *graph);
      // Rewrite subgraphs with many MMs into expressions that batch them.
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

      FuseTensorExprs(graph, getFusionGroupInlining() ? 2 : 1);
      GRAPH_DEBUG(
          "After Fusion, before RemoveTensorTypeSpecializations\n", *graph);

      // Wipe tensor type info from the IR
      // RemoveTensorTypeSpecializations(graph);
      GRAPH_DEBUG(
          "After RemoveTensorTypeSpecializations, before customPostPasses\n",
          *graph);
    } else {
      // Rewrite subgraphs with many MMs into expressions that batch them.
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

      FuseGraph(graph, true);
      GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);
    }

    // Run custom post-fusion passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
  }
  GRAPH_DEBUG("After customPostPasses (end of runDiffGraphPasses)\n", *graph);
}

void runNoGradOptimizations(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "After customPostPasses (beginning of runNoGradOptimizations)\n", *graph);
  // runNondiffOptimization
  {
    // Run custom passes that different backends can register.
    for (const auto& passPair : getCustomPrePasses()) {
      passPair.first(graph);
    }
    GRAPH_DEBUG("After customPrePasses, before LowerSimpleTuples\n", *graph);

    // TupleConstruct / TupleUnpack pairs can still be present at this point
    // and must be removed for fusion.
    LowerSimpleTuples(graph);
    GRAPH_DEBUG("After LowerSimpleTuples\n", *graph);

    if (tensorExprFuserEnabled()) {
      // Remove prim::profile nodes and embed the profile info directly in the
      // IR in value types. We're doing such transformation as optimizations
      // that try to merge/fuse nodes in the graph (e.g. BatchMM and GraphFuser)
      // work worse in the presence of intermittent prim::profile nodes.
      // Optimizations relying on the type info are also responsible for
      // inserting proper type checks. Once we're done with these optimizations
      // we will wipe the tensor type information from the IR, so that it's not
      // accidentally used by any other pass.
      RemoveProfileNodesAndSpecializeTypes(graph);
      GRAPH_DEBUG(
          "After RemoveProfileNodesAndSpecializeTypes, before BatchMM\n",
          *graph);
      // Rewrite subgraphs with many MMs into expressions that batch them.
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

      FuseTensorExprs(graph, getFusionGroupInlining() ? 2 : 1);
      GRAPH_DEBUG(
          "After Fusion, before RemoveTensorTypeSpecializations\n", *graph);

      // Wipe tensor type info from the IR
      RemoveTensorTypeSpecializations(graph);
      GRAPH_DEBUG(
          "After RemoveTensorTypeSpecializations, before customPostPasses\n",
          *graph);
    } else {
      // Rewrite subgraphs with many MMs into expressions that batch them.
      BatchMM(graph);
      GRAPH_DEBUG("After BatchMM, before Fusion\n", *graph);

      FuseGraph(graph, true);
      GRAPH_DEBUG("After Fusion, before customPostPasses\n", *graph);
    }

    // Run custom post-fusion passes
    for (const auto& passPair : getCustomPostPasses()) {
      passPair.first(graph);
    }
  }
  GRAPH_DEBUG(
      "After customPostPasses (end of runNoGradOptimizations)\n", *graph);
}

void ProfilingGraphExecutorImpl::runProfilingOptimizations(
    std::shared_ptr<Graph>& copy) {
  GRAPH_DEBUG("Before runProfilingOptimizations:\n", *copy);
  if (!getGraphExecutorOptimize()) {
    runNooptPassPipeline(copy);
    return;
  }

  runPreAutodiffPassPipeline(copy);

  if (needsGradientInProfilingMode(copy->block())) {
    auto diff_nodes = CreateAutodiffSubgraphs(
        copy,
        getAutodiffSubgraphInlining() ? autodiffSubgraphNodeThreshold : 1);
    GRAPH_DEBUG("After CreateAutodiffSubgraphs\n", *copy);
    size_t idx = 0;
    for (Node* dnode : diff_nodes) {
      GRAPH_DEBUG("Optimizing diff node ", idx);

      auto unopt_dnode = copy->createClone(dnode, [&](Value* v) { return v; });
      auto diff_graph = std::move(dnode->g(attr::Subgraph));
      unopt_dnode->g_(attr::Subgraph, nullptr);
      auto unopt_diff_graph = diff_graph->copy();
      

      
      // we specialize before differntiate so we can avoid
      // computing gradients for tensors that don't require them
      if (tensorExprFuserEnabled()) {
        RemoveProfileNodesAndSpecializeTypes(diff_graph);
      }
      Gradient gradient = differentiate(diff_graph);

      std::vector<TypePtr> profiled_types;
      for (auto i : gradient.f->inputs()) {
        profiled_types.push_back(i->type());
      }
      GRAPH_DEBUG("Forward graph:\n", *(gradient.f));
      GRAPH_DEBUG("Backward graph:\n", *(gradient.df));
      runDiffGraphPasses(gradient.f);
      packGradient(gradient, dnode);

      RemoveProfilingNodes(unopt_diff_graph);
      Gradient unopt_gradient = differentiate(unopt_diff_graph);
      packGradient(unopt_gradient, unopt_dnode);


    // Fixup types of the subgraph inputs
    std::vector<Value*> inputs_to_check;
    std::vector<size_t> input_indices_to_check;
    //for (Value* input : dnode->inputs()) {
    for (size_t i = 0; i < dnode->inputs().size(); i++) {
      Value* input = dnode->input(i);
      // We only check inputs of the fusion group and expect NNC to infer
      // intermediates and outputs shapes
      // TODO also check NumberTypes
      if (!input->type()->cast<TensorType>()) {
        continue;
      }

      if (input->node()->kind() == prim::Constant) {
        continue;
      }
      inputs_to_check.push_back(input);
      input_indices_to_check.push_back(i);
    }

    if (!inputs_to_check.size()) {
      return;
    }

    Node* typecheck_node =
        copy->create(
                prim::TypeCheck, inputs_to_check, inputs_to_check.size() + 1)
            ->insertBefore(dnode);
    Value* typecheck_result = typecheck_node->output(inputs_to_check.size());
    typecheck_result->setType(BoolType::get());

    auto versioning_if =
    copy->create(
            prim::If, {typecheck_result}, dnode->outputs().size())
        ->insertAfter(typecheck_node);


    auto true_block = versioning_if->addBlock();
    auto false_block = versioning_if->addBlock();
    false_block->appendNode(unopt_dnode);
    
    for (auto v: unopt_dnode->outputs()) {
      false_block->registerOutput(v);
    }

    for (size_t idx = 0; idx < dnode->outputs().size(); ++idx) {
      versioning_if->output(idx)->setType(dnode->output(idx)->type());
      dnode->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
    }

    dnode->moveBefore(*true_block->nodes().begin());
    for (auto v: dnode->outputs()) {
      true_block->registerOutput(v);
    }

    GRAPH_DEBUG("Before replaceFallbackGraphWithFallbackFunction graph:\n", *copy);
    replaceBlockWithFallbackGraph(false_block, dnode->inputs());
    replaceFallbackGraphWithFallbackFunction(false_block);       
    GRAPH_DEBUG("After replaceFallbackGraphWithFallbackFunction graph:\n", *copy);

    GRAPH_DUMP("Before fix types: ", copy);
    // Fixup types of the typecheck node outputs, which are used by the op in
    // execution
    auto opt_graph = dnode->g(attr::Subgraph);
    for (size_t i = 0; i < input_indices_to_check.size(); ++i) {
      typecheck_node->output(i)->setType(profiled_types[input_indices_to_check[i]]);
      GRAPH_DEBUG("Replacing %", dnode->input(input_indices_to_check[i])->debugName(), " with ", typecheck_node->output(i)->debugName(), " after ", *typecheck_node);
      dnode->replaceInput(input_indices_to_check[i], typecheck_node->output(i));
    }
    
    GRAPH_DEBUG("Finished optimizing diff node ", idx++);
    }
    RemoveProfilingNodes(copy);
    // InlineAutodiffSubgraphs(
    //     copy,
    //     getAutodiffSubgraphInlining() ? autodiffSubgraphInlineThreshold : 1);
    
    GRAPH_DEBUG(
        "After InlineAutodiffSubgraphs and Removing Profiling Nodes\n", *copy);
  } else {
    runNoGradOptimizations(copy);
  }
  EliminateDeadCode(copy);
  GRAPH_DEBUG("After runProfilingOptimizations:\n", *copy);
}

void ProfilingGraphExecutorImpl::runProfilingInsensitiveOptimizations(
    std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before inlining (beginning of runProfilingInsensitiveOptimizations)\n",
      *graph);
  // TODO: maybe this can go later in pipeline / directly in autodiff forward
  // creation
  if (getGraphExecutorOptimize()) {
    Inline(*graph);
  }
  GRAPH_DEBUG("After inlining, before ClearProfilingInformation\n", *graph);
  ClearProfilingInformation(graph);
  GRAPH_DEBUG("After ClearProfilingInformation, before LowerGradOf\n", *graph);
  LowerGradOf(*graph);
  GRAPH_DEBUG("After LowerGradOf, before ClearUndefinedness\n", *graph);
  // clear any residual undefinedness
  // as double backward graph inputs'
  // may carry over undefinedness
  // from profiled backward graphs
  ClearUndefinedness(graph);
  // runRequiredPasses
  {
    GRAPH_DEBUG("After ClearUndefinedness, before RemoveExpands\n", *graph);
    RemoveExpands(graph);
    GRAPH_DEBUG("After RemoveExpands, before CanonicalizeOps\n", *graph);
    CanonicalizeOps(graph);
    GRAPH_DEBUG("After CanonicalizeOps, before EliminateDeadCode\n", *graph);
    EliminateDeadCode(graph);
  }
  if (!getGraphExecutorOptimize()) {
    GRAPH_DEBUG(
        "After EliminateDeadCode (end of runProfilingInsensitiveOptimizations)\n",
        *graph);
    return;
  }

  GRAPH_DEBUG("After EliminateDeadCode, before DecomposeOps\n", *graph);
  DecomposeOps(graph);
  GRAPH_DEBUG("After DecomposeOps, before ConstantPropagation\n", *graph);
  ConstantPropagation(graph);
  GRAPH_DEBUG("After ConstantPropagation, before EliminateDeadCode\n", *graph);
  EliminateDeadCode(graph);
  GRAPH_DEBUG(
      "After EliminateDeadCode, before EliminateCommonSubexpression\n", *graph);
  EliminateCommonSubexpression(graph);
  GRAPH_DEBUG(
      "After EliminateCommonSubexpression, before ConstantPooling\n", *graph);
  ConstantPooling(graph);
  GRAPH_DEBUG("After ConstantPooling, before PeepholeOptimize\n", *graph);
  PeepholeOptimize(graph);
  GRAPH_DEBUG("After PeepholeOptimize, before EliminateDeadCode\n", *graph);
  EliminateDeadCode(graph);
  GRAPH_DEBUG("After EliminateDeadCode, before LowerSimpleTuples\n", *graph);
  LowerSimpleTuples(graph);
  GRAPH_DEBUG("After LowerSimpleTuples, before CheckInplace\n", *graph);
  CheckInplace(graph);
  GRAPH_DEBUG(
      "After CheckInplace (end of runProfilingInsensitiveOptimizations)\n",
      *graph);
}

ProfilingGraphExecutorImpl::ProfilingGraphExecutorImpl(
    const std::shared_ptr<Graph>& graph,
    std::string function_name)
    : GraphExecutorImplBase(graph, std::move(function_name)) {}

const ExecutionPlan& ProfilingGraphExecutorImpl::getOptimizedPlanFor(
    Stack& stack,
    size_t remaining_bailout_depth) {
  GRAPH_DEBUG("Running ProfilingGraphExecutorImpl ", this);

  // no opt mode
  if (!getGraphExecutorOptimize()) {
    if (!fallback_plan_) {
      auto copy = graph->copy();
      GRAPH_DEBUG(
          "Before LowerGradOf (beginning of runNooptPassPipeline)\n", *graph);
      LowerGradOf(*copy);
      GRAPH_DEBUG("After LowerGradOf, before RemoveExpands\n", *graph);
      RemoveExpands(copy);
      fallback_plan_ = ExecutionPlan(copy, function_name_);
      GRAPH_DUMP("NoOpt Graph: ", copy);
    }
    return *fallback_plan_;
  }

  // if tensorExprFuserEnabled() returns true we need to persist the very first
  // time ProfilingGraphExecutorImpl is called, so we can update it correctly
  // for fallback functions in ProfilingGraphExecutorImpl Else,
  // getPlanFor(remaining_bailout_depth) is corrected and persisted by the Code
  // object in interpreter.
  if (!remaining_bailout_depth_.has_value() || !tensorExprFuserEnabled()) {
    remaining_bailout_depth_ = remaining_bailout_depth;
  }

  // simple executor
  if (*remaining_bailout_depth_ == 0) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    GRAPH_DUMP("Optimized SimpleExecutor Graph: ", copy);
    optimized_plan_ = ExecutionPlan(copy, function_name_);
    return *optimized_plan_;
  }

  // if a profiling graph hasn't been created yet
  if (!pr_) {
    auto copy = graph->copy();
    runProfilingInsensitiveOptimizations(copy);
    pr_ = ProfilingRecord::instrumentGraph(copy);
    GRAPH_DUMP("Profiled Graph: ", pr_->graph());
    profiling_plan_ = ExecutionPlan(pr_->graph(), function_name_);
    // fall-through
  }

  // profile until a graph is ready
  if (!pr_->ready()) {
    return *profiling_plan_;
  }

  auto copy = pr_->graph()->copy();
  ProfilingRecord::removeProfileCounter(copy->block());
  runProfilingOptimizations(copy);
  // replaces a fallback graph inserted by
  // specialize_autogradzero if one exists
  replaceFallbackGraphWithFallbackFunction(copy->block());
  GRAPH_DUMP("Optimized Graph: ", copy);
  optimized_plan_ =
      ExecutionPlan(copy, function_name_, *remaining_bailout_depth_);
  return *optimized_plan_;
}

const ExecutionPlan& ProfilingGraphExecutorImpl::getPlanFor(
    Stack& stack,
    size_t remaining_bailout_depth) {
  std::lock_guard<std::mutex> lock(compile_mutex);

  // IMPORTANT: This is a hot path of calling a torchscript function. Try not to
  // add any code above this.
  if (optimized_plan_) {
    return *optimized_plan_;
  }

  return getOptimizedPlanFor(stack, remaining_bailout_depth);
}

GraphExecutorState ProfilingGraphExecutorImpl::getDebugState() {
  GraphExecutorState state;
  TORCH_INTERNAL_ASSERT(optimized_plan_);
  auto opt_plan = *optimized_plan_;
  state.execution_plans.emplace(ArgumentSpec{0, 0}, opt_plan);
  return state;
}

Node* insertFallbackFunctionCall(
    Graph* graph,
    Function* func,
    ArrayRef<Value*> inputs) {
  auto tuple_type = func->graph()->return_node()->input(0)->type();
  Value* fn_constant = graph->insertNode(graph->create(prim::Constant))
                           ->s_(attr::name, func->name())
                           ->i_(Symbol::attr("fallback"), 1)
                           ->output()
                           ->setType(FunctionType::create(func));
  std::vector<Value*> func_call_inputs = {fn_constant};
  func_call_inputs.insert(func_call_inputs.end(), inputs.begin(), inputs.end());
  Value* result =
      graph->insertNode(graph->create(prim::CallFunction, func_call_inputs))
          ->output()
          ->setType(tuple_type);

  auto fun_unpack_tuple = graph->insertNode(graph->createTupleUnpack(result));
  return fun_unpack_tuple;
}

Function* createFallbackPathFunction(
    Block* b,
    const std::string& function_name) {
  auto value_map = [](Value* v) { return v; };
  auto graph = std::make_shared<Graph>();
  graph->block()->cloneFrom(b, value_map);

  auto otypes = c10::fmap(
      graph->return_node()->inputs(), [](Value* v) { return v->type(); });
  // a GraphFunction call only have one output, so all the outputs
  // need to be packed into a tuple
  auto tuple_type = TupleType::create(otypes);
  auto return_tuple = graph->createTuple(graph->return_node()->inputs());
  graph->appendNode(return_tuple);
  for (int i = static_cast<int>(graph->outputs().size()) - 1; i >= 0; i--) {
    graph->eraseOutput(i);
  }
  graph->registerOutput(return_tuple->output());
  return new GraphFunction(function_name, graph, nullptr);
}

void ProfilingGraphExecutorImpl::replaceFallbackGraphWithFallbackFunction(
    Block* b) {
  Stack s;
  for (auto it = b->nodes().begin(); it != b->nodes().end();) {
    if (it->kind() == prim::FallbackGraph) {
      auto fallback_func = createFallbackPathFunction(
          it->g(attr::Subgraph)->block(), "fallback_function");
      TORCH_INTERNAL_ASSERT(*remaining_bailout_depth_ > 0);
      GRAPH_DEBUG(
          "getPlanFor for", getHeader(*it), " ", *remaining_bailout_depth_);
      fallback_func->get_executor().getPlanFor(
          s, *remaining_bailout_depth_ - 1);
      fallback_functions_.emplace_back(fallback_func);
      WithInsertPoint wip{*it};
      auto function_call = insertFallbackFunctionCall(
          b->owningGraph(), fallback_func, it->inputs());
      for (size_t i = 0; i < function_call->outputs().size(); i++) {
        it->output(i)->replaceAllUsesWith(function_call->output(i));
      }
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        replaceFallbackGraphWithFallbackFunction(ib);
      }
      it++;
    }
  }
}

} // namespace jit
} // namespace torch
