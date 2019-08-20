#include <torch/csrc/jit/profiling_record.h>
#include <torch/csrc/jit/passes/constant_propagation.h>

namespace torch {
namespace jit {

ProfilingRecord::ProfilingRecord(std::shared_ptr<Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(1) {}

ProfileOp* ProfilingRecord::createProfileNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = new ProfileOp(profiled_graph_.get(), fp);

  for (auto in : inputs) {
    pn->addInput(in);
  }
  return pn;
}

static void unprofileGraphInputs(const std::shared_ptr<Graph>& graph)
{
  for (auto i : graph->inputs())
  {
    if (i->type()->isSubclass(TypeKind::TensorType)) {
      i->setType(TensorType::get());
    }
  }
}

static void unprofileBlock(Block* block)
{
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (auto o : n->outputs()) {
      if (o->type()->isSubclass(TypeKind::TensorType)) {
        o->setType(TensorType::get());
      }
    }

    for (auto b : n->blocks()) {
      unprofileBlock(b);
    }
  }
}

void ProfilingRecord::insertShapeProfile(Node* n, Value* i) {

      auto pn = createProfileNode(nullptr, {i});
      auto pno = pn->addOutput();
      pno->setType(i->type());
      std::function<void(Stack&)> shape_profiler = [this, pno](Stack& stack) {
        IValue t;
        pop(stack, t);
        if (t.isTensor()) {

          if (t.toTensor().defined())
          {
            std::cout << "pno = " << pno->debugName() << std::endl;
            auto pttp = ProfiledTensorType::create(t.toTensor());
            std::lock_guard<std::mutex> lock(this->mutex_);
            if (pno->type()->isSubclass(TypeKind::ProfiledTensorType)) {
              auto type = pno->type()->cast<ProfiledTensorType>();
              std::cout << "pno undefined  = " << type->is_undefined_grad_tensor() << std::endl;
              std::cout << "pttp = " << pttp << " " << *pttp << std::endl;
              auto merged = type->merge(pttp);
              std::cout << "merged = " << merged << " " << *merged << std::endl;
              pno->setType(merged);
            } else {
              pno->setType(pttp);
            }
          }
          else
          {
            
            pno->setType(ProfiledTensorType::createUndefinedTensorGrad());
            std::cout << "setting undefined tensor type for value " << pno->debugName() << " type = " << pno->type() << " type = " << *pno->type() << std::endl;
          }
        }
        // passing t through
        push(stack, t);
      };

      pn->setCallback(shape_profiler);
      pn->insertBefore(n);
      n->replaceInputWith(i, pn->output());
}

void ProfilingRecord::instrumentBlock(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (auto i : n->inputs()) {
      if (!i->type()->isSubclass(TypeKind::TensorType) ||
          i->node()->kind() == prim::profile) {
        continue;
      }

      insertShapeProfile(n, i);
    }

    for (auto b : n->blocks()) {
      instrumentBlock(b);
    }
  }
}

std::unique_ptr<ProfilingRecord> ProfilingRecord::instrumentGraph(
    const std::shared_ptr<Graph>& graph) {
  auto new_g = graph->copy();
  auto pr = std::unique_ptr<ProfilingRecord>(new ProfilingRecord(new_g));
  auto raw_pr = pr.get();
  
  unprofileGraphInputs(new_g);
  unprofileBlock(new_g->block());
  std::cout << "after unprofiling\n";
  new_g->dump();
  pr->instrumentBlock(new_g->block());



  std::cout << "return node = " << *new_g->return_node() << std::endl;

  for (auto i : new_g->return_node()->inputs()) {
    pr->insertShapeProfile(new_g->return_node(), i);
  }
  std::function<void(Stack&)> counter = [raw_pr](Stack&) {
    std::lock_guard<std::mutex> lock(raw_pr->mutex_);
    if (raw_pr->profiling_count_ > 0)
    {
        raw_pr->profiling_count_--;
    }
  };

  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);
  return pr;
}

ProfiledTensorTypePtr ProfilingRecord::toProfiledTensorTypePtr(
    const IValue& ival) {
  if (ival.isTensor()) {
    auto tensor = ival.toTensor();
    return ProfiledTensorType::create(tensor);
  }

  return {nullptr};
}

} // namespace jit
} // namespace torch
