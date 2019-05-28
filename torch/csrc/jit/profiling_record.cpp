#include <torch/csrc/jit/profiling_record.h>

namespace torch {
namespace jit {

ProfilingRecord::ProfilingRecord(std::shared_ptr<Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(3) {}

ProfileOp* ProfilingRecord::createProfileNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = new ProfileOp(profiled_graph_.get(), fp);

  for (auto in : inputs) {
    pn->addInput(in);
  }
  return pn;
}

void ProfilingRecord::instrumentBlock(Block* block) {

  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (auto i : n->inputs()) {
      if (!i->type()->isSubclass(TypeKind::TensorType) || i->node()->kind() == prim::profile) {
        continue;
      }

      auto pn = createProfileNode(nullptr, {i});
      auto pno = pn->addOutput();
      pno->setType(i->type());
      std::function<void(Stack&)> shape_profiler = [this, pno](Stack& stack) {
        IValue t;
        pop(stack, t);
        if (t.isTensor()) {
          auto pttp = ProfiledTensorType::create(t.toTensor());
          std::lock_guard<std::mutex> lock(this->mutex_);
          if (pno->type()->isSubclass(TypeKind::ProfiledTensorType)) {
            auto type = pno->type()->cast<ProfiledTensorType>();
            pno->setType(type->merge(pttp));
          } else {
            pno->setType(pttp);
          }
        }
        //passing t through
        push(stack, t);
      };

      pn->setCallback(shape_profiler);
      pn->insertBefore(n);
      n->replaceInputWith(i, pn->output());
    }

    for (auto b : n->blocks()) {
      instrumentBlock(b);
    }
  }
}

/*
void ProfilingRecord::instrumentBlock2(Block* block) {
  // iterating backwards allows us to easily insert profile nodes
  // without affecting an iterator
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;

    for (auto o : n->inputs()) {
      // only insert a profile instruction if an input is a tensor
      // and we don't already have a profile instruction on this value
      // this typically happens when the same value is used 2+ in n->inputs()
      // and it has already been replaced with the profile instruction
      // for some previous input
      if (!o->type()->isSubclass(TypeKind::TensorType) || o->node()->kind() == prim::profile) {
        continue;
      }

      std::function<void(Stack&)> shape_profiler = [this, o](Stack& stack) {
        IValue t;
        pop(stack, t);
        if (t.isTensor()) {
          auto pttp = ProfiledTensorType::create(t.toTensor());
          std::lock_guard<std::mutex> lock(this->mutex_);
          if (o->type()->isSubclass(TypeKind::ProfiledTensorType)) {
            auto type = o->type()->cast<ProfiledTensorType>();
            o->setType(type->merge(pttp));
          } else {
            o->setType(pttp);
          }
        }
      };

      auto pn = createProfileNode(shape_profiler, {o});
      auto pn_o = pn->addOutput();
      pn_o->setType(o->type());
      pn->insertBefore(n);
      n->replaceInputWith(o, pn->output());
    }

    for (auto b : n->blocks()) {
      instrumentBlock2(b);
    }
  }
}
*/

std::unique_ptr<ProfilingRecord> ProfilingRecord::instrumentGraph(
    const std::shared_ptr<Graph>& graph) {
  auto new_g = graph->copy();
  auto pr = std::unique_ptr<ProfilingRecord>(new ProfilingRecord(new_g));
  auto raw_pr = pr.get();

  pr->instrumentBlock(new_g->block());
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
