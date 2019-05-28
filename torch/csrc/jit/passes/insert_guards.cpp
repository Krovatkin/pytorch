#include <unordered_set>
#include <memory>
#include <torch/csrc/jit/passes/insert_guards.h>

namespace torch {
namespace jit {

  struct GuardInserter {

    GuardInserter(std::shared_ptr<Graph> graph):
    graph_(std::move(graph)) {}

    void run ()
    {
      run(graph_->block());
    }

private:

    void guardify_uses(Value* v)
    {
      // TODO: why not one guard per all uses of a value?
      auto uses = use_list(v->uses().begin(), v->uses().end());
      for (auto use : uses)
      {
        auto guard = graph_->create(prim::Guard, {v}, 1);
        guard->output()->setType(v->type());
        auto user = use.user;
        guard->insertBefore(user);
        user->replaceInput(use.offset, guard->output());
      }
      v->setType(TensorType::create());
    }

    void run(Block* b) {

      for (auto n : b->nodes())
      {
        if (n->kind() == prim::Guard)
        {
          continue;
        }

        for (auto v : n->outputs())
        {
          if (auto pttp = v->type()->cast<ProfiledTensorType>())
          {
            guardify_uses(v);
          }
        }

        for (Block* ib : n->blocks())
        {
          run(ib);
        }
      }
    }

    std::shared_ptr<Graph> graph_;
    std::unordered_set<Value*> processed_values;

  };


  struct GuardInserter2 {

    GuardInserter2(std::shared_ptr<Graph> graph):
    graph_(std::move(graph)) {}

    void run ()
    {
      run(graph_->block());
    }

private:


    void run(Block* b) {

      for (auto it = b->nodes().begin(); it != b->nodes().end(); it++)
      {
        auto n = *it;
        if (n->kind() == prim::profile && n->outputs().size() == 1)
        {
          // n->input() is Tensor type
          auto guard = graph_->create(prim::Guard, {n->input()}, 1);
          auto go = guard->output();
          // borrow profiling information
          // TODO: we should really make a copy in case
          // we are still collecting information
          go->setType(n->output()->type());
          guard->insertBefore(n);
          n->output()->replaceAllUsesWith(go);
          // remove a prim::profile
          // TODO: should we just leave this up DCE
          // then we also need to make sure alias analysis
          // works well with prim::profile
          std::cout << "destroying \n";
          it->dump();
          it.destroyCurrent();
        }
        else
        {
          for (Block* ib : n->blocks())
          {
            run(ib);
          }
        }
      }
    }

    std::shared_ptr<Graph> graph_;


  };


static void removeProfilingNodes(Block* b)
{
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++)
  {
    if (it->kind() == prim::profile)
    {
      std::cout  << "destroying\n";
      it->dump();
      it.destroyCurrent();
    }
    else
    {
      for (Block* ib : it->blocks())
      {
        removeProfilingNodes(ib);
      }
    }
  }
}

void InsertGuards(std::shared_ptr<Graph> graph) {
  GuardInserter2 gi(graph);
  gi.run();
  removeProfilingNodes(graph->block());
  std::cout << "remove profiling instructions\n";
  graph->dump();
}

} // namespace jit
} // namespace torch
