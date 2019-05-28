#include <unordered_set>
#include <memory>
#include <torch/csrc/jit/passes/guard_elimination.h>
#include <torch/csrc/jit/passes/alias_analysis.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

  struct GuardElimination {

    GuardElimination(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph))
    , aliasDb_(caffe2::make_unique<AliasDb>(graph_))
    {

    }

    void moveGuardsToDefs(Block* b)
    {
      // alias db gets confused if we ask it to move expressions
      // around prim::load, so we insert a dummy anchor to
      // which we will be moving qualifying uses of params
      auto start_node = b->owningGraph()->create(prim::Constant, 1);
      start_node->output()->setType(IntType::create());
      // the answer to the mystery of life
      start_node->i_(attr::value, 42);
      start_node->insertAfter(*b->nodes().begin());
      for (auto it = b->nodes().begin(); it != b->nodes().end();)
      {
        auto n = *it;
        if (n->kind() == prim::Guard)
        {
          //grab the next node before we move this one all the way back
          it++;
          auto guardee = n->inputs().at(0)->node();
          if (guardee->kind() == prim::Param)
          {
            guardee = start_node;
          }
          if (guardee->owningBlock() != n->owningBlock())
          {
            guardee = *n->owningBlock()->nodes().begin();
          }
          //std::cout << "guardee = " << *guardee << std::endl;
          //std::cout << "moving " << n->output()->uniqueName() << " after "
          //<< n->inputs().at(0)->uniqueName() << std::endl;
          //std::cout << "next value " << it->outputs().at(0)->uniqueName() << std::endl;
          aliasDb_->moveAfterTopologicallyValid(n, guardee);
          //std::cout << "moved " << moved;
        }
        else
        {
          it++;
          for (Block* ib : n->blocks())
          {
            moveGuardsToDefs(ib);
          }
        }
      }
    }

    void collectOpTypes(Block* b)
    {
      for (auto n : b->nodes())
      {
        if (!op_types.count(n->kind()))
        {
          op_types.insert({n->kind(), n});
        }

        for (auto ib : n->blocks())
        {
          collectOpTypes(ib);
        }
      }
    }
    void printOpSummary()
    {
      collectOpTypes(graph_->block());
      std::cout << "op summary\n";
      for (auto e : op_types)
      {
        if (e.first != prim::Loop && e.first != prim::If)
        {
          std::cout << *e.second;
        }
      }
    }

    std::unordered_map<NodeKind, Node*> op_types;

    void coalesceGuards(Block* b)
    {
      std::unordered_map<Value*, Node*> inputs_to_guards;
      for (auto it = b->nodes().begin(); it != b->nodes().end(); it++)
      {
        /*
        %24 : Tensor(dtype = Float , shape = (2, 3) = prim::Guard(%x)
       %23 : Tensor(dtype = Float , shape = (2, 3) = prim::Guard(%y)
       %22 : Tensor(dtype = Float , shape = (2, 3) = prim::Guard(%x)
       %21 : Tensor(dtype = Float , shape = (2, 3) = prim::Guard(%y)
       %20 : Tensor(dtype = Float , shape = (2, 3) = prim::Guard(%x)
       %a : Tensor = aten::add(%20, %21, %2)
        */
        auto n = *it;
        if (n->kind() == prim::Guard)
        {
          std::cout << "looking at guard " << n->output()->uniqueName() << std::endl;
          if (inputs_to_guards.count(n->input()))
          {
            auto prev = inputs_to_guards[n->input()];
            std::cout << "coalescing guard " << n->output()->uniqueName()
            << " with " << prev->output()->uniqueName() << std::endl;
            n->output()->replaceAllUsesWith(prev->output());
            it.destroyCurrent();
          }
          else
          {
            inputs_to_guards.insert({n->input(), n});
          }
        }
        else if (n->kind() != prim::Constant)
        {
          inputs_to_guards.clear();
          for (Block* ib : n->blocks())
          {
            coalesceGuards(ib);
          }
        }
      }
    }

    void eliminateGuards(Block* b)
    {
      for (auto it = b->nodes().rbegin(); it != b->nodes().rend();)
      {
        auto n = *it;
        if (n->kind() == prim::Guard && removableGuard(n->inputs().at(0)->node()))
        {
          std::cout << "eliminating " << n->output()->uniqueName() << std::endl;
          auto pttp = n->output()->type();
          n->output()->replaceAllUsesWith(n->inputs().at(0));
          n->inputs().at(0)->setType(pttp);
          it.destroyCurrent();
        }
        else
        {
          it++;
          for (Block* ib : n->blocks())
          {
            eliminateGuards(ib);
          }
        }
      }
    }
private:

    bool removableGuard(Node* n)
    {
      if (!simple_ops_.count(n->kind()))
      {
        return false;
      }

      bool all_inputs_guarded = true;
      for (auto input : n->inputs())
      {
        if (input->node()->kind() == prim::Guard || input->node()->kind() == prim::Constant)
        {
          AT_ASSERT(input->node()->kind() != prim::Guard || input->type()->expect<ProfiledTensorType>());
        }
        else
        {
          all_inputs_guarded = false;
          break;
        }
      }
      return all_inputs_guarded;
    }

    std::shared_ptr<Graph> graph_;
    std::unique_ptr<AliasDb> aliasDb_;
     static std::unordered_set<Symbol> simple_ops_;
  };


std::unordered_set<Symbol> GuardElimination::simple_ops_ = {aten::add, aten::sub, aten::mul, aten::div};

static void removeProfilingNodes(Block* b)
{
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++)
  {
    if (it->kind() == prim::profile)
    {
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

void EliminateGuards(std::shared_ptr<Graph> graph) {
  GuardElimination ge(graph);
  std::cout << "before moving guards\n";
  graph->dump();
  ge.moveGuardsToDefs(graph->block());
  std::cout << "before coalescing guards\n";
  graph->dump();
  ge.coalesceGuards(graph->block());
  EliminateDeadCode(graph);
  LowerSimpleTuples(graph);
  std::cout << "before eliminating guards\n";
  graph->dump();
  ge.printOpSummary();
  // ge.eliminateGuards(graph->block());
}

} // namespace jit
} // namespace torch
