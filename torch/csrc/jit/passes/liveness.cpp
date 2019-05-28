#include <unordered_set>
#include <memory>
#include <torch/csrc/jit/passes/bailout_graph.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

namespace torch {
namespace jit {

  struct BailoutGraph {

    BailoutGraph(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph)) {}


    Value* addNewInputForValue(Value* old_value, Value* new_value = nullptr)
    {
      auto node = old_value->node();
      if (node->kind() == prim::Constant)
      {
          auto new_const = copy_graph_->createClone(node, {nullptr});
          copy_graph_->block()->appendNode(new_const);
          return new_const->output();
      }

      if (!new_value)
      {
        new_value = copy_graph_->block()->addInput();
        //std::cout << "adding a new value " << new_value->uniqueName() << " for " << old_value << std::endl;
        //live_inputs_[new_value] = old_value;
        live_inputs_.push_back(old_value);
      }
      this->old_to_new_[old_value] = new_value;
      new_value->copyMetadata(old_value);
      //std::cout << "mapping " << old_value->uniqueName() << "  to " << new_value->uniqueName() << std::endl;
      return new_value;
    }

    void buildBailOutBlockFrom(Node* n)
    {
      std::cout << "building block for " << n->outputs().at(0)->uniqueName() << std::endl;
      auto outer_node = n->owningBlock()->owningNode();
      auto* block = copy_graph_->block();
      if (n->kind() == prim::Loop)
      {
        auto new_max_count =  addNewInputForValue(n->inputs()[0]);
        auto cur_iter =  addNewInputForValue(n->blocks()[0]->inputs()[0]);
        auto updated_max_trip_count = copy_graph_->create(aten::sub);
        block->appendNode(updated_max_trip_count);
        updated_max_trip_count->addInput(new_max_count);
        updated_max_trip_count->addInput(cur_iter);
        addNewInputForValue(n->inputs()[0], updated_max_trip_count->output());
      }
      else if (n->kind() == prim::If)
      {
        //nothing to do; outputs should've already been mapped properly
        n = n->next();
      }

      auto b = n->owningBlock();
      graph_node_list_iterator it(n, kNextDirection);
      for (; it != b->nodes().end(); it++)
      {
        //std::cout << "processing node " << *it << std::endl;
        auto env = [this](Value* v)
        {
          auto new_value = (this->old_to_new_.count(v) == 0) ? nullptr : this->old_to_new_[v];
          return addNewInputForValue(v, new_value);
          // if (this->old_to_new_.count(v) == 0)
          // {
          //
          //   // TODO: figure out how ordering of this will work will loops
          //   // since we aren't starting from the first node
          //   auto new_output = this->old_to_new_[v] = block->addInput();
          //   live_inputs_[block->addInput()] = v;
          //   v->copyMetadata(new_output);
          //   return new_output;
          // }
          //
          // return this->old_to_new_[v];
        };
        auto node = *it;

        auto new_node = block->appendNode(copy_graph_->createClone(node, env));
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          auto oo = node->outputs()[i];
          auto no = new_node->outputs()[i];
          old_to_new_[oo] = no;
          no->copyMetadata(oo);
      }
      //std::cout << "mapping value " << node->outputs().at(0)->uniqueName() << " to " << new_node->outputs().at(0)->uniqueName() << std::endl;
    }

    if (outer_node)
    {
      //std::cout << "in outer_node\n";
      //support for case

      //dumpValueMap();
      auto block_outputs = n->owningBlock()->outputs();
      // skip the first input for loops (current iteration count)
      size_t i = outer_node->kind() == prim::Loop;
      auto new_outputs = outer_node->kind() == prim::Loop ? outer_node->inputs() : outer_node->outputs();
      for (; i < block_outputs.size(); i++)
      {
        //std::cout << "registering " << block_outputs.at(i)->uniqueName() << " " << block_outputs.at(i) << std::endl;
        auto nv = old_to_new_[block_outputs[i]];
        old_to_new_[new_outputs.at(i)] = nv;
      }
      buildBailOutBlockFrom(outer_node);
    }
  }

    void removeGuards(Block* b)
    {
      for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it)
      {
        if (it->kind() == prim::Guard)
        {
          it->output()->replaceAllUsesWith(it->input());
          it.destroyCurrent();
        }
        else
        {
          for (auto ib : it->blocks())
          {
            removeGuards(ib);
          }
        }
      }
    }

    std::shared_ptr<Graph> buildBailOutGraphFrom(Node* n)
    {
      std::cout << "building bailout graph for " << n->outputs().at(0)->uniqueName() << std::endl;
      //AT_ASSERT(n->owningGraph() == graph_.get());
      old_to_new_.clear();
      copy_graph_ = std::make_shared<Graph>();
      guard_ = n;
      //add graph outputs
      buildBailOutBlockFrom(n);
      //copy_graph_->dump();
      for (auto ov : graph_->outputs())
      {
        auto nv = addNewInputForValue(ov);
        copy_graph_->registerOutput(nv);
      }

      removeGuards(copy_graph_->block());
      return copy_graph_;
    }

    Node* locatePrint(Block* b) {

        for (auto n : b->nodes())
        {
          if (n->kind() == prim::Print)
          {
            return n;
          }
          else
          {
            for (auto ib : n->blocks())
            {
              if (auto print = locatePrint(ib))
              {
                return print;
              }
            }
          }
        }

        return nullptr;
    }

    void dumpValueMap () {

      //std::cout << "old_to_new_\n";
      for (auto e : old_to_new_)
      {
        std::cout << e.first->uniqueName() << " " << e.first << " -> "
        << e.second->uniqueName() << " " << e.second << std::endl;
      }
    }

    std::shared_ptr<Graph> graph_;
    std::shared_ptr<Graph> copy_graph_;
    //std::unordered_map<Value*, Value*> live_inputs_;
    std::vector<Value*> live_inputs_;
    Node* guard_;
    std::unordered_map<Value*, Value*> old_to_new_;
  };


struct InsertBailOuts
{
  InsertBailOuts(std::shared_ptr<Graph> graph):
  graph_(graph) {}

  void run()
  {
    insertBailOuts(graph_->block());
    setSubgraphs();
  }

  void sanitizeGraph(Block* b)
  {
    //TODO: remove Guards and BailOuts in bail-out graphs
  }

  void setSubgraphs()
  {
    for (auto e : subgraphs)
    {
      sanitizeGraph(e.second->block());
      e.first->g_(attr::Subgraph, e.second);
    }
  }
  void insertBailOuts(Block* b)
  {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it)
    {
      if (it->kind() == prim::Guard)
      {
        auto bailout_node = b->owningGraph()->create(prim::BailOut);
        auto node = *it;
        //std::cout << "graph =\n";
        //graph->dump();
        BailoutGraph bg(graph_);
        auto bailout_graph = bg.buildBailOutGraphFrom(node);
        bailout_graph->lint();
        subgraphs.insert({bailout_node, bailout_graph});
        //bailout_graph->dump();
        subgraphs.insert({bailout_node, bailout_graph});
        for (size_t i = 0; i < bg.live_inputs_.size(); i++)
        {
          bailout_node->addInput(bg.live_inputs_[i]);
          if (it->input() == bg.live_inputs_[i])
          {
            bailout_node->i_(attr::slot, i);
            bailout_node->output()->setType(it->output()->type());
          }
        }

        bailout_node->insertBefore(*it);
        //insert a bailout
        it->output()->replaceAllUsesWith(bailout_node->output());
        //it->output()->replaceAllUsesWith(it->input());
        it.destroyCurrent();
      }
      else
      {
        for (auto ib : it->blocks())
        {
          insertBailOuts(ib);
        }
      }
    }
  }

  std::shared_ptr<Graph> graph_;
  std::unordered_map<Node*,std::shared_ptr<Graph>> subgraphs;
};

void insertBailOuts(Block* b, std::shared_ptr<Graph> graph, std::shared_ptr<Graph> orig_graph)
{
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it)
  {
    if (it->kind() == prim::Guard)
    {
      auto bailout_node = b->owningGraph()->create(prim::BailOut);
      auto node = *it;
      //std::cout << "graph =\n";
      //graph->dump();
      BailoutGraph bg(orig_graph);
      auto bailout_graph = bg.buildBailOutGraphFrom(node);



      //bailout_graph->dump();
      bailout_graph->lint();
      bailout_node->g_(attr::Subgraph, bailout_graph);

      for (size_t i = 0; i < bg.live_inputs_.size(); i++)
      {
        bailout_node->addInput(bg.live_inputs_[i]);
        if (it->input() == bg.live_inputs_[i])
        {
          bailout_node->i_(attr::slot, i);
          bailout_node->output()->setType(it->output()->type());
        }
      }

      bailout_node->insertBefore(*it);
      //insert a bailout
      it->output()->replaceAllUsesWith(bailout_node->output());
      //it->output()->replaceAllUsesWith(it->input());
      it.destroyCurrent();
    }
    else
    {
      for (auto ib : it->blocks())
      {
        insertBailOuts(ib, graph, orig_graph);
      }
    }
  }
}

void insertBailOuts(std::shared_ptr<Graph> graph)
{
  InsertBailOuts ibo(graph);
  ibo.run();
  //insertBailOuts(graph->block(), graph, orig_graph);
}


void buildBailoutGraphForPrint(std::shared_ptr<Graph> graph) {
  //std::cout << "graph =\n";
  //graph->dump();
  BailoutGraph bg(graph);
  auto print = bg.locatePrint(graph->block());
  AT_ASSERT(print);
  //std::cout << "processing print = " << *print << std::endl;
  auto bailout_graph = bg.buildBailOutGraphFrom(print->next());
  bailout_graph->dump();
  bailout_graph->lint();
}

} // namespace jit
} // namespace torch
