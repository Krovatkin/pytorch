#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include <c10/macros/Macros.h>

#include "test/cpp/tensorexpr/padded_buffer.h"
#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"

#include "torch/csrc/jit/passes/utils/subgraph_utils.h"
#include "torch/csrc/jit/runtime/graph_executor.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/codegen/fuser/interface.h"
#include "torch/csrc/jit/passes/tensorexpr_fuser.h"
#include "test/cpp/jit/test_utils.h"
#include <chrono>
namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

static void wrapGraphWithIntWrapper(std::shared_ptr<Graph> graph) {

  auto block = graph->block();
  auto first_node = *block->nodes().begin();
  auto iw_node = SubgraphUtils::createSingletonSubgraph(first_node, prim::IntWrapper);
  //first_node->replaceAllUsesWith(iw_node);
  for (auto it = iw_node->next()->iterator(); it != block->nodes().end();) {
    auto to_merge = *it;
    it++;
    SubgraphUtils::mergeNodeIntoSubgraph(to_merge, iw_node);
  }
}

static void wrapGraphWithIntWrapper2(std::shared_ptr<Graph> graph) {

  // Merge everything into a single subgraph
  bool first = true;
  Node* subgraph;
  for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend();) {
    if (first) {
      subgraph = SubgraphUtils::createSingletonSubgraph(
          *it, prim::IntWrapper);
      it = ++subgraph->reverseIterator();
      first = false;
    }

    SubgraphUtils::mergeNodeIntoSubgraph(*it, subgraph);
    it = ++subgraph->reverseIterator();
  }
}

TEST(FuserTest, TestSimpleIntWrapper) {
  getProfilingMode() = true;
  setTensorExprFuserEnabled(true);
  setTexprReductionsEnabled(true);
  torch::jit::overrideCanFuseOnCPU(true);

  // const auto graph_string = R"IR(
  //     graph(%0 : Tensor,
  //           %1 : Tensor):
  //       %2 : Tensor = aten::mul(%0, %1)
  //       %3 : Tensor = aten::mul(%2, %1)
  //       return (%3))IR";

  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor):
      %5 : Tensor = aten::mm(%0, %3)
      %6 : Tensor = aten::mm(%1, %4)
      %7 : int = prim::Constant[value=1]()
      %8 : Tensor = aten::add(%5, %6, %7)
      %9 : Tensor, %10 : Tensor, %11 : Tensor, %12 : Tensor = prim::ConstantChunk[chunks=4, dim=1](%8)
      %13 : Tensor = aten::sigmoid(%9)
      %14 : Tensor = aten::sigmoid(%12)
      %15 : Tensor = aten::tanh(%11)
      %16 : Tensor = aten::sigmoid(%10)
      %17 : Tensor = aten::mul(%16, %2)
      %18 : Tensor = aten::mul(%13, %15)
      %19 : int = prim::Constant[value=1]()
      %20 : Tensor = aten::add(%17, %18, %19)
      %21 : Tensor = aten::tanh(%20)
      %22 : Tensor = aten::mul(%14, %21)
      return (%22, %20))IR";

  auto graph = std::make_shared<Graph>();        
  torch::jit::parseIR(graph_string, &*graph);

#define ENV_PARAM(NAME, DEFAULT_NUM) \
  auto static const c_ ## NAME = std::getenv(#NAME); \
  static const size_t NAME = c_ ## NAME ? std::atoi(c_ ## NAME) : DEFAULT_NUM; \
  std::cout << #NAME << ": " << NAME << std::endl;

  ENV_PARAM(batch_size, 1);
  ENV_PARAM(input_size, 256);
  ENV_PARAM(times, 1000);
  
  int hidden_size = 2 * input_size;

  auto input = at::randn({batch_size, input_size}, at::kCUDA);
  auto hx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto cx = at::randn({batch_size, hidden_size}, at::kCUDA);
  auto w_ih = at::randn({4 * hidden_size, input_size}, at::kCUDA).t();
  auto w_hh = at::randn({4 * hidden_size, hidden_size}, at::kCUDA).t();

  GraphExecutor ge(graph, "ge");
  {
    auto stack = Stack({input, hx, cx, w_ih, w_hh});
    ge.run(stack);
  }

  {
    auto stack = Stack({input, hx, cx, w_ih, w_hh});
    ge.run(stack);
  }

  //time baseline


  auto bench = [&](const std::string& name) {

    
    double cum = 0;
    for (auto i = 0; i < times; i++) {
      auto stack = Stack({input, hx, cx, w_ih, w_hh});
      auto begin = std::chrono::high_resolution_clock::now();
      ge.run(stack);
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
      cum += elapsed.count();
    }

    cum = cum * 1e-6 / times;
    std::cout << name << ": " << cum << " us\n";
    return cum;
  };
  auto baseline = bench("baseline");
  {
    auto stack = Stack({input, hx, cx, w_ih, w_hh});
    ExecutionPlan& plan = const_cast<ExecutionPlan&>(ge.getPlanFor(stack, 0));
    wrapGraphWithIntWrapper2(plan.graph);
    plan.code = Code(plan.graph, "int wrapper");
    ge.run(stack);
    GRAPH_DUMP("INT WRAPPED: ", plan.graph);
  }
  auto int_wrap = bench("int_wrapped");
  std::cout << "diff: " << ((int_wrap / baseline - 1) * 100) << std::endl;
}

TEST(ATen, _cast_Float) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Cast::make(kFloat, load_a);
  Stmt* store_b = b_buf.store({index}, to_float);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), static_cast<float>(i));
  }
}

TEST(ATen, negInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Sub::make(0, load_a);
  Stmt* store_b = b_buf.store({index}, to_float);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), -static_cast<float>(i));
  }
}

TEST(ATen, negFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Sub::make(0, load_a);
  Stmt* store_b = b_buf.store({index}, to_float);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), -i);
  }
}

TEST(ATen, addInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a + load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}

TEST(ATen, addFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a + load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}

TEST(ATen, subInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a - load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}

TEST(ATen, subFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a - load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}

TEST(ATen, lerp) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a + load_c * (load_b - load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + c_v(i) * (b_v(i) - a_v(i)));
  }
}

TEST(ATen, addcmulInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kInt));
  Placeholder e_buf(BufHandle("E", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  ExprHandle load_d = d_buf.load(index);
  Stmt* store_e = e_buf.store({index}, load_a + load_b * load_c * load_d);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);
  PaddedBuffer<int> e_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
    d_v(i) = 5 * i + 3;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf, e_buf);
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), 5 * i + 3);
    ASSERT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i));
  }
}

TEST(ATen, addcmulFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder e_buf(BufHandle("E", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  ExprHandle load_d = d_buf.load(index);
  Stmt* store_e = e_buf.store({index}, load_a + load_b * load_c * load_d);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);
  PaddedBuffer<float> e_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
    d_v(i) = 5 * i + 3;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf, e_buf);
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), 5 * i + 3);
    ASSERT_FLOAT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i));
  }
}

TEST(ATen, mulInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a * load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));
  }
}

TEST(ATen, mulFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a * load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));
  }
}

TEST(ATen, divInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a / load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

TEST(ATen, divFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a / load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

TEST(ATen, maxInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::max(a_v(i), b_v(i)));
  }
}

TEST(ATen, maxFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::fmax(a_v(i), b_v(i)));
  }
}

TEST(ATen, minInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::min(a_v(i), b_v(i)));
  }
}

TEST(ATen, minFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::fmin(a_v(i), b_v(i)));
  }
}

void __ubsan_ignore_float_divide_by_zero__ testATenreciprocal() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, FloatImm::make(1.0f) / load_a);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 1.0f / i);
  }
}

TEST(ATen, reluInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, Max::make(load_a, 0, false));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::max(a_v(i), 0));
  }
}

TEST(ATen, reluFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store(
      {index}, Max::make(load_a, 0, false) // relu does not propagate nans
  );
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::fmax(a_v(i), 0));
  }
}

TEST(ATen, logFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, log(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log(a_v(i)));
  }
}

TEST(ATen, log10Float) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, log10(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log10(a_v(i)));
  }
}

TEST(ATen, log2Float) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, log2(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log2(a_v(i)));
  }
}

TEST(ATen, expFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, exp(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::exp(a_v(i)));
  }
}

TEST(ATen, erfFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, erf(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::erf(a_v(i)));
  }
}

TEST(ATen, cosFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, cos(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::cos(a_v(i)));
  }
}

TEST(ATen, eqInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, geInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGE)));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, gtInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 6);
  std::vector<int> b_buffer(N, 3);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGT)));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, leInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLE)));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, ltInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLT)));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 0);
}

} // namespace jit
} // namespace torch
