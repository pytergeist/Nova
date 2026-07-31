#include <gtest/gtest.h>

#include "Fusion/autodiff/AutodiffContext.hpp"
#include "Fusion/autodiff/Engine.hpp"

#include "fixtures.h"

TEST(AutodiffContextTest, set_and_get_returns_engine_instance) {
   AutodiffContextReset reset;
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   AutodiffContext<float>::set(&engine);
   EXPECT_EQ(&AutodiffContext<float>::get(), &engine);
}

TEST(AutodiffContextTest, clear_results_in_empty_context) {
   AutodiffContextReset reset;
   Engine<float> engine1(AutodiffContext<float>::runtime().grad_store());
   Engine<float> engine2(AutodiffContext<float>::runtime().grad_store());
   AutodiffContext<float>::set(&engine1);
   AutodiffContext<float>::set(&engine2);
   AutodiffContext<float>::clear();
   EXPECT_FALSE(AutodiffContext<float>::has());
}

TEST(AutodiffContextTest,
     sets_multiple_and_get_returns_latest_engine_instance) {
   AutodiffContextReset reset;
   Engine<float> engine1(AutodiffContext<float>::runtime().grad_store());
   Engine<float> engine2(AutodiffContext<float>::runtime().grad_store());
   AutodiffContext<float>::set(&engine1);
   AutodiffContext<float>::set(&engine2);
   EXPECT_EQ(&AutodiffContext<float>::get(), &engine2);
}

TEST(AutodiffContextTest, returns_correct_engine_after_multiple_set_and_pop) {
   AutodiffContextReset reset;
   Engine<float> engine1(AutodiffContext<float>::runtime().grad_store());
   Engine<float> engine2(AutodiffContext<float>::runtime().grad_store());
   AutodiffContext<float>::set(&engine1);
   AutodiffContext<float>::set(&engine2);
   AutodiffContext<float>::pop_noexcept();
   EXPECT_EQ(&AutodiffContext<float>::get(), &engine1);
}

TEST(AutodiffContextTest, empty_throws_runtime_error) {
   AutodiffContextReset reset;
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   EXPECT_THROW(AutodiffContext<float>::get(), std::runtime_error);
}

TEST(AutodiffContextTest, has_returns_true_when_set) {
   AutodiffContextReset reset;
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   AutodiffContext<float>::set(&engine);
   EXPECT_TRUE(AutodiffContext<float>::has());
}

TEST(AutodiffContextTest, has_returns_false_when_not_set) {
   AutodiffContextReset reset;
   Engine<float> engine(AutodiffContext<float>::runtime().grad_store());
   EXPECT_FALSE(AutodiffContext<float>::has());
}

TEST(EngineScopeTest, engine_scope_enter_sets_engine) {
   AutodiffContextReset reset;
   EngineScope<float> scope{};
   scope.enter();

   EXPECT_TRUE(scope.active());
   EXPECT_TRUE(AutodiffContext<float>::has());
   EXPECT_EQ(&AutodiffContext<float>::get(), &scope.eng());
}

TEST(EngineScopeTest, engine_scope_exit_after_enter_clears_engine_state) {
   AutodiffContextReset reset;

   EngineScope<float> scope{};
   scope.enter();

   ASSERT_TRUE(scope.active());
   ASSERT_TRUE(AutodiffContext<float>::has());

   scope.exit();

   EXPECT_FALSE(scope.active());
   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_THROW(AutodiffContext<float>::get(), std::runtime_error);
}

TEST(EngineScopeTest, engine_scope_destructor_clears_engine_state_after_enter) {
   AutodiffContextReset reset;

   {
      EngineScope<float> scope{};
      scope.enter();

      EXPECT_TRUE(scope.active());
      EXPECT_TRUE(AutodiffContext<float>::has());
   }

   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_THROW(AutodiffContext<float>::get(), std::runtime_error);
}

TEST(EngineScopeTest,
     multiple_engine_scope_exit_after_enter_clears_engine_state) {
   AutodiffContextReset reset;

   EngineScope<float> scope1{};
   scope1.enter();
   EngineScope<float> scope2{};
   scope2.enter();

   EXPECT_TRUE(scope1.active());
   EXPECT_TRUE(AutodiffContext<float>::has());

   Engine<float> &engine1 = AutodiffContext<float>::get();

   scope1.exit();

   Engine<float> &engine2 = AutodiffContext<float>::get();

   EXPECT_TRUE(scope2.active());
   EXPECT_NE(&engine1, &engine2);
   scope2.exit();
   EXPECT_FALSE(AutodiffContext<float>::has());
   EXPECT_THROW(AutodiffContext<float>::get(), std::runtime_error);
}

// to test:
// test new exit?