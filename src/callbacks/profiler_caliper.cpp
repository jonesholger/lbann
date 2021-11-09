////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/profiler_caliper.hpp"
//#include "lbann/utils/profiling.hpp"
#include "lbann/utils/serialize.hpp"

#include <callbacks.pb.h>

#include <algorithm>
#include <string>

#ifdef LBANN_HAS_CALIPER
namespace lbann {
namespace callback {


profiler_caliper::profiler_caliper(bool skip_init) :
    callback_base(), m_skip_init(skip_init) {
  if (!m_skip_init) {
    start();
  }
}

profiler_caliper::~profiler_caliper() {}

template <class Archive>
void profiler_caliper::serialize(Archive & ar) {
  ar(::cereal::make_nvp(
       "BaseCallback",
       ::cereal::base_class<callback_base>(this)),
     CEREAL_NVP(m_skip_init));
}

void profiler_caliper::on_epoch_begin(model *m) {
  const auto& c = static_cast<sgd_execution_context&>(m->get_execution_context());
  // Skip the first epoch
  if (m_skip_init && c.get_epoch() == 1) {
    start();
  }
  CALI_MARK_BEGIN("epoch");
}

void profiler_caliper::on_epoch_end(model *m) {
  CALI_MARK_END("epoch");
}

void profiler_caliper::on_validation_begin(model *m) {
  CALI_MARK_BEGIN("validation");
}

void profiler_caliper::on_validation_end(model *m) {
  CALI_MARK_END("validation");
}

void profiler_caliper::on_test_begin(model *m) {
  CALI_MARK_BEGIN("test");
}

void profiler_caliper::on_test_end(model *m) {
  CALI_MARK_END("test");
}

void profiler_caliper::on_batch_begin(model *m) {
  CALI_MARK_BEGIN("batch");
}

void profiler_caliper::on_batch_end(model *m) {
  CALI_MARK_END("batch");
}

void profiler_caliper::on_batch_evaluate_begin(model *m) {
  CALI_MARK_BEGIN("batch_evaluate");
}

void profiler_caliper::on_batch_evaluate_end(model *m) {
  CALI_MARK_END("batch_evaluate");
}

void profiler_caliper::on_forward_prop_begin(model *m) {
  CALI_MARK_BEGIN("forward_prop");
}

void profiler_caliper::on_forward_prop_end(model *m) {
  CALI_MARK_END("forward_prop");
}

void profiler_caliper::on_evaluate_forward_prop_begin(model *m) {
  CALI_MARK_BEGIN("evaluate_forward_prop");
}

void profiler_caliper::on_evaluate_forward_prop_end(model *m) {
  CALI_MARK_END("evaluate_forward_prop");
}

void profiler_caliper::on_backward_prop_begin(model *m) {
  CALI_MARK_BEGIN("backward_prop");
}

void profiler_caliper::on_backward_prop_end(model *m) {
  CALI_MARK_END("backward_prop");
}

void profiler_caliper::on_optimize_begin(model *m) {
  CALI_MARK_BEGIN("optimize");
}

void profiler_caliper::on_optimize_end(model *m) {
  CALI_MARK_END("optimize");
}

void profiler_caliper::on_forward_prop_begin(model *m, Layer *l) {
  std::string mark = "fw:" + l->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_forward_prop_end(model *m, Layer *l) {
  std::string mark = "fw:" + l->get_name();
  CALI_MARK_END(mark.c_str());
}

void profiler_caliper::on_evaluate_forward_prop_begin(model *m, Layer *l) {
  std::string mark = "eval_fw:" + l->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_evaluate_forward_prop_end(model *m, Layer *l) {
  std::string mark = "eval_fw:" + l->get_name();
  CALI_MARK_END(mark.c_str());
}

void profiler_caliper::on_backward_prop_begin(model *m, Layer *l) {
  std::string mark = "bw:" + l->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_backward_prop_end(model *m, Layer *l) {
  std::string mark = "bw:" + l->get_name();
  CALI_MARK_END(mark.c_str());
}

void profiler_caliper::on_optimize_begin(model *m, weights *w) {
  std::string mark = "opt:" + w->get_name();
  CALI_MARK_BEGIN(mark.c_str());
}

void profiler_caliper::on_optimize_end(model *m, weights *w) {
  std::string mark = "opt:" + w->get_name();
  CALI_MARK_END(mark.c_str());
}

std::unique_ptr<callback_base>
build_profiler_caliper_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, const std::shared_ptr<lbann_summary>&) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackProfilerCaliper&>(proto_msg);
  return make_unique<profiler_caliper>(params.skip_init());
}


} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::profiler_caliper
#include <lbann/macros/register_class_with_cereal.hpp>
#endif // LBANN_HAS_CALIPER

