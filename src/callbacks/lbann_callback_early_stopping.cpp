////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC. 
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
// lbann_early_stopping .hpp .cpp - Callback hooks for early stopping
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/lbann_callback_early_stopping.hpp"

namespace lbann {

lbann_callback_early_stopping::lbann_callback_early_stopping(int64_t patience) :
  patience(patience), last_score(0.0f), wait(0) {}

void lbann_callback_early_stopping::on_validation_end(model* m) {
  for (auto&& metric : m->metrics) {
    if(metric->supports_early_termination()) {
      double score = metric->report_metric(execution_mode::validation);
      if ((metric->higher_score_is_better() && score > last_score) || (metric->lower_score_is_better() && score < last_score)) {
        if (m->get_comm()->am_model_master()) {
          std::cout << "Model " << m->get_comm()->get_model_rank() <<
            " score is improving " << last_score << " >> " << score << std::endl;
        }
        last_score = score;
        wait = 0;
      } else {
        if (wait >= patience) {
          m->set_terminate_training(true);
          if (m->get_comm()->am_model_master()) {
            std::cout << "Model " << m->get_comm()->get_model_rank() <<
              " terminating training due to early stopping" << std::endl;
          }
        } else {
          ++wait;
        }
      }
      break; /// Only use the first available metric for early termination
    }
  }
}

}  // namespace lbann
