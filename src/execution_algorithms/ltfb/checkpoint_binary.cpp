////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////
#include "lbann/execution_algorithms/ltfb/random_pairwise_exchange.hpp"

#include "lbann/comm_impl.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/serialization/rooted_archive_adaptor.hpp"
#include "lbann/weights/data_type_weights.hpp"

#include "checkpoint_common.hpp"

namespace lbann {

namespace ltfb {

CheckpointBinary::CheckpointBinary(std::set<std::string> const& weights_names)
  : BaseType(weights_names)
{}

CheckpointBinary::CheckpointBinary(std::set<std::string>&& weights_names)
  : BaseType(std::move(weights_names))
{}

std::unique_ptr<model>
CheckpointBinary::get_partner_model(model const& m,
                                    El::Int partner_trainer,
                                    size_t /*step*/)
{
  auto const& comm = *m.get_comm();

  // Start by copying this model, then do the exchange.
  auto partner_model_ptr = m.copy_model();
  auto& partner_model = *partner_model_ptr;

  // Keep track of weights that shouldn't be exchanged
  std::unordered_map<std::string, std::unique_ptr<weights>> restore_weights;
  auto const& weights_names = this->weights_names();
  if (!weights_names.empty()) {
    for (auto w : partner_model.get_weights()) {
      if (weights_names.find(w->get_name()) == weights_names.cend()) {
        using TensorDataType = DataType;
        using WeightsType = data_type_weights<TensorDataType>;
        restore_weights[w->get_name()] =
          std::make_unique<WeightsType>(dynamic_cast<WeightsType&>(*w));
      }
    }
  }
  exchange(comm, partner_model, partner_trainer);
  restore_model_weights(partner_model, restore_weights);

  return partner_model_ptr;
}

} // namespace ltfb
} // namespace lbann
