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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/metrics/lbann_metric_categorical_accuracy.hpp"

using namespace std;
using namespace El;

lbann::categorical_accuracy::categorical_accuracy(lbann_comm* comm) 
  : metric(comm),
    YsColMax(comm->get_model_grid()),
    YsColMaxStar(comm->get_model_grid()) {}

lbann::categorical_accuracy::~categorical_accuracy() {
  YsColMax.Empty();
  YsColMaxStar.Empty();
  m_max_index.Empty();
  m_reduced_max_indicies.Empty();
}

void lbann::categorical_accuracy::setup(int num_neurons, int mini_batch_size) {
  metric::setup(num_neurons, mini_batch_size);
  // Clear the contents of the intermediate matrices
  Zeros(YsColMax, mini_batch_size, 1);
  Zeros(YsColMaxStar, mini_batch_size, 1);
  Zeros(m_max_index, mini_batch_size, 1); // Clear the entire matrix
  m_max_mini_batch_size = mini_batch_size;
}

void lbann::categorical_accuracy::fp_set_std_matrix_view(int64_t cur_mini_batch_size) {
  // Set the view based on the size of the current mini-batch
  //  View(m_cross_entropy_cost_v, m_cross_entropy_cost, IR(0, m_cross_entropy_cost.Height()), IR(0, cur_mini_batch_size));
}

double lbann::categorical_accuracy::compute_metric(ElMat& predictions_v, ElMat& groundtruth_v) {

  // Clear the contents of the intermediate matrices
  Zeros(YsColMax, m_max_mini_batch_size, 1);
  Zeros(YsColMaxStar, m_max_mini_batch_size, 1);

  /// Compute the error between the previous layers activations and the ground truth
  ColumnMax((DistMat)predictions_v, YsColMax); /// For each minibatch (column) find the maximimum value
  Copy(YsColMax, YsColMaxStar); /// Give every rank a copy so that they can find the max index locally

  Zeros(m_max_index, m_max_mini_batch_size, 1); // Clear the entire matrix

  /// Find which rank holds the index for the maxmimum value
  for(int mb_index = 0; mb_index < predictions_v.LocalWidth(); mb_index++) { /// For each sample in mini-batch that this rank has
    int mb_global_index = predictions_v.GlobalCol(mb_index);
    DataType sample_max = YsColMaxStar.GetLocal(mb_global_index, 0);
    for(int f_index = 0; f_index < predictions_v.LocalHeight(); f_index++) { /// For each feature
      if(predictions_v.GetLocal(f_index, mb_index) == sample_max) {
        m_max_index.Set(mb_global_index, 0, predictions_v.GlobalRow(f_index));
      }
    }
  }

  Zeros(m_reduced_max_indicies, m_max_mini_batch_size, 1); // Clear the entire matrix
  /// Merge all of the local index sets into a common buffer, if there are two potential maximum values, highest index wins
  comm->model_allreduce(m_max_index.Buffer(), m_max_index.Height() * m_max_index.Width(), m_reduced_max_indicies.Buffer(), mpi::MAX);

  /// Check to see if the predicted results match the target results
  int num_errors = 0;

  //  Copy(groundtruth_v, Y_local);

  /// @todo - BVE I believe that the following code works, but doesn't
  /// need to be this hard  it shouldn't have to check the inequality
  /// since there should only be one category turned on in the ground truth

  /// Distributed search over the groundtruth matrix
  /// Each rank will search its local portion of the matrix to find if it has the true category
  for(Int mb_index= 0; mb_index < groundtruth_v.LocalWidth(); mb_index++) { /// For each sample in mini-batch
    Int targetidx = -1;
    for(Int f_index= 0; f_index < groundtruth_v.LocalHeight(); f_index++) {
      if(groundtruth_v.GetLocal(f_index, mb_index) == (DataType) 1.) {
        targetidx = groundtruth_v.GlobalRow(f_index); /// If this rank holds the correct category, return the global row index
      }
    }
    if(targetidx != -1) { /// Only check against the prediction if this rank holds the groundtruth value
      Int global_mb_index = groundtruth_v.GlobalCol(mb_index);
      if(m_reduced_max_indicies.Get(global_mb_index, 0) != targetidx) {
        num_errors++;
      }
    }
  }
  
  num_errors = comm->model_allreduce(num_errors);
  
#if 0
  /// Allow the current root to compute the errors, since it has the data locally
  if(is_current_root()) {
    for (int mb_index= 0; mb_index < Y_local_v.Width(); mb_index++) { /// For each sample in mini-batch
      int targetidx = -1;
      float targetmax = 0;
      for (int f_index= 0; f_index < Y_local_v.Height(); f_index++) {
        if (targetmax < Y_local_v.Get(f_index, mb_index)) {
          targetmax = Y_local_v.Get(f_index, mb_index);
          targetidx = f_index;
        }
      }
      if(m_reduced_max_indicies.Get(mb_index, 0) != targetidx) {
        num_errors++;
      }
    }
  }
  num_errors = comm->model_broadcast(m_root, num_errors);
#endif

  return num_errors;
}
