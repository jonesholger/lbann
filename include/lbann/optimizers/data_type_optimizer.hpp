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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_HPP_INCLUDED
#define LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_HPP_INCLUDED

#include "lbann/optimizers/optimizer.hpp"

// Forward declarations
namespace cereal {
class access;
} // namespace cereal

namespace lbann {

// Forward declarations
template <typename TensorDataType>
class data_type_weights;

template <typename TensorDataType>
class data_type_optimizer
  : public Cloneable<HasAbstractFunction<data_type_optimizer<TensorDataType>>,
                     optimizer>
{

  using BaseType =
    Cloneable<HasAbstractFunction<data_type_optimizer<TensorDataType>>,
              optimizer>;

  friend class data_type_weights<TensorDataType>;

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

public:
  data_type_optimizer(TensorDataType learning_rate = 0);
  virtual ~data_type_optimizer() = default;

  /** @brief Human-readable description. */
  virtual description get_description() const override;

  /** @brief Must be called before training.
   *
   *  @param w Weights being optimized. If null, no change is made to
   *  the weights.
   */
  void setup(weights* w) override;
  virtual void setup(data_type_weights<TensorDataType>* w = nullptr);
  void setup_base(data_type_weights<TensorDataType>* w);
  /** @name Weights management */
  ///@{

  /** @brief Weights being optimized. */
  data_type_weights<TensorDataType>& get_weights();
  /** @brief Weights being optimized. */
  const data_type_weights<TensorDataType>& get_weights() const;
  /** @brief Weights being optimized. */
  void set_weights(data_type_weights<TensorDataType>* w) { m_weights = w; }

  ///@}
  /** @name Gradient update management */
  ///@{

  /** @brief Objective function gradient w.r.t. the weights.
   *
   *  An allreduce may be launched and/or synchronized if needed.
   */
  AbsDistMatrixType& get_gradient();

  /** @brief Optimization step. */
  void step() override;
  ///@}

  /** @brief Access the scaling factor for optimization step sizes. */
  double get_learning_rate() const final;
  /** @brief Set the scaling factor for optimization step sizes. */
  void set_learning_rate(double learning_rate) override;

  /** @name Checkpointing functionality */
  ///@{
  /** @brief Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

protected:
  data_type_optimizer(const data_type_optimizer& other);
  data_type_optimizer& operator=(const data_type_optimizer& other);

  /** @brief Computation for an optimization step.
   *
   *  @c values and @c gradient can be assumed to have the same
   *  distribution.
   */
  virtual void step_compute(AbsDistMatrixType& values,
                            const AbsDistMatrixType& gradient) = 0;

  /** @brief Get the info needed to construct a new gradient matrix.
   *  @return Tuple of height, width, and DistData.
   */
  std::tuple<El::Int, El::Int, El::DistData> get_matrix_info() const final;

private:
  /** @brief Weights being optimized. */
  data_type_weights<TensorDataType>* m_weights = nullptr;

  /** @brief Objective function gradient w.r.t. weights. */
  std::unique_ptr<AbsDistMatrixType> m_gradient;

  /** @brief Workspace matrix.
   *
   *  Helps ensure gradient contributions are in the right
   *  distribution. Most of the time, this should just be a matrix
   *  view.
   */
  std::unique_ptr<AbsDistMatrixType> m_gradient_v;

  /** @brief Communication request object for gradient allreduce.
   *
   *  Used to synchronize non-blocking allreduce.
   */
  Al::request m_gradient_allreduce_req;

  /** @brief Scaling factor for optimization step sizes.
   *
   *  This is not used by the base optimizer class, but is currently
   *  used by all derived optimizer classes. There are several cases
   *  where it is convenient to expose this in the base class,
   *  e.g. for variable learning rate schedules.
   *
   *  @todo Consider moving this to the derived classes.
   */
  double m_learning_rate;
};

#ifndef LBANN_DATA_TYPE_OPTIMIZER_INSTANTIATE
#define PROTO(T) extern template class data_type_optimizer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_DATA_TYPE_OPTIMIZER_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_HPP_INCLUDED
