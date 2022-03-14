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

#include "lbann/proto/factories.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/typename.hpp"

#include "lbann/layers/layer.hpp"
#include "lbann/layers/operator_layer.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/identity.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/image/bilinear_resize.hpp"
#include "lbann/layers/image/composite_image_transformation.hpp"
#include "lbann/layers/image/rotation.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/channelwise_fully_connected.hpp"
#include "lbann/layers/learning/channelwise_scale_bias.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/learning/deconvolution.hpp"
#include "lbann/layers/learning/embedding.hpp"
#include "lbann/layers/learning/entrywise_scale_bias.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/gru.hpp"
#include "lbann/layers/loss/categorical_accuracy.hpp"
#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/layers/loss/l1_norm.hpp"
#include "lbann/layers/loss/l2_norm2.hpp"
#include "lbann/layers/loss/mean_absolute_error.hpp"
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"
#include "lbann/layers/math/math_builders.hpp"
#include "lbann/layers/misc/argmax.hpp"
#include "lbann/layers/misc/argmin.hpp"
#include "lbann/layers/misc/channelwise_mean.hpp"
#include "lbann/layers/misc/channelwise_softmax.hpp"
#include "lbann/layers/misc/covariance.hpp"
#include "lbann/layers/misc/dist_embedding.hpp"
#include "lbann/layers/misc/dft_abs_builder.hpp"
#include "lbann/layers/misc/mini_batch_index.hpp"
#include "lbann/layers/misc/mini_batch_size.hpp"
#include "lbann/layers/misc/one_hot.hpp"
#include "lbann/layers/misc/rowwise_weights_norms.hpp"
#include "lbann/layers/misc/uniform_hash.hpp"
#include "lbann/layers/misc/variance.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/layers/regularizers/local_response_normalization.hpp"
#include "lbann/layers/regularizers/selu_dropout.hpp"
#include "lbann/layers/regularizers/entrywise_batch_normalization.hpp"
#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/layers/regularizers/instance_norm.hpp"
#include "lbann/layers/transform/batchwise_reduce_sum.hpp"
#include "lbann/layers/transform/bernoulli.hpp"
#include "lbann/layers/transform/categorical_random.hpp"
#include "lbann/layers/transform/concatenate.hpp"
#include "lbann/layers/transform/constant.hpp"
#include "lbann/layers/transform/crop.hpp"
#include "lbann/layers/transform/discrete_random.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/layers/transform/gather.hpp"
#include "lbann/layers/transform/gaussian.hpp"
#include "lbann/layers/transform/hadamard.hpp"
#include "lbann/layers/transform/in_top_k.hpp"
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/layers/transform/reduction.hpp"
#include "lbann/layers/transform/reshape.hpp"
#include "lbann/layers/transform/scatter.hpp"
#include "lbann/layers/transform/slice.hpp"
#include "lbann/layers/transform/sort.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/stop_gradient.hpp"
#include "lbann/layers/transform/sum.hpp"
#include "lbann/layers/transform/cross_grid_sum_slice.hpp"
#include "lbann/layers/transform/cross_grid_sum.hpp"
#include "lbann/layers/transform/tessellate.hpp"
#include "lbann/layers/transform/uniform.hpp"
#include "lbann/layers/transform/unpooling.hpp"
#include "lbann/layers/transform/weighted_sum.hpp"
#include "lbann/layers/transform/weights.hpp"

#include "lbann/data_coordinator/data_coordinator_metadata.hpp"
#include "lbann/utils/peek_map.hpp"

#include <layers.pb.h>

#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB

namespace lbann {
namespace proto {

namespace {

// Define the factory type.
using factory_type = lbann::generic_factory<
  lbann::Layer,
  std::string,
  generate_builder_type<lbann::Layer,
                        lbann_comm*,
                        const lbann_data::Layer&>,
  nullptr_key_error_policy>;

/** @brief Singleton holder for a factory.
 *
 *  @note This design requires that the builder function be valid for
 *  every combination of T, L, and D. That is, layer types for which a
 *  combination is invalid must handle that error inside their builder
 *  function.
 */
template <typename T, data_layout L, El::Device D>
class factory_manager
{
public:

  factory_manager() { register_default_builders(); }
  factory_type const& get() const noexcept { return factory_; }

private:

  // This macro simplifies the process of adding default builders
#define LBANN_REGISTER_BUILDER(KEY, LAYER_NAME)                         \
    factory_.register_builder(                                          \
      #KEY, build_##LAYER_NAME##_layer_from_pbuf<T,L,D>)
#define LBANN_REGISTER_DEFAULT_BUILDER(KEY, LAYER_NAME)                 \
    factory_.register_builder(                                          \
      #KEY,                                                             \
      [](lbann_comm*,                                                   \
         lbann_data::Layer const&){                                     \
        return std::make_unique<LAYER_NAME##_layer<T,L,D>>();         \
      })
#define LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(KEY, LAYER_NAME)       \
    factory_.register_builder(                                          \
      #KEY,                                                             \
      [](lbann_comm* comm,                                              \
         lbann_data::Layer const&){                                     \
        return std::make_unique<LAYER_NAME##_layer<T,L,D>>(comm);     \
      })

  // Builder registration happens here
  void register_default_builders() {

    // For now, we add a custom builder that will use the same
    // input/output type for the multi-precision-capable
    // OperatorLayer. This is temporary, until more of the factory
    // infrastructure considers multiple in/out types.
    factory_.register_builder(
      "OperatorLayer",
      [](lbann_comm* c, lbann_data::Layer const& params) {
        return build_operator_layer_from_pbuf<T, T, L, D>(c, params);
      });

    // Learning layers
    LBANN_REGISTER_BUILDER(Convolution, convolution);
    LBANN_REGISTER_BUILDER(ChannelwiseFullyConnected, channelwise_fully_connected);
    LBANN_REGISTER_BUILDER(ChannelwiseScaleBias, channelwise_scale_bias);
    LBANN_REGISTER_BUILDER(Deconvolution, deconvolution);
    LBANN_REGISTER_BUILDER(Embedding, embedding);
    LBANN_REGISTER_BUILDER(EntrywiseScaleBias, entrywise_scale_bias);
    LBANN_REGISTER_BUILDER(FullyConnected, fully_connected);
    LBANN_REGISTER_BUILDER(GRU, gru);

    // Math layers
    LBANN_REGISTER_BUILDER(MatMul, matmul);

    // Transform layers
    LBANN_REGISTER_BUILDER(BatchwiseReduceSum, batchwise_reduce_sum);
    LBANN_REGISTER_BUILDER(Bernoulli, bernoulli);
    LBANN_REGISTER_BUILDER(CategoricalRandom, categorical_random);
    LBANN_REGISTER_BUILDER(Concatenation, concatenate);
    LBANN_REGISTER_BUILDER(Constant, constant);
    LBANN_REGISTER_BUILDER(Crop, crop);
    LBANN_REGISTER_BUILDER(Cross_Grid_Sum_Slice, cross_grid_sum_slice);
    LBANN_REGISTER_BUILDER(Cross_Grid_Sum, cross_grid_sum);
    LBANN_REGISTER_BUILDER(Dummy, dummy);
    LBANN_REGISTER_BUILDER(Evaluation, evaluation);
    LBANN_REGISTER_BUILDER(Gather, gather);
    LBANN_REGISTER_BUILDER(Hadamard, hadamard);
    LBANN_REGISTER_BUILDER(Pooling, pooling);
    LBANN_REGISTER_BUILDER(Reduction, reduction);
    LBANN_REGISTER_BUILDER(Scatter, scatter);
    LBANN_REGISTER_BUILDER(Split, split);
    LBANN_REGISTER_BUILDER(StopGradient, stop_gradient);
    LBANN_REGISTER_BUILDER(Sum, sum);
    LBANN_REGISTER_BUILDER(WeightedSum, weighted_sum);
    LBANN_REGISTER_BUILDER(WeightsLayer, weights);

    // Activations
    LBANN_REGISTER_DEFAULT_BUILDER(Identity, identity);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(LogSoftmax, log_softmax);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(Relu, relu);
    LBANN_REGISTER_BUILDER(Softmax, softmax);

    // Loss Layers
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(CategoricalAccuracy, categorical_accuracy);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(L1Norm, l1_norm);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(L2Norm2, l2_norm2);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(MeanAbsoluteError, mean_absolute_error);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(MeanSquaredError, mean_squared_error);

    // Regularizer layers
    LBANN_REGISTER_BUILDER(Dropout, dropout);
    LBANN_REGISTER_BUILDER(InstanceNorm, instance_norm);
    LBANN_REGISTER_BUILDER(LocalResponseNormalization,
                           local_response_normalization);

    // Miscellaneous layers
    LBANN_REGISTER_BUILDER(ChannelwiseSoftmax, channelwise_softmax);
    LBANN_REGISTER_BUILDER(DFTAbs, dft_abs);
    LBANN_REGISTER_BUILDER(DistEmbedding, dist_embedding);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(MiniBatchIndex, mini_batch_index);
    LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM(MiniBatchSize, mini_batch_size);
    LBANN_REGISTER_BUILDER(OneHot, one_hot);
    LBANN_REGISTER_DEFAULT_BUILDER(RowwiseWeightsNorms, rowwise_weights_norms);
    LBANN_REGISTER_BUILDER(UniformHash, uniform_hash);

  }

  // Just to be clear/safe.
#undef LBANN_REGISTER_DEFAULT_BUILDER
#undef LBANN_REGISTER_DEFAULT_BUILDER_WITH_COMM

private:
  factory_type factory_;
}; // class factory_manager

template <typename T, data_layout L, El::Device D>
factory_type const& get_layer_factory() noexcept
{
  static factory_manager<T,L,D> factory_mgr_;
  return factory_mgr_.get();
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> construct_layer_legacy(
  lbann_comm* comm,
  int training_dr_linearized_data_size,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer) {
  std::stringstream err;

  // Input layers
  // Currently this cannot be suitably removed from this function
  // because it relies on LBANN_OPTION_NUM_PARALLEL_READERS and "data_readers"
  // arguments.
  if (proto_layer.has_input()) {
    const auto& params = proto_layer.input();
    const auto& data_field = params.data_field();
    if (Layout != data_layout::DATA_PARALLEL) {
      LBANN_ERROR("input layer is only supported with "
                  "a data-parallel layout");
    }
    /// @todo Question for Tim Moon and Tom Benson, I had to change this line from Layout to
    /// data_layout::DATA_PARALLEL to make it compile with clang on OS X, but it seems like
    /// this is not related to this PR.
    if ((typeid(TensorDataType) == typeid(DataType))
        && (Layout == data_layout::DATA_PARALLEL)) {
      return std::make_unique<
        input_layer<DataType, data_layout::DATA_PARALLEL, Device>>(comm,
                                                                   data_field);
    }
    else {
      LBANN_ERROR("Input layers are only valid with "
                    "TensorDataType == DataType and Layout == DATA_PARALLEL");
    }
  }

  // Transform layers
  // Currently this cannot be suitably removed from this function
  // because it relies on LBANN_OPTION_NUM_PARALLEL_READERS and "data_readers"
  // arguments.
  if (proto_layer.has_reshape()) {
    const auto& params = proto_layer.reshape();
    std::vector<int> dims = parse_list<int>(params.dims());
    if (params.num_dims() != 0) {
      LBANN_WARNING("found unused and deprecated prototext field (Reshape.num_dims)");
    }
    if (proto_layer.num_neurons_from_data_reader()) {
      dims.clear();
      if (training_dr_linearized_data_size == -1) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      dims.push_back(training_dr_linearized_data_size);
    }
    return std::make_unique<reshape_layer<TensorDataType, Layout, Device>>(comm, dims);
  }

  // Currently this cannot be suitably removed from this function
  // because it relies on LBANN_OPTION_NUM_PARALLEL_READERS and "data_readers"
  // arguments.
  if (proto_layer.has_slice()) {
    const auto& params = proto_layer.slice();
    std::vector<size_t> slice_points;

    auto layer = std::make_unique<slice_layer<TensorDataType, Layout, Device>>(comm);

    if (params.get_slice_points_from_reader() != "") {
      const slice_points_mode var = slice_points_mode_from_string(params.get_slice_points_from_reader());
      layer->setup_slice_points(params.axis(), true, var);
    } else {
      std::string slice_point_method_name = "'slice_points'";
      slice_points = parse_list<size_t>(params.slice_points());
      if (slice_points.size() < 2u) {
        err << "Failed to get slice points via " << slice_point_method_name << '.';
        LBANN_ERROR(err.str());
        return nullptr;
      }
      layer->setup_slice_points(params.axis(), slice_points);
    }
    return layer;
  }
  if (proto_layer.has_gaussian()) {
    const auto& params = proto_layer.gaussian();
    const auto& dims = parse_list<int>(params.neuron_dims());
    double mean = params.mean();
    double stdev = params.stdev();
    if (mean == 0.0 && stdev == 0.0) {
      mean = 0.0;
      stdev = 1.0;
    }
    return std::make_unique<gaussian_layer<TensorDataType,Layout,Device>>(
      comm,
      dims,
      mean,
      stdev,
      params.training_only());
  }
  if (proto_layer.has_uniform()) {
    const auto& params = proto_layer.uniform();
    const auto& dims = parse_list<int>(params.neuron_dims());
    double min = params.min();
    double max = params.max();
    if (min == 0.0 && max == 0.0) {
      min = 0.0;
      max = 1.0;
    }
    return std::make_unique<uniform_layer<TensorDataType,Layout,Device>>(
      comm,
      dims,
      min,
      max,
      params.training_only());
  }
  if (proto_layer.has_unpooling()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return std::make_unique<unpooling_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("unpooling layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_discrete_random()) {
    const auto& params = proto_layer.discrete_random();
    const auto& values = parse_list<DataType>(params.values());
    const auto& dims = parse_list<int>(params.dims());
    if (Layout == data_layout::DATA_PARALLEL
        && Device == El::Device::CPU) {
      return std::make_unique<discrete_random_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(
               comm, values, dims);
    } else {
      LBANN_ERROR("discrete random layer is only supported on CPU");
    }
  }
  if (proto_layer.has_in_top_k()) {
    const auto& params = proto_layer.in_top_k();
    return std::make_unique<in_top_k_layer<TensorDataType, Layout, Device>>(comm, params.k());
  }
  if (proto_layer.has_sort()) {
    const auto& params = proto_layer.sort();
    if (Layout == data_layout::DATA_PARALLEL) {
      return std::make_unique<sort_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, params.descending());
    } else {
      LBANN_ERROR("sort layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_tessellate()) {
    const auto& params = proto_layer.tessellate();
    const auto& dims = parse_list<int>(params.dims());
    return std::make_unique<tessellate_layer<TensorDataType, Layout, Device>>(comm, dims);
  }

  // Regularizer layers
  if (proto_layer.has_batch_normalization()) {
    const auto& params = proto_layer.batch_normalization();
    if (Layout == data_layout::DATA_PARALLEL) {
      int statistics_group_size = params.statistics_group_size();
      if (statistics_group_size < 0) {
        statistics_group_size = 0;  // Global statistics.
      } else if (statistics_group_size == 0) {
        statistics_group_size = 1;  // Default to local.
      }
      const auto& aggr_str = params.stats_aggregation();
      if (!aggr_str.empty()) {
        LBANN_WARNING("stats_aggregation field for BatchNormalization is deprecated");
        if (aggr_str == "local") {
          statistics_group_size = 1;
        } else if (aggr_str == "node_local") {
          statistics_group_size = comm->get_procs_per_node();
        } else if (aggr_str == "global") {
          statistics_group_size = 0;
        } else {
          err << "Invalid batch normalization stats aggregation " << aggr_str;
          LBANN_ERROR(err.str());
          return nullptr;
        }
      }
      // Set defaults if not given.
      auto decay = params.decay();
      if (decay == 0.0) {
        decay = 0.9;
      }
      auto epsilon = params.epsilon();
      if (epsilon == 0.0) {
        epsilon = 1e-5;
      }
      return std::make_unique<batch_normalization_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
        decay,
        epsilon,
        statistics_group_size);
    } else {
      LBANN_ERROR("batch normalization layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_selu_dropout()) {
    const auto& params = proto_layer.selu_dropout();
    const auto& keep_prob = params.keep_prob();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return std::make_unique<selu_dropout<TensorDataType, Layout, Device>>(keep_prob, alpha, scale);
    } else {
      return std::make_unique<selu_dropout<TensorDataType, Layout, Device>>(keep_prob);
    }
  }
  if (proto_layer.has_entrywise_batch_normalization()) {
    const auto& params = proto_layer.entrywise_batch_normalization();
    return std::make_unique<entrywise_batch_normalization_layer<TensorDataType, Layout, Device>>(params.decay(), params.epsilon());
  }
  if (proto_layer.has_layer_norm()) {
    const auto& params = proto_layer.layer_norm();
    const double epsilon = (params.has_epsilon()
                            ? params.epsilon().value()
                            : 1e-5);
    return std::make_unique<layer_norm_layer<TensorDataType, Layout, Device>>(epsilon);
  }

  // Activation layers
  if (proto_layer.has_elu()) {
    const auto& params = proto_layer.elu();
    const auto& alpha = params.alpha();
    if (alpha != 0) {
      return std::make_unique<elu_layer<TensorDataType, Layout, Device>>(comm, alpha);
    } else {
      return std::make_unique<elu_layer<TensorDataType, Layout, Device>>(comm);
    }
  }
  if (proto_layer.has_leaky_relu()) {
    const auto& params = proto_layer.leaky_relu();
    const auto& negative_slope = params.negative_slope();
    if (negative_slope != 0) {
      return std::make_unique<leaky_relu_layer<TensorDataType, Layout, Device>>(comm, negative_slope);
    } else {
      return std::make_unique<leaky_relu_layer<TensorDataType, Layout, Device>>(comm);
    }
  }

  // Loss layers
  if (proto_layer.has_cross_entropy()) {
    const auto& params = proto_layer.cross_entropy();
    return std::make_unique<cross_entropy_layer<TensorDataType, Layout, Device>>(comm, params.use_labels());
  }
  if (proto_layer.has_top_k_categorical_accuracy()) {
    const auto& params = proto_layer.top_k_categorical_accuracy();
    return std::make_unique<top_k_categorical_accuracy_layer<TensorDataType, Layout, Device>>(comm, params.k());
  }

  // Image layers
  if (proto_layer.has_bilinear_resize()) {
    const auto& params = proto_layer.bilinear_resize();
    if (Layout == data_layout::DATA_PARALLEL) {
      return std::make_unique<bilinear_resize_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, params.height(), params.width());
    } else {
      LBANN_ERROR("bilinear resize layer is only supported with "
                  "a data-parallel layout");
    }
  }

  if (proto_layer.has_composite_image_transformation()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return std::make_unique<composite_image_transformation_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("composite image transformation layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }

  if (proto_layer.has_rotation()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return std::make_unique<rotation_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("rotation layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }

  // Miscellaneous layers
  if (proto_layer.has_covariance()) {
    const auto& params = proto_layer.covariance();
    return std::make_unique<covariance_layer<TensorDataType, Layout, Device>>(comm, params.biased());
  }
  if (proto_layer.has_variance()) {
    const auto& params = proto_layer.variance();
    return std::make_unique<variance_layer<TensorDataType, Layout, Device>>(comm, params.biased());
  }
  if (proto_layer.has_channelwise_mean()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      return std::make_unique<channelwise_mean_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm);
    } else {
      LBANN_ERROR("channel-wise mean layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_argmax()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return std::make_unique<argmax_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("argmax layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_argmin()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return std::make_unique<argmin_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("argmin layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }

  // Throw exception if layer has not been constructed
  err << "could not construct layer " << proto_layer.name();
  LBANN_ERROR(err.str());
  return nullptr;

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> construct_layer(
  lbann_comm* comm,
  int training_dr_linearized_data_size,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer) {

  // Construct layer
  auto const& factory = get_layer_factory<TensorDataType, Layout, Device>();
  auto const& msg =
    helpers::get_oneof_message(proto_layer, "layer_type");
  std::unique_ptr<Layer> l = factory.create_object(
    msg.GetDescriptor()->name(), comm, proto_layer);
  if(!l) {
    if (typeid(TensorDataType) == typeid(DataType))
      l = construct_layer_legacy<DataType, Layout, Device>(
            comm, training_dr_linearized_data_size, num_parallel_readers, proto_layer);
    else
      LBANN_ERROR("Currently, layers of type \"", msg.GetDescriptor()->name(),
                  "\" are not constructible with any type other than the "
                  "default DataType.");
  }

  // Set additional parameters
  if (proto_layer.parallel_strategy().has_grid_tag()) {
    l->set_grid_tag(proto_layer.parallel_strategy().grid_tag().value());
  }

  return l;
}

// Template instantiation
#define PROTO_DEVICE(T, Device) \
  template std::unique_ptr<Layer> construct_layer<T, data_layout::DATA_PARALLEL, Device>(  \
    lbann_comm* comm,                                                                      \
    int training_dr_linearized_data_size,                                                  \
    int num_parallel_readers,                                                              \
    const lbann_data::Layer& proto_layer                                                   \
  );                                                                                       \
  template std::unique_ptr<Layer> construct_layer<T, data_layout::MODEL_PARALLEL, Device>( \
    lbann_comm* comm,                                                                      \
    int training_dr_linearized_data_size,                                                  \
    int num_parallel_readers,                                                              \
    const lbann_data::Layer& proto_layer                                                   \
  )

#include "lbann/macros/instantiate_device.hpp"

} // namespace proto
} // namespace lbann
