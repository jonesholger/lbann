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

#ifndef LBANN_MODELS_MODEL_HPP_INCLUDED
#define LBANN_MODELS_MODEL_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/data_coordinator/data_coordinator_metadata.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/utils/graph.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/proto/factories.hpp"
#include "lbann/weights/weights.hpp"
#include "lbann/utils/threads/thread_pool.hpp"

// Note (trb): There's what is, IMO, an STL error in GCC in which the
// dtor for unique_ptr is checking sizeof(T), so this must be a
// complete type. Sigh. (The greater implication of this is that you
// cannot have `unique_ptr<IncompleteType>` as a drop-in for
// `IncompleteType*`, which is annoying.
#include <optimizers.pb.h>

#include <vector>
#include <string>
#include <unordered_map>

// Forward-declare protobuf class
namespace lbann_data {
class Model;
}

namespace lbann {

// Forward declarations
class lbann_callback;
class training_algorithm;
class callback_base;

/** @brief Abstract base class for neural network models. */
class model {
public:

  // ===========================================
  // Life cycle functions
  // ===========================================

  model(lbann_comm* comm,
        std::unique_ptr<objective_function> obj_fn,
        std::unique_ptr<lbann_data::Optimizer> default_optimizer_msg = nullptr);
  model(const model& other);
  model& operator=(const model& other);
  virtual ~model() = default;
  virtual std::unique_ptr<model> copy_model() const = 0;

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar);

  // ===========================================
  // Access functions
  // ===========================================

  /** @brief Model type's name.
   *  @details Should be a brief, human-readable description of the
   *  model's architecture.
   */
  virtual std::string get_type() const = 0;

  /** @brief Model instance name.
   *  @details Each model in a trainer should have a unique, and
   *  preferably human-readable, name.
   */
  std::string get_name() const noexcept { return m_name; }
  /** @brief Model instance name.
   *  @details Each model in a trainer should have a unique, and
   *  preferably human-readable, name.
   */
  void set_name(std::string name);

  void set_subgrid_communication_type(int type)
  {
    vector_communication_subgraph = type;
  }

  int get_subgrid_communication_type()
  {
    return vector_communication_subgraph;
  }

  void set_subgraph_num_parent_resources(int num_resources)
  {
    subgraph_num_resources_parent = num_resources;
  }

  int get_subgraph_num_parent_resources()
  {
    return subgraph_num_resources_parent;
  }

  void set_subgrid_topology(bool type)
  {
    enable_subgraph_topology = type;
  }

  bool get_subgrid_topology()
  {
    return enable_subgraph_topology;
  }

  void enable_subgraph_parallelism()
  {
    apply_subgraph_parallelism = true;
  }

  bool is_subgraph_parallelism_enabled()
  {
    return apply_subgraph_parallelism;
  }

  int get_num_resources_non_branch_layers()
  {
    return num_resources_non_branch_layers;
  }

  int get_num_resources_branch_layers()
  {
    return num_resources_branch_layers;
  }

  void set_num_resources_non_branch_layers(int num)
  {
    num_resources_non_branch_layers = num;
  }

  void set_num_resources_branch_layers(int num)
  {
    num_resources_branch_layers = num;
  }



  /** @brief Human-readable description. */
  virtual description get_description() const;

  /** @brief Mathematical function to be minimized during training. */
  observer_ptr<objective_function> get_objective_function() const {
    return m_objective_function.get();
  }

  /** @brief Return the model's metrics. */
  std::vector<metric*> get_metrics() const;

  /** @brief Size of model's list of layers. */
  El::Int get_num_layers() const noexcept;
  /** @param pos Position in model's list of layers. */
  Layer& get_layer(El::Int pos);
  /** @param pos Position in model's list of layers. */
  const Layer& get_layer(El::Int pos) const;
  /** @brief Return list of layers in model.
   *  @details The list is in execution order for forward propagation.
   */
  std::vector<Layer*> get_layers();
  /** @brief Return list of layers in model.
   *  @details The list is in execution order for forward propagation.
   */
  const std::vector<Layer*> get_layers() const;

  const std::vector<weights*> get_weights() const;
  std::vector<weights*> get_weights();
  std::vector<ViewingWeightsPtr> get_weights_pointers() const;

  /** @brief Get the list of callbacks for the model. */
  virtual std::vector<observer_ptr<callback_base>> get_callbacks() {
    std::vector<observer_ptr<callback_base>> callback_list;
    callback_list.reserve(m_callbacks.size());
    for (const auto& ptr : m_callbacks) {
      callback_list.push_back(ptr.get());
    }
    return callback_list;
  }

  virtual std::vector<std::shared_ptr<callback_base>>& get_callbacks_with_ownership() {
    return m_callbacks;
  }

  /** @brief Get the model's comm. */
  lbann_comm *get_comm() const {
    return m_comm;
  }

  /** Check to see if there is a valid training context for the model */
  bool has_valid_execution_context() const {
    return (m_execution_context != nullptr);
  }

  /** Grab the training context of the model */
  const execution_context& get_execution_context() const {
    if(m_execution_context == nullptr) {
      LBANN_ERROR("execution context is not set");
    }
    return *m_execution_context;
  }

  /** Grab the training context of the model */
  execution_context& get_execution_context() {
    return const_cast<execution_context&>(static_cast<const model&>(*this).get_execution_context());
  }

  // ===========================================
  // Model specification
  // ===========================================

  /** @brief Add layer to model. */
  virtual void add_layer(OwningLayerPtr&& l);

  /** @brief Add weights to model. */
  void add_weights(OwningWeightsPtr&& w);

  /** @brief Remove weights from model. */
  void remove_weights(std::string const& name);

  /** @brief Register a new callback for the model. */
  void add_callback(std::shared_ptr<callback_base> cb);

  /** @brief Register a new callback for the model. */
  //  void add_callbacks(std::vector<std::shared_ptr<callback_base>>& cb);

  /** @brief Register a new metric for the model. */
  void add_metric(std::unique_ptr<metric> m);

  /** @brief Copy trained weights from input parameter w.
   *
   *  Only weight values are placed, pointers and layer structure are in place.
   *  Weights to be copied are of the same name
   */
  void copy_trained_weights_from(std::vector<weights *>& w);

  /** @brief Construct an instance of the default optimizer.
   *
   *  If there is no default optimizer, a null pointer is returned.
   */
  template <typename TensorDataType>
  std::unique_ptr<optimizer> create_optimizer() const
  {
    if (m_default_optimizer_msg)
      return proto::construct_optimizer<TensorDataType>(
        *m_default_optimizer_msg);
    return nullptr;
  }

  /** @brief Set a flag that can be used to enable / disable the
   *         background I/O activities
   */
  void allow_background_io_activity(bool enable) { m_background_io_allowed = enable; }

  /** @brief Are background I/O activities enabled by the input layers */
  bool background_io_activity_allowed() { return m_background_io_allowed; }

  void swap_layers(model& other);
  void swap_weights(model& other);
  void swap_metrics(model& other);
  void swap_objective_function(model& other);

  // ===========================================
  // Model modification 
  // ===========================================

  /** @brief Insert layer in model. */
  void insert_layer(OwningLayerPtr&& l, std::string const& name);

  /** @brief Remove layer from model. */
  void remove_layer(std::string const& name);

  /** @brief Replace layer in model. */
  void replace_layer(OwningLayerPtr&& l, std::string const& name);

  // ===========================================
  // Setup
  // ===========================================

  /** @details Must be called after model specification and before
   *  execution. */
  virtual void setup(
    size_t max_mini_batch_size,
    DataReaderMetaData& dr_metadata,
    bool force=false);

  virtual void make_data_store_preloaded(execution_mode mode);

  virtual void mark_data_store_explicitly_loading(execution_mode mode);

  // ===========================================
  // Summarizer
  // ===========================================

  /**
   * Summarize statistics (e.g. timers, counters); these should be computable
   * quickly.
   */
  virtual void summarize_stats(lbann_summary& summarizer);
  /**
   * Summarize matrices (e.g. means); these are called less frequently and can
   * be more expensive.
   */
  virtual void summarize_matrices(lbann_summary& summarizer);

  // ===========================================
  // Checkpointing
  // ===========================================

  /** @brief Checkpoint model to given file descriptor, return number of bytes written */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** @brief Restore model by reading checkpoint from given file descriptor, return number of bytes read */
  virtual bool load_from_checkpoint_shared(persist& p);

  virtual bool save_to_checkpoint_distributed(persist& p);
  virtual bool load_from_checkpoint_distributed(persist& p);

  /** @brief Saves the model explicitly if the save_model callback is present */
  virtual bool save_model();

  /** @brief Write model to proto file */
  virtual void write_proto(lbann_data::Model* proto);

protected:

  /** @brief Reorder layer list with a gather.
   *
   *  The new layer list is the same length as @c gather_indices and
   *  its entries are given by
   *  @f[ \text{new\_list}[i] = \text{old\_list}[\text{gather\_indices}[i]] @f]
   *
   *  Since entries in the layer list must be unique, this will fail
   *  if @c gather_indices has any repeated entries.
   */
  void reorder_layers(const std::vector<El::Int>& gather_indices);

  /** @brief Remap pointers.
   *
   *  Layer and weights pointers are remapped using the provided
   *  maps. If a pointer is not a key in the corresponding map, the
   *  pointer is not changed.
   */
  virtual void remap_pointers(
    const std::unordered_map<Layer*,ViewingLayerPtr>& layer_map,
    const std::unordered_map<weights*,ViewingWeightsPtr>& weights_map);

  /** @brief
   *
   *  In case that a layer is frozen, also freeze layers that precede
   *  it if that makes senses for the particular model, such as
   *  sequential or siamese.  For othe models, users can manually
   *  control the behaivor by indicating whether to freeze each layer
   *  in the model description prototext.
   *
   *  For general DAG models, users need to manually specify each
   *  layer to freeze in the model description prototext.
   */
  virtual void freeze_layers_under_frozen_surface() {}

  /** @brief Set up topology of layer graph.
   *
   *  Called in setup function. All layers in connected component of
   *  layer graph are added to the model and all parent/child
   *  relationships between layers are reciprocated.
   */
  virtual void setup_layer_topology();

  /** setup sub grids for the sub graph parallelism	

  */	
  virtual void setup_subgrids();	

  virtual void get_subgrids_order(std::vector<int> &ranks_order, int num_branches);

  virtual int get_max_subgraph_branches();

  virtual void check_subgraph_parallelism();

  virtual void setup_subgrid_layers_run_condition();

  virtual void get_parent_subgrid_tags(int layer_index );

  virtual void get_subgraph_subgrids_ranks(std::vector<int> &parent_ranks, std::vector<int> &subgrid_ranks, int layer_index,int number_ranks_in_grid);

  virtual void get_resources_for_spliting_point(std::vector<int> &parent_ranks, 
                  std::vector<int> &subgrid_ranks, 
                  int layer_index,
                  int number_ranks_in_grid,
                  int num_subgrids);
  virtual void get_resources_for_merge_layers(std::set<int>& pooled_set,int child_index, int num_subgrids);

  virtual void get_resources_for_input_layer(std::vector<int>& masterSubGrid, int num_subgrids);

  virtual void setup_subcommunicators();
  
  /** @brief Set up layer execution order.
   *
   *  Called in setup function.
   */

  virtual void setup_layer_execution_order();
  /** @brief Set up layers.
   *
   *  Called in setup function.
   */
  virtual void setup_layers(size_t max_mini_batch_size, DataReaderMetaData& dr_metadata);
  /** @brief Set up weights.
   *
   *  Called in setup function. All weights being used by layers or
   *  the objective function are added to the model and all unused
   *  weights are deleted.
   */
  virtual void setup_weights();

public:
  // ===========================================
  // Execution
  // ===========================================

  /** @brief Reset model pointer and execution mode. */
  virtual void reset_mode(execution_context& context, execution_mode mode);
  /** @brief Reset model statistics for an epoch. */
  virtual void reset_epoch_statistics(execution_mode mode);

  /** @brief Forward propagation step. */
  virtual void forward_prop(execution_mode mode);
  /** @brief Backward propagation step. */
  virtual void backward_prop();
  /** Evaluate any metrics in the model */
  virtual void evaluate_metrics(execution_mode mode,
                                size_t current_mini_batch_size);
  /** @brief Clear each optimizer's gradient.
   *
   *  This must be called before training forward prop since layers
   *  set an optimizer flag during forward prop.
   */
  virtual void clear_gradients();
  /** @brief Update weights step. */
  virtual void update_weights();
  /** @brief Update layers step. */
  virtual bool update_layers();
  /** @brief Reconcile weight values.
   *
   *  If weight values are duplicated across multiple processes, they
   *  are set to the average across the processes.
   */
  virtual void reconcile_weight_values();

  // ===========================================
  // Callbacks
  // ===========================================

  /** @brief Execute callbacks at end of setup. */
  virtual void do_setup_end_cbs();
  /** @brief Execute callbacks at start of model forward propagation. */
  virtual void do_model_forward_prop_begin_cbs(execution_mode mode);
  /** @brief Execute callbacks at end of model forward propagation. */
  virtual void do_model_forward_prop_end_cbs(execution_mode mode);
  /** @brief Execute callbacks at start of layer forward propagation. */
  virtual void do_layer_forward_prop_begin_cbs(execution_mode mode, Layer *l);
  /** @brief Execute callbacks at end of layer forward propagation. */
  virtual void do_layer_forward_prop_end_cbs(execution_mode mode, Layer *l);
  /** @brief Execute callbacks at start of model backward propagation. */
  virtual void do_model_backward_prop_begin_cbs();
  /** @brief Execute callbacks at end of model backward propagation. */
  virtual void do_model_backward_prop_end_cbs();
  /** @brief Execute callbacks at start of layer backward propagation. */
  virtual void do_layer_backward_prop_begin_cbs(Layer *l);
  /** @brief Execute callbacks at end of layer backward propagation. */
  virtual void do_layer_backward_prop_end_cbs(Layer *l);
  /** @brief Execute callbacks at start of model optimization. */
  virtual void do_model_optimize_begin_cbs();
  /** @brief Execute callbacks at end of model optimization. */
  virtual void do_model_optimize_end_cbs();
  /** @brief Execute callbacks at the start of weight optimization. */
  virtual void do_weight_optimize_begin_cbs(weights *w);
  /** @brief Execute callbacks at the end of weight optimization. */
  virtual void do_weight_optimize_end_cbs(weights *w);


#ifdef LBANN_HAS_DISTCONV
  /* @brief Return the maximum mini-batch size used by Distconv. */
  size_t get_max_mini_batch_size_distconv() const { return m_max_mini_batch_size_distconv; }
#endif

	
private:	
  // map to store all distinct grids in the model
  std::unordered_map<std::string, std::shared_ptr<El::Grid>> grids; 

  std::unordered_map<std::string, std::shared_ptr<El::mpi::Comm>> subCommunicatorsSubgrids; 
  // map to store all distinct mpi groups in the model (one to one mapping with grids)
  std::unordered_map<std::string, std::unique_ptr<El::mpi::Group>> grids_mpi_groups; 



private:

  /** Pointer to the execution context object used for training or evaluating this model */
  observer_ptr<execution_context> m_execution_context;

  /** @brief LBANN communicator. */
  lbann_comm* m_comm;

  /*experimental code for Sub graph*/
  /** Enable vector communication for the subgraph parallelism */
  //0: send-recv based subgrid communication
  //1: collective based subgrid communication without optimization that requires specific assumptions like subgrids should have same size and creates sub-communicators everytime 
  //2: collective based subgrid communication with optimization

  int vector_communication_subgraph = 0;

  //Number of resources for parent (common) grid
  //0: use all resources (default) 
  int subgraph_num_resources_parent = 0;

  //0: no topology aware design
  //1: master grid in round robin manner of nodes (GPUs per node 4)  1 3 5 7, 2 4 6 8     
  bool enable_subgraph_topology = false;

  // whether subgraph parallelism is enabled or not for the model 
  bool apply_subgraph_parallelism = false;

  // total number of resources / ranks for branch (subgrid) layers
  int num_resources_branch_layers;

  // total number of resources / ranks for common/seq layers
  int num_resources_non_branch_layers;
  
  /** @brief Model instance's name.
   *  @details Each model in a trainer should have a unique,
   *  preferably human-readable, name.
   */
  std::string m_name;

  /** @brief Tensor operations.
   *  @details The list is in execution order for forward propagation.
   */
  std::vector<OwningLayerPtr> m_layers;

  /** @brief Trainable parameters. */
  std::vector<OwningWeightsPtr> m_weights;

  /** @details If a layer needs to construct an optimizer during
   *  setup, it will make a copy of the default optimizer. This object
   *  is just used to create copies and is not actually used for
   *  optimization.
   */
  std::unique_ptr<lbann_data::Optimizer> m_default_optimizer_msg;

  /** @brief Mathematical function to be minimized during training. */
  std::unique_ptr<objective_function> m_objective_function;

  /** @brief Numerical quantities to evaluate model performance.
   *  @details Does not affect training.
   */
  std::vector<std::unique_ptr<metric>> m_metrics;

  /** @brief Current callbacks to process. */
  std::vector<std::shared_ptr<callback_base>> m_callbacks;

  /** @brief Flag that allows input layers to fetch data in the background */
  bool m_background_io_allowed = true;

  /** @brief Is the model setup
   *  @details Flag to indicate if the setup function has been called
   */
  bool m_model_is_setup = false;

  // ===========================================
  // Functions to add utility layers
  // ===========================================

  /** @brief Insert evaluation layers where needed.
   *
   *  If a @c lbann::layer_term or @c lbann::layer_metric corresponds
   *  to a layer that is not an evaluation_layer, an evaluation layer
   *  is created and added to the model.
   *
   *  @param layer_set      Layers in model. Updated with any newly
   *                        created layers.
   *  @param layer_names    Names of layers in model. Updated with any
   *                        newly created layers.
   */
  void add_evaluation_layers(std::unordered_set<Layer*>& layer_set,
                             std::unordered_set<std::string>& layer_names);

  /** @brief Insert dummy layers after layers with too few children.
   *
   *  If a layer expects more child layers than it has, add dummy
   *  layers until it has enough children.
   *
   *  @param layer_names    Names of layers in model. Updated with any
   *                        newly created layers.
   */
  void add_dummy_layers(std::unordered_set<std::string>& layer_names);
  /** @brief Insert split layers after layers with too many children.
   *
   *  If a layer expects one child layer but has multiple, add a split
   *  layer to the model.
   *
   *  @param layer_names    Names of layers in model. Updated with any
   *                        newly created layers.
   */
  void add_split_layers(std::unordered_set<std::string>& layer_names);

#ifdef LBANN_HAS_DISTCONV
  void setup_distconv();
  void setup_distributions();
  void print_distributions() const;

  /** @brief The maximum mini-batch size used by Distconv.
   *  @details This should be set before setup_distconv() is called.
   */
  size_t m_max_mini_batch_size_distconv;

#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann

#endif // LBANN_MODELS_MODEL_HPP_INCLUDED
