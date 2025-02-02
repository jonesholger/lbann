////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////

#include "lbann_config.hpp"


#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "lbann/lbann.hpp"
#include <time.h>
#include <cfloat>

using namespace lbann;

//==========================================================================
int main(int argc, char *argv[]) {
  world_comm_ptr comm = initialize(argc, argv);
  bool master = comm->am_world_master();
  const int rank = comm->get_rank_in_world();
  const int np = comm->get_procs_in_world();

  try {
    auto& arg_parser = global_argument_parser();
    construct_std_options();
    construct_jag_options();
    try {
      arg_parser.parse(argc, argv);
    }
    catch (std::exception const& e) {
      auto guessed_rank = guess_global_rank();
      if (guessed_rank <= 0)
        // Cannot call `El::ReportException` because MPI hasn't been
        // initialized yet.
        std::cerr << "Error during argument parsing:\n\ne.what():\n\n  "
                  << e.what() << "\n\nProcess terminating." << std::endl;
      std::terminate();
    }

    if (arg_parser.get<std::string>(FILELIST) == "" ||
        arg_parser.get<std::string>(OUTPUT_DIR) == "") {
      if (master) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: usage: " + argv[0] + " --filelist=<string> --output_dir=<string>");
      }
    }

    const std::string dir = arg_parser.get<std::string>(OUTPUT_DIR);

    if (master) {
      std::stringstream s;
      s << "mkdir -p " << arg_parser.get<std::string>(OUTPUT_DIR);
      int r = system(s.str().c_str());
      if (r != 0) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " :: system call failed: " + s.str());
      }
    }

    std::vector<std::string> files;
    std::string f;
    int size;
    if (master) {
      std::stringstream s;
      std::ifstream in(arg_parser.get<std::string>(FILELIST).c_str());
      if (!in) {
        throw lbann_exception(std::string{} + __FILE__ + " " +
                              std::to_string(__LINE__) + " :: failed to open " +
                              arg_parser.get<std::string>(FILELIST) +
                              " for reading");
      }
      std::string line;
      while (getline(in, line)) {
        if (line.size()) {
          s << line << " ";
        }
      }
      in.close();
      f = s.str();
      size = s.str().size();
      std::cout << "size: " << size << "\n";
    }
    comm->world_broadcast<int>(0, &size, 1);
    f.resize(size);
    comm->world_broadcast<char>(0, &f[0], size);

    std::stringstream s2(f);
    std::string filename;
    while (s2 >> filename) {
      if (filename.size()) {
        files.push_back(filename);
      }
    }
    if (rank==1) std::cerr << "num files: " << files.size() << "\n";

    //=======================================================================

    hid_t hdf5_file_hnd{};
    std::string key;
    conduit::Node n_ok;
    conduit::Node tmp;

    //    if (master) std::cout << np << hdf5_file_hnd << "\n";

    int num_samples = 0;

    std::vector<float> v_max(12, FLT_MIN);
    std::vector<float> v_min(12, FLT_MAX);

    size_t h = 0;
    for (size_t j=rank; j<files.size(); j+= np) {
      h += 1;
      if (h % 10 == 0) std::cout << rank << " :: processed " << h << " files\n";

      try {
std::cerr << rank << " :: opening for reading: " << files[j] << "\n";
        hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( files[j].c_str() );
      } catch (...) {
        std::cerr << rank << " :: exception hdf5_open_file_for_read: " << files[j] << "\n";
        continue;
      }

      std::vector<std::string> cnames;
      try {
        conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", cnames);
      } catch (...) {
        std::cerr << rank << " :: exception hdf5_group_list_child_names; " << files[j] << "\n";
        continue;
      }

      for (size_t i=0; i<cnames.size(); i++) {

        key = "/" + cnames[i] + "/performance/success";
        try {
          conduit::relay::io::hdf5_read(hdf5_file_hnd, key, n_ok);
        } catch (...) {
          std::cerr << rank << " :: exception reading success flag: " << files[j] << "\n";
          continue;
        }

        int success = n_ok.to_int64();
        if (success == 1) {


            try {
              key = cnames[i] + "/outputs/images/(0.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float32_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              //out.write((char*)&emi[0], sizeof(float)*image_size);
              for (int channel = 0; channel<4; channel++) {
                for (size_t hh=channel; hh<image_size; hh += 4) {
                  float val = emi[hh];
                  if (val < v_min[channel]) v_min[channel] = val;
                  if (val > v_max[channel]) v_max[channel] = val;
                }
              }
            } catch (...) {
              std::cerr << rank << " :: " << "exception reading image: (0.0, 0.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

            try {
              key = cnames[i] + "/outputs/images/(90.0, 0.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float32_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              //out.write((char*)&emi[0], sizeof(float)*image_size);
              for (int channel = 0; channel<4; channel++) {
                for (size_t hh=channel; hh<image_size; hh += 4) {
                  float val = emi[hh];
                  if (val < v_min[channel+4]) v_min[channel+4] = val;
                  if (val > v_max[channel+4]) v_max[channel+4] = val;
                }
              }
            } catch (...) {
              std::cerr << rank << " :: " << "exception reading image: (0.0, 0.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

            try {
              key = cnames[i] + "/outputs/images/(90.0, 78.0)//0.0/emi";
              conduit::relay::io::hdf5_read(hdf5_file_hnd, key, tmp);
              conduit::float32_array emi = tmp.value();
              const size_t image_size = emi.number_of_elements();
              //out.write((char*)&emi[0], sizeof(float)*image_size);
              for (int channel = 0; channel<4; channel++) {
                for (size_t hh=channel; hh<image_size; hh += 4) {
                  float val = emi[hh];
                  if (val < v_min[channel+8]) v_min[channel+8] = val;
                  if (val > v_max[channel+8]) v_max[channel+8] = val;
                }
              }
            } catch (...) {
              std::cerr << rank << " :: " << "exception reading image: (0.0, 0.0) for sample: " << cnames[i] << " which is " << i << " of " << cnames[i] << "; "<< files[j] << "\n";
              continue;
            }

          ++num_samples;
        }
      }
    }


    std::vector<float> global_v_min(12);
    std::vector<float> global_v_max(12);
    MPI_Reduce(v_min.data(), global_v_min.data(), 12, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(v_max.data(), global_v_max.data(), 12, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (master) {
      for (int j=0; j<12; j++) {
        std::cout << global_v_min[j] << " " << global_v_max[j] << "\n";
      }
    }

  } catch (exception const &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    El::ReportException(e);
    return EXIT_FAILURE;
  }

  // Clean up
  return EXIT_SUCCESS;
}
