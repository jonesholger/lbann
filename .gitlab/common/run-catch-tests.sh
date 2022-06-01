################################################################################
## Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################

#!/bin/bash

# Just in case
source ${HOME}/spack_repos/spack_${SYSTEM_NAME}.git/share/spack/setup-env.sh

# Load up the spack environment
SPACK_ARCH=$(spack arch)
SPACK_ARCH_TARGET=$(spack arch -t)
spack env activate lbann-${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET}
spack load lbann@${SPACK_ENV_NAME}-${SPACK_ARCH_TARGET} arch=${SPACK_ARCH}

# Configure the output directory
OUTPUT_DIR=${CI_PROJECT_DIR}/${RESULTS_DIR}
if [[ -d ${OUTPUT_DIR} ]];
then
    rm -rf ${OUTPUT_DIR}
fi
mkdir -p ${OUTPUT_DIR}

FAILED_JOBS=""

SPACK_BUILD_DIR=$(find . -iname "spack-build-*" -type d | head -n 1)
cd ${SPACK_BUILD_DIR}
srun --jobid=${JOB_ID} -N 1 -n 1 -t 5 \
     ./unit_test/seq-catch-tests \
     -r JUnit \
     -o ${OUTPUT_DIR}/seq-catch-results.xml
if [[ $? -ne 0 ]]; then
    FAILED_JOBS+=" seq"
fi

srun --jobid=${JOB_ID} \
     -N 2 -n $(($TEST_TASKS_PER_NODE * 2)) \
     --ntasks-per-node=$TEST_TASKS_PER_NODE \
     -t 5 ${TEST_MPIBIND_FLAG} \
     ./unit_test/mpi-catch-tests \
     -r JUnit \
     -o "${OUTPUT_DIR}/mpi-catch-results-rank=%r-size=%s.xml"
if [[ $? -ne 0 ]]; then
    FAILED_JOBS+=" mpi"
fi

srun --jobid=${JOB_ID} \
     -N 2 -n $(($TEST_TASKS_PER_NODE * 2)) \
     --ntasks-per-node=$TEST_TASKS_PER_NODE \
     -t 5 ${TEST_MPIBIND_FLAG} \
     ./unit_test/mpi-catch-tests "[filesystem]" \
     -r JUnit \
     -o "${OUTPUT_DIR}/mpi-catch-filesystem-results-rank=%r-size=%s.xml"
if [[ $? -ne 0 ]];
then
    FAILED_JOBS+=" mpi-filesystem"
fi

# Try to write a semi-useful message to this file since it's being
# saved as an artifact. It's not completely outside the realm that
# someone would look at it.
if [[ -n "${FAILED_JOBS}" ]];
then
    echo "Some Catch2 tests failed:${FAILED_JOBS}" > ${OUTPUT_DIR}/catch-tests-failed.txt
fi

# Return "success" so that the pytest-based testing can run.
exit 0
