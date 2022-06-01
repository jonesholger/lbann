#!/bin/sh

# Function to setup the following fields based on the center that you are running at:
# CENTER
# BUILD_SUFFIX
# COMPILER
# NINJA_NUM_PROCESSES
set_center_specific_fields()
{
    SYS=$(uname -s)

    if [[ ${SYS} = "Darwin" ]]; then
        CENTER="osx"
        COMPILER="clang"
        BUILD_SUFFIX=llnl.gov
    else
        CORI=$([[ $(hostname) =~ (cori|cgpu) ]] && echo 1 || echo 0)
        PERLMUTTER=$([[ $(printenv LMOD_SYSTEM_NAME) =~ (perlmutter) ]] && echo 1 || echo 0)
#	LMOD_SITE_NAME
	DOMAINNAME=$(python3 -c 'import socket; domain = socket.getfqdn().split("."); print(domain[-2] + "." + domain[-1]) if len(domain) > 1 else print(domain)')
#	domainname -A | egrep -o '[a-z]+\.[a-z]+( |$)' | sort -u
        if [[ ${CORI} -eq 1 || ${PERLMUTTER} -eq 1 ]]; then
            CENTER="nersc"
        elif [[ ${DOMAINNAME} = "ornl.gov" ]]; then
            CENTER="olcf"
            NINJA_NUM_PROCESSES=16 # Don't let OLCF kill build jobs
        elif [[ ${DOMAINNAME} = "llnl.gov" ]]; then
            CENTER="llnl_lc"
        elif [[ ${DOMAINNAME} = "riken.jp" ]]; then
            CENTER="riken"
        else
            CENTER="unknown"
        fi
        COMPILER="gnu"
    fi
}

# Function to setup the following fields based on the center that you are running at:
# GPU_ARCH_VARIANTS
# CMAKE_GPU_ARCH
set_center_specific_gpu_arch()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "power9le") # Lassen
                GPU_ARCH_VARIANTS="cuda_arch=70"
                CMAKE_GPU_ARCH="70"
                ;;
            "power8le") # Ray
                GPU_ARCH_VARIANTS="cuda_arch=60"
                CMAKE_GPU_ARCH="60"
                ;;
            "broadwell") # Pascal
                GPU_ARCH_VARIANTS="cuda_arch=60"
                CMAKE_GPU_ARCH="60"
                ;;
            "haswell") # RZHasGPU
                GPU_ARCH_VARIANTS="cuda_arch=37"
                CMAKE_GPU_ARCH="37"
                ;;
            "ivybridge") # Catalyst
                ;;
            "sandybridge") # Surface
                GPU_ARCH_VARIANTS="cuda_arch=35"
                CMAKE_GPU_ARCH="35"
                ;;
            "zen" | "zen2") # Corona
                # Use a HIP Clang variant
                GPU_ARCH_VARIANTS="amdgpu_target=gfx906 %clang@amd"
                ;;
            *)
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "zen3") # Perlmutter
                GPU_ARCH_VARIANTS="cuda_arch=80"
                CMAKE_GPU_ARCH="80"
                ;;
            *)
                ;;
        esac
    fi
}


set_center_specific_modules()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        # Disable the StdEnv for systems in LC
        case ${spack_arch_target} in
            "power9le") # Lassen
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.1.1 spectrum-mpi/rolling-release python/3.7.2"
                # MODULE_CMD="module --force unload StdEnv; module load clang/12.0.1 cuda/11.1.1 spectrum-mpi/rolling-release python/3.7.2 essl/6.2.1"
                ;;
            "power8le") # Ray
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.1.1 spectrum-mpi/rolling-release python/3.7.2"
                # MODULE_CMD="module --force unload StdEnv; module load clang/12.0.1 cuda/11.1.1 spectrum-mpi/rolling-release python/3.7.2 essl/6.2.1"
                ;;
            "broadwell" | "haswell" | "sandybridge") # Pascal, RZHasGPU, Surface
                MODULE_CMD_GCC="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.4.1 mvapich2/2.3 python/3.7.2"
                MODULE_CMD_CLANG="module --force unload StdEnv; module load clang/12.0.1 cuda/11.4.1 mvapich2/2.3.6 python/3.7.2"
                ;;
            "ivybridge" | "cascadelake") # Catalyst, Ruby
                MODULE_CMD="module --force unload StdEnv; module load gcc/10.2.1 mvapich2/2.3 python/3.7.2"
                ;;
            "zen" | "zen2") # Corona
                MODULE_CMD="module --force unload StdEnv; module load clang/11.0.0 python/3.7.2 opt rocm/4.2.0 openmpi-gnu/4.0"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le")
                MODULE_CMD="module load gcc/9.3.0 cuda/11.1.1 spectrum-mpi/10.3.1.2-20200121"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
                MODULE_CMD="module purge; module load cgpu modules/3.2.11.4 gcc/8.3.0 cuda/11.1.1 openmpi/4.0.3 cmake/3.18.2"
                ;;
            "zen3") # Perlmutter
		MODULE_CMD="module load PrgEnv-cray craype-x86-rome libfabric/1.11.0.4.75 craype-network-ofi cmake/3.22.0 cce/13.0.0 craype/2.7.13 cray-mpich/8.1.12 cray-libsci/21.08.1.2 PrgEnv-cray/8.2.0 nccl/2.11.4 cudnn/8.2.0 cray-python/3.9.4.2 craype-accel-host cudatoolkit/21.9_11.4"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                MODULE_CMD="module load system/fx700 gcc/10.1 gcc/openmpi/4.0.3"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    else
        echo "No pre-specified modules found for this system. Make sure to setup your own"
    fi
}

set_center_specific_spack_dependencies()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        if [[ -z "${SKIP_MIRRORS:-}" ]]; then
            POSSIBLE_MIRRORS="/p/vast1/lbann/spack/mirror /p/vast1/atom/spack/mirror"
            for m in ${POSSIBLE_MIRRORS}
            do
                if [[ -r "${m}" ]]; then
                    MIRRORS="${m} $MIRRORS"
                fi
            done
        fi
        # MIRRORS="/p/vast1/lbann/spack/mirror /p/vast1/atom/spack/mirror"
        case ${spack_arch_target} in
            "power9le") # Lassen
                CENTER_COMPILER="%gcc"
                CENTER_DEPENDENCIES="^spectrum-mpi ^openblas@0.3.12 threads=openmp ^cuda@11.1.105 ^libtool@2.4.2 ^python@3.9.10"
                CENTER_BLAS_LIBRARY="blas=openblas"
                # CENTER_COMPILER="%clang"
                # CENTER_DEPENDENCIES="^spectrum-mpi ^cuda@11.1.105 ^libtool@2.4.2 ^python@3.9.10"
                # CENTER_BLAS_LIBRARY="blas=essl"
                ;;
            "power8le") # Ray
                CENTER_COMPILER="%gcc"
                CENTER_DEPENDENCIES="^spectrum-mpi ^openblas@0.3.12 threads=openmp ^cuda@11.1.105 ^libtool@2.4.2 ^python@3.9.10"
                CENTER_BLAS_LIBRARY="blas=openblas"
                # CENTER_COMPILER="%clang"
                # CENTER_DEPENDENCIES="%clang ^spectrum-mpi ^cuda@11.1.105 ^libtool@2.4.2 ^py-packaging@17.1 ^python@3.9.10 ^py-scipy@1.6.3"
                # CENTER_BLAS_LIBRARY="blas=essl"
                ;;
            "broadwell" | "haswell" | "sandybridge") # Pascal, RZHasGPU, Surface
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%clang"
                CENTER_DEPENDENCIES="^mvapich2 ^hwloc@1.11.13 ^libtool@2.4.2 ^python@3.9.10"
                ;;
            "ivybridge" | "cascadelake") # Catalyst, Ruby
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%gcc"
                CENTER_DEPENDENCIES="^mvapich2 ^hwloc@1.11.13 ^libtool@2.4.2 ^python@3.9.10"
                ;;
            "zen" | "zen2") # Corona
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%clang"
                CENTER_DEPENDENCIES="^openmpi ^hwloc@2.3.0"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le")
                CENTER_DEPENDENCIES="^spectrum-mpi ^openblas@0.3.12"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
                CENTER_DEPENDENCIES="^openmpi"
                ;;
            "zen3") # Perlmutter
                CENTER_COMPILER="%cce@13.0.0"
                CENTER_DEPENDENCIES="^mpich@8.1.12 ^python@3.9.4 ^cuda+allow-unsupported-compilers"
                CENTER_BLAS_LIBRARY="blas=libsci"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                CENTER_DEPENDENCIES="^openmpi"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "osx" ]]; then
        case ${spack_arch_target} in
            "skylake")
                CENTER_DEPENDENCIES="^hdf5+hl"
                CENTER_BLAS_LIBRARY="blas=accelerate"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    else
        echo "No center found and no center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
    fi
}

set_center_specific_externals()
{
    local center="$1"
    local spack_arch_target="$2"
    local spack_arch="$3"
    local yaml="$4"

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "broadwell" | "haswell" | "sandybridge" | "ivybridge")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            "power9le" | "power8le")
cat <<EOF  >> ${yaml}
  packages:
    essl:
      buildable: False
      version:
      - 6.2.1
      externals:
      - spec: essl@6.2.1 arch=${spack_arch}
        modules:
        - essl/6.2.1
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            "zen" | "zen2")
cat <<EOF  >> ${yaml}
  compilers:
    - compiler:
        spec: clang@amd
        paths:
          cc: /opt/rocm-4.2.0/llvm/bin/clang
          cxx: /opt/rocm-4.2.0/llvm/bin/clang++
          f77: /usr/bin/gfortran
          fc: /usr/bin/gfortran
        flags: {}
        operating_system: rhel7
        target: x86_64
        modules: []
        environment: {}
        extra_rpaths: []
  packages:
    hip:
      buildable: False
      version:
      - 4.2.0
      externals:
      - spec: hip@4.2.0 arch=${spack_arch}
        prefix: /opt/rocm-4.2.0/hip
        extra_attributes:
          compilers:
            c: /opt/rocm-4.2.0/llvm/bin/clang
            c++: /opt/rocm-4.2.0/llvm/bin/clang++
            hip: /opt/rocm-4.2.0/hip/bin/hipcc
    hipcub:
      buildable: False
      version:
      - 4.2.0
      externals:
      - spec: hipcub@4.2.0 arch=${spack_arch}
        prefix: /opt/rocm-4.2.0/hipcub
        extra_attributes:
          compilers:
            c: /opt/rocm-4.2.0/llvm/bin/clang
            c++: /opt/rocm-4.2.0/llvm/bin/clang++
    hsa-rocr-dev:
      buildable: False
      version:
      - 4.2.0
      externals:
      - spec: hsa-rocr-dev@4.2.0 arch=${spack_arch}
        prefix: /opt/rocm-4.2.0
        extra_attributes:
          compilers:
            c: /opt/rocm-4.2.0/llvm/bin/clang
            c++: /opt/rocm-4.2.0/llvm/bin/clang++
    llvm-amdgpu:
      buildable: False
      version:
      - 4.2.0
      externals:
      - spec: llvm-amdgpu@4.2.0 arch=${spack_arch}
        prefix: /opt/rocm-4.2.0/llvm
        extra_attributes:
          compilers:
            c: /opt/rocm-4.2.0/llvm/bin/clang
            c++: /opt/rocm-4.2.0/llvm/bin/clang++
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
    openmpi:
      buildable: False
      version:
      - 4.0
      externals:
      - spec: openmpi@4.0.0 arch=${spack_arch}
        prefix: /opt/openmpi/4.0/gnu
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le" | "power8le")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            "zen3") #perlmutter
cat <<EOF  >> ${yaml}
  packages:
    all:
      providers:
        mpi: [mpich]
    nvhpc:
      buildable: False
      version:
      - 21.9
      externals:
      - spec: nvhpc@21.9 arch=${spack_arch}
        modules:
        - cudatoolkit/21.9_11.4
    cudnn:
      buildable: False
      version:
      - 8.2.0
      externals:
      - spec: cudnn@8.2.0 arch=${spack_arch}
        modules:
        - cudnn/8.2.0
    cray-libsci:
      buildable: False
      version:
      - 21.08.1.2
      externals:
      - spec: cray-libsci@21.08.1.2 arch=${spack_arch}
        modules:
        - cray-libsci/21.08.1.2
    mpich:
      buildable: False
      version:
      - 8.1.12
      externals:
      - spec: "mpich@8.1.12 arch=${spack_arch}"
        modules:
        - cray-mpich/8.1.12
    nccl:
      buildable: False
      version:
      - 2.11.4
      externals:
      - spec: nccl@2.11.4 arch=${spack_arch}
        modules:
        - nccl/2.11.4
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            *)
        echo "No center-specified externals."
                ;;
        esac
    else
        echo "No center-specified externals."
    fi

    echo "Setting up a sane definition of how to represent modules."
cat <<EOF >> ${yaml}
  modules:
    default:
      lmod:
        core_compilers:
          - 'cce@13.0.0'
EOF
}

cleanup_clang_compilers()
{
    local center="$1"
    local yaml="$2"

    if [[ ${center} = "llnl_lc" ]]; then
	# Point compilers that don't have a fortran compiler a default one
	sed -i.sed_bak -e 's/\([[:space:]]*f[c7]7*:[[:space:]]*\)null$/\1\/usr\/tce\/packages\/gcc\/gcc-8.3.1\/bin\/gfortran/g' ${yaml}
	echo "Updating Clang compiler's to see the gfortran compiler."

        # LC uses a old default gcc and clang needs a newer default gcc toolchain
        # Also set LC clang compilers to use lld for faster linking ldflags: -fuse-ld=lld
        perl -i.perl_bak -0pe 's/(- compiler:.*?spec: clang.*?flags:) (\{\})/$1 \{cflags: --gcc-toolchain=\/usr\/tce\/packages\/gcc\/gcc-8.3.1, cxxflags: --gcc-toolchain=\/usr\/tce\/packages\/gcc\/gcc-8.3.1\}/smg' ${yaml}
    else
	# Point compilers that don't have a fortran compiler a default one
	sed -i.sed_bak -e 's/\([[:space:]]*f[c7]7*:[[:space:]]*\)null$/\1\/usr\/bin\/gfortran/g' ${yaml}
	echo "Updating Clang compiler's to see the gfortran compiler."
    fi
}

set_center_specific_variants()
{
    local center="$1"
    local spack_arch_target="$2"

    STD_USER_VARIANTS="+vision +numpy"
    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "power9le" | "power8le" | "broadwell" | "haswell" | "sandybridge") # Lassen, Ray, Pascal, RZHasGPU, Surface
                CENTER_USER_VARIANTS="+cuda"
                ;;
            "ivybridge") # Catalyst
                CENTER_USER_VARIANTS="+onednn"
                ;;
            "zen" | "zen2") # Corona
                CENTER_USER_VARIANTS="+rocm"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le") # Summit
                CENTER_USER_VARIANTS="+cuda"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512") # CoriGPU
                CENTER_USER_VARIANTS="+cuda"
                ;;
            "zen3") # Perlmutter
                CENTER_USER_VARIANTS="+cuda"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                CENTER_USER_VARIANTS="+onednn"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    else
        echo "No center found and no center-specified CENTER_USER_VARIANTS."
    fi
    CENTER_USER_VARIANTS+=" ${STD_USER_VARIANTS}"
}
