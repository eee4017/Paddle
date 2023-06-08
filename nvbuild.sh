set -Eeuox pipefail
PYVER=3.8

BUILD_DIR=build
export CCACHE_DIR=/dev/shm/.ccache
SKIP_DOWNLOAD_INFERENCE_DATA=ON

RERUN_CMAKE=true
#RERUN_CMAKE=false

ulimit -n 4096

if $RERUN_CMAKE; then
\time --verbose cmake -B${BUILD_DIR} -S. \
    -GNinja \
    -DINFERENCE_DEMO_INSTALL_DIR=/workspace/paddle/data \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS='-march=sandybridge -mtune=broadwell -Wno-error=maybe-uninitialized -fuse-ld=lld' \
    -DCMAKE_CUDA_FLAGS='-t0 --forward-unknown-to-host-compiler -Xfatbin=-compress-all -march=sandybridge -mtune=broadwell -fuse-ld=lld' \
    -DCUDA_ARCH_NAME=Manual \
    -DCUDA_ARCH_BIN="80" \
    -DRUN_GENERATE_CUTLASS_KERNEL=ON \
    -DWITH_INCREMENTAL_COVERAGE=OFF \
    -DWITH_INFERENCE_API_TEST=ON \
    -DWITH_DISTRIBUTE=ON \
    -DWITH_COVERAGE=OFF \
    -DWITH_TENSORRT=ON \
    -DWITH_TESTING=ON \
    -DWITH_CONTRIB=ON \
    -DWITH_ROCM=OFF \
    -DWITH_RCCL=OFF \
    -DWITH_STRIP=ON \
    -DWITH_MKL=OFF \
    -DWITH_AVX=OFF \
    -DWITH_GPU=ON \
    -DWITH_PYTHON=ON \
    -DWITH_UNITY_BUILD=ON \
    -DPY_VERSION=$PYVER \
    -Wno-dev 2>&1 | tee c-${BUILD_DIR}.log
fi

if dpkg-query -W -f='${Status}' lld 2>/dev/null | grep -q "ok installed"; then
    echo "lld installed"
else
    echo "lld not installed"
    sudo apt install -y lld
fi

if [ -d "/usr/include/mct" ]; then
    echo "Directory exists."
else
    echo "Directory does not exist."
    cd /tmp 
    wget -q https://pslib.bj.bcebos.com/libmct/libmct.tar.gz 
    tar xf libmct.tar.gz 
    sudo cp -r libmct/include/mct /usr/include
    cd -
fi



#cmake --build ${BUILD_DIR} 2>&1 | tee build.log
ninja paddle_python -C ${BUILD_DIR} 2>&1 | tee ${BUILD_DIR}.log
#ninja test_multihead_matmul_fuse_pass -C ${BUILD_DIR}

#pip install -U build/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
pip install --force-reinstall --no-deps -U build/python/dist/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl
