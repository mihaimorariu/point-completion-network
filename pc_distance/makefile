cuda_inc = /opt/cuda/include/
cuda_lib = /opt/cuda/lib64/
nvcc = /opt/cuda/bin/nvcc
tf_lib = /home/mmorariu/.local/lib/python3.8/site-packages/tensorflow
tf_inc = $(tf_lib)/include

all: tf_nndistance_so.so tf_approxmatch_so.so

tf_nndistance.cu.o: tf_nndistance.cu
	$(nvcc) tf_nndistance.cu -o tf_nndistance.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_nndistance_so.so: tf_nndistance.cpp tf_nndistance.cu.o
	g++ tf_nndistance.cpp tf_nndistance.cu.o -o tf_nndistance_so.so \
	-I $(cuda_inc) -I $(tf_inc) -L $(cuda_lib) -lcudart -L $(tf_lib)/libtensorflow_framework.so.2 \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

tf_approxmatch.cu.o: tf_approxmatch.cu
	$(nvcc) tf_approxmatch.cu -o tf_approxmatch.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_approxmatch_so.so: tf_approxmatch.cpp tf_approxmatch.cu.o
	g++ -shared $(CPPFLAGS) tf_approxmatch.cpp tf_approxmatch.cu.o -o tf_approxmatch_so.so \
	-I $(cuda_inc) -I $(tf_inc) -L $(cuda_lib) -lcudart -L $(tf_lib)/libtensorflow_framework.so.2 \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

clean:
	rm -rf *.o *.so
