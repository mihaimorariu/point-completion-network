cuda_inc = /home/mmorariu/.conda/envs/pcn/include/
cuda_lib = /home/mmorariu/.conda/envs/pcn/lib/
nvcc     = /home/mmorariu/.conda/envs/pcn/bin/nvcc
tf_inc   = /home/mmorariu/.conda/envs/pcn/lib/python3.6/site-packages/tensorflow/include
tf_lib   = /home/mmorariu/.conda/envs/pcn/lib/python3.6/site-packages/tensorflow

all: tf_nndistance_so.so tf_approxmatch_so.so

tf_nndistance.cu.o: tf_nndistance.cu
	$(nvcc) tf_nndistance.cu -o tf_nndistance.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_nndistance_so.so: tf_nndistance.cpp tf_nndistance.cu.o
	g++ tf_nndistance.cpp tf_nndistance.cu.o -o tf_nndistance_so.so -shared -std=c++11 -fPIC -O2 \
            -I $(tf_inc) -D_GLIBCXX_USE_CXX11_ABI=1 -L $(tf_lib) -ltensorflow_framework -L $(cuda_lib) -lcudart

tf_approxmatch.cu.o: tf_approxmatch.cu
	$(nvcc) tf_approxmatch.cu -o tf_approxmatch.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_approxmatch_so.so: tf_approxmatch.cpp tf_approxmatch.cu.o
	g++ -shared $(CPPFLAGS) tf_approxmatch.cpp tf_approxmatch.cu.o -o tf_approxmatch_so.so -shared -std=c++11 -fPIC -O2 \
            -I $(tf_inc) -D_GLIBCXX_USE_CXX11_ABI=1 -L $(tf_lib) -ltensorflow_framework -L $(cuda_lib) -lcudart

clean:
	rm -rf *.o *.so
