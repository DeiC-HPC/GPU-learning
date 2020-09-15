all:
	g++ -ftree-vectorize -fopt-info-vec-missed -O3 -fopenmp -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" matrixaddition.cpp -o matrixaddition
	g++ -ftree-vectorize -fopt-info-vec-missed -O3 -fopenmp -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" matrixadditionoptimized.cpp -o matrixadditionoptimized

clean:
	rm matrixaddition matrixadditionoptimized
