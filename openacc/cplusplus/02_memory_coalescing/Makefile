all:
	g++ -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" matrixaddition.cpp -o matrixaddition
	g++ -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" matrixadditionoptimized.cpp -o matrixadditionoptimized

clean:
	rm matrixaddition matrixadditionoptimized
