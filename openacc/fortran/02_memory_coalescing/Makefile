all:
	gfortran -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" matrixaddition.f90 -o matrixaddition
	gfortran -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" matrixadditionoptimized.f90 -o matrixadditionoptimized

clean:
	rm matrixaddition matrixadditionoptimized
