all:
	gfortran -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" mapreduce.f90 -o mapreduce
	gfortran -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" mapreduceoptimized.f90 -o mapreduceoptimized

clean:
	rm mapreduce mapreduceoptimized
