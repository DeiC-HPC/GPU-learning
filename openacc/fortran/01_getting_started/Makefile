all:
	gfortran -O3 mandelbrot.f90 -o mandelbrot
	gfortran -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" mandelbrot_openacc.f90 -o openacc

clean:
	rm mandelbrot openacc
