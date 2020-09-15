all:
	g++ -O3 mandelbrot.cpp -o mandelbrot
	g++ -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" mandelbrot_openacc.cpp -o openacc

clean:
	rm mandelbrot openacc
