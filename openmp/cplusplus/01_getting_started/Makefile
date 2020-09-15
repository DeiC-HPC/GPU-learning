all:
	g++ -O3 mandelbrot.cpp -o mandelbrot
	g++ -O3 -fopenmp -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" mandelbrot_openmp.cpp -o openmp

clean:
	rm mandelbrot mandelbrot_openmp
