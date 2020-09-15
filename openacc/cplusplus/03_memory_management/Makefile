all:
	g++ -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" mapreduce.cpp -o mapreduce
	g++ -O3 -fopenacc -fno-stack-protector -foffload="-lm" -foffload="-misa=sm_35" mapreduceoptimized.cpp -o mapreduceoptimized

clean:
	rm mapreduce mapreduceoptimized
