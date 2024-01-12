# Overlapping data transfers
When working with GPUs, one of the performance bottlenecks you might experience
is communication between the CPU and GPU. Every time you send or recieve data,
or even when starting kernels, you communicate with the GPU.

Some of the latency, you will see here, we can remove. By sending and receiving
data asynchronously while the GPU is working, we can minimise the time in
between running kernels.

