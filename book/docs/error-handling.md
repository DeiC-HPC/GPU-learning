# Error handling

When using the GPU, things can go wrong. But it will not crash our program.
Therefore it is important to check for errors when using the GPU.

When communicating with the GPU you will get a status response. With this
you can check whether it went well or something went wrong.
With communication we mean every memory allocation, copy or synchronisation
happening between the CPU and GPU.

=== "CUDA"
    In CUDA every function will return a value of type `cudaError_t`. This we
    can use to check if everything went well by comparing it to the value
    `cudaSuccess`.

    ```c++ linenums="1"
    float* p;
    cudaError_t status;
    // This will produce error.
    status = cudaMalloc(&p, 1000000 * sizeof(float));
    if (status != cudaSuccess) {
      // handle error
    }
    ```

=== "HIP"
    In HIP every function will return a value of type `hipError_t`. This we
    can use to check if everything went well by comparing it to the value
    `hipSuccess`.

    ```c++ linenums="1"
    float* p;
    hipError_t status;
    // This will produce error.
    status = hipMalloc(&p, 1000000 * sizeof(float));
    if (status != hipSuccess) {
      // handle error
    }
    ```

Checking error message
----------------------

When you have received an error from the GPU, you probably want to know what
went wrong. There is also functionality for that. Of course you could just look
up the error code. But it is also possible to get it printed out.

=== "CUDA"

    ```c++ linenums="1"
    cudaError_t status;
    cudaGetErrorString(status)
    ```

=== "HIP"

    ```c++ linenums="1"
    hipError_t status;
    hipGetErrorString(status)
    ```

a
-

=== "CUDA"

    ```c++ linenums="1"
    #define CUDA_ERROR_CHECK(err) cuda_error_check(err, __FILE__, __LINE__)
    template <typename T>
    void cuda_error_check(T err, const char *file, int line) {
        if (val != cudaSuccess) {
            std::cout << "CUDA error: " << cudaGetErrorString(err) << ", file: " << file << " line: " << line << std::endl;
        }
    }
    ```

=== "HIP"

    ```c++ linenums="1"
    #define HIP_ERROR_CHECK(err) hip_error_check(err, __FILE__, __LINE__)
    template <typename T>
    void hip_error_check(T err, const char *file, int line) {
        if (val != hipSuccess) {
            std::cout << "HIP error: " << hipGetErrorString(err) << ", file: " << file << " line: " << line << std::endl;
        }
    }
    ```