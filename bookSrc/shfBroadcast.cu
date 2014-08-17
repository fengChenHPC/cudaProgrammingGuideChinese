__global__ void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    if (laneId == 0)        // Note unused variable for
        value = arg;        // all threads except lane 0
    value = __shfl(value, 0);   // Get ¡°value¡± from lane 0
    if (value != arg)
        printf(¡°Thread %d failed.\n¡±, threadIdx.x);
}

void main() {
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();
}

