// MIT License
//
// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hipsparselt_utils.hpp"

#include <hipsparselt/hipsparselt.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

int main()
{
    // 1. Set up input data.
    // hipSPARSELt requires minimum matrix dimensions
    constexpr int m = 128;
    constexpr int k = 128;
    constexpr int n = 128;

    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    // Scalar alpha and beta
    constexpr float alpha = 1.0f;
    constexpr float beta  = 0.0f;

    std::cout << "hipSPARSELt Basic SPMM Example" << std::endl;
    std::cout << "Matrix dimensions: " << m << "x" << k << " x " << k << "x" << n << std::endl;

    // 2. Initialize matrices with 2:4 structured sparsity
    std::vector<int8_t> h_A(m * k);
    std::vector<int8_t> h_B(k * n);
    std::vector<int8_t> h_C(m * n, 0);

    // Create structured 2:4 sparsity pattern for matrix A
    std::mt19937                       rng(42);
    std::uniform_int_distribution<int> dist(-10, 10);

    for(int i = 0; i < m * k; i += 4)
    {
        // For every 4 consecutive elements, keep exactly 2 non-zero
        std::array<int, 4> values;
        for(int j = 0; j < 4; ++j)
        {
            values[j] = dist(rng);
        }

        // Sort indices by absolute value to keep the 2 largest
        std::array<int, 4> indices = {0, 1, 2, 3};
        std::sort(indices.begin(),
                  indices.end(),
                  [&](int a, int b) { return std::abs(values[a]) > std::abs(values[b]); });

        // Set values: keep 2 largest, zero out 2 smallest
        for(int j = 0; j < 4; ++j)
        {
            if(i + j < m * k)
            {
                h_A[i + j] = (j < 2) ? static_cast<int8_t>(values[indices[j]]) : int8_t(0);
            }
        }
    }

    // Initialize dense matrix B
    for(int i = 0; i < k * n; ++i)
    {
        h_B[i] = static_cast<int8_t>(dist(rng));
    }

    // 3. Allocate device memory
    int8_t* d_A{};
    int8_t* d_B{};
    int8_t* d_C{};

    const size_t A_size = sizeof(int8_t) * m * k;
    const size_t B_size = sizeof(int8_t) * k * n;
    const size_t C_size = sizeof(int8_t) * m * n;

    HIP_CHECK(hipMalloc(&d_A, A_size));
    HIP_CHECK(hipMalloc(&d_B, B_size));
    HIP_CHECK(hipMalloc(&d_C, C_size));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), C_size, hipMemcpyHostToDevice));

    // 4. Initialize hipSPARSELt
    hipsparseLtHandle_t handle;
    HIPSPARSELT_CHECK(hipsparseLtInit(&handle));

    // 5. Create matrix descriptors
    hipsparseLtMatDescriptor_t descr_A{};
    hipsparseLtMatDescriptor_t descr_B{};
    hipsparseLtMatDescriptor_t descr_C{};

    // Structured sparse matrix A
    HIPSPARSELT_CHECK(hipsparseLtStructuredDescriptorInit(&handle,
                                                          &descr_A,
                                                          m,
                                                          k,
                                                          lda,
                                                          16, // alignment
                                                          HIP_R_8I,
                                                          HIPSPARSE_ORDER_COL,
                                                          HIPSPARSELT_SPARSITY_50_PERCENT));

    // Dense matrix B
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(&handle,
                                                     &descr_B,
                                                     k,
                                                     n,
                                                     ldb,
                                                     16, // alignment
                                                     HIP_R_8I,
                                                     HIPSPARSE_ORDER_COL));

    // Dense matrix C
    HIPSPARSELT_CHECK(hipsparseLtDenseDescriptorInit(&handle,
                                                     &descr_C,
                                                     m,
                                                     n,
                                                     ldc,
                                                     16, // alignment
                                                     HIP_R_8I,
                                                     HIPSPARSE_ORDER_COL));

    // 6. Create matmul descriptor, algorithm selection, and plan
    hipsparseLtMatmulDescriptor_t matmul_descr;
    HIPSPARSELT_CHECK(hipsparseLtMatmulDescriptorInit(&handle,
                                                      &matmul_descr,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &descr_A,
                                                      &descr_B,
                                                      &descr_C,
                                                      &descr_C,
                                                      HIPSPARSELT_COMPUTE_32I));

    hipsparseLtMatmulAlgSelection_t alg_selection;
    HIPSPARSELT_CHECK(hipsparseLtMatmulAlgSelectionInit(&handle,
                                                        &alg_selection,
                                                        &matmul_descr,
                                                        HIPSPARSELT_MATMUL_ALG_DEFAULT));

    hipsparseLtMatmulPlan_t plan;
    HIPSPARSELT_CHECK(hipsparseLtMatmulPlanInit(&handle, &plan, &matmul_descr, &alg_selection));

    // 7. Get workspace size and allocate if needed
    size_t workspace_size;
    HIPSPARSELT_CHECK(hipsparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size));

    void* d_workspace{};
    if(workspace_size > 0)
    {
        HIP_CHECK(hipMalloc(&d_workspace, workspace_size));
    }

    // 8. Compress the sparse matrix A
    size_t compressed_size, compress_buffer_size;
    HIPSPARSELT_CHECK(
        hipsparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compress_buffer_size));

    void* d_A_compressed{};
    void* d_compress_buffer{};

    HIP_CHECK(hipMalloc(&d_A_compressed, compressed_size));
    if(compress_buffer_size > 0)
    {
        HIP_CHECK(hipMalloc(&d_compress_buffer, compress_buffer_size));
    }

    HIPSPARSELT_CHECK(
        hipsparseLtSpMMACompress(&handle, &plan, d_A, d_A_compressed, d_compress_buffer, 0));

    // 9. Perform sparse matrix multiplication
    // C = alpha * A * B + beta * C
    HIPSPARSELT_CHECK(hipsparseLtMatmul(&handle,
                                        &plan,
                                        &alpha,
                                        d_A_compressed,
                                        d_B,
                                        &beta,
                                        d_C,
                                        d_C,
                                        d_workspace,
                                        nullptr,
                                        1));

    // 10. Copy result back to host
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, C_size, hipMemcpyDeviceToHost));

    // 11. Print sample results (first 4x4 for readability)
    std::cout << "\nResult matrix C (first 4x4):" << std::endl;
    for(int i = 0; i < 4; ++i)
    {
        std::cout << "  ";
        for(int j = 0; j < 4; ++j)
        {
            std::printf("%4d", static_cast<int>(h_C[i + j * ldc]));
        }
        std::cout << std::endl;
    }

    // 12. Cleanup
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&descr_A));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&descr_B));
    HIPSPARSELT_CHECK(hipsparseLtMatDescriptorDestroy(&descr_C));
    HIPSPARSELT_CHECK(hipsparseLtMatmulPlanDestroy(&plan));
    HIPSPARSELT_CHECK(hipsparseLtDestroy(&handle));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipFree(d_A_compressed));
    if(d_workspace)
        HIP_CHECK(hipFree(d_workspace));
    if(d_compress_buffer)
        HIP_CHECK(hipFree(d_compress_buffer));

    std::cout << "\nhipSPARSELt basic SPMM example completed successfully!" << std::endl;

    return 0;
}
