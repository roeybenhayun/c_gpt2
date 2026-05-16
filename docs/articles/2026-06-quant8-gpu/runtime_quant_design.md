This is the second phase, run time quantization

1. INT8 support

1.1 add USE_INT8 compile time flag
1.2 This flag would still requires the USE_BF16, with this a new typedef will be added - qweight_t (this is not a replacement
for the weight_t). 
1.3 Declare the new int8 quantized pointers toghether with the bf16 pointers but under the compile time flag (USE_INT8) (I can see this already in the loader changes phase)
1.4 allocate device memory for the new int8 gemm weight and scale
1.5 add a stub to GEMM dispatch to support INT8 mode. 


2. Loader changes
- New fread_int8_with_scale helper that reads the INT8 weights + FP32 scale   
  block pair into device memory with cudaMemcpy.
  - Existing fread_weights_or_exit stays untouched for the preserved FP32       
  tensors (wte, wpe, LN, biases) — they still get the FP32→BF16 cast at load.   
  - Per-layer device-side state grows by 4 pointers + 4 scale vectors
  (W_qkv_int8, scale_W_qkv[3*d_model], etc.).                                   
  - Loader path selection by compile-time flag — file format is positional, no
  header to inspect.

3. Activation quantization kernel (this is a new CUDA kernel)
- Input: X_bf16 [tokens, d]. Output: X_int8 [tokens, d] + scale_X [tokens].   
  - Per-token dynamic: each row computes amax = max(|x|) → scale_X = amax / 127
  → int8 = round(x / scale_X).                                                  
  - Reduction kernel — one block per token, threads cooperate on amax, then a
  second pass to write INT8.                                                    
  - No calibration, no offline statistics. Recomputed every forward pass.
  - Make sure to not use the thread_max variable name

4. cuBLAS INT8 GEMM call 
- cublasGemmEx with:                                                          
    - A, B = CUDA_R_8I
    - C (output) = CUDA_R_32I                                                   
    - compute type = CUBLAS_COMPUTE_32I
    - algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP                                      
  - Both inputs INT8 — that's the W8A8 commitment forced by cuBLAS.             
  - Replaces the current BF16 cublasGemmEx calls one-for-one in W_qkv,          
  attn_proj, W1, W2 paths only. Other ops stay BF16.  

5.  Dequant + bias-add kernel (fused, new CUDA kernel)                         
                  
  - Input: Y_int32 [tokens, out], scale_W [out], scale_X [tokens], bias_bf16    
  [out].
  - Output: Y_bf16 [tokens, out] = scale_W[j] * scale_X[i] * Y_int32[i,j] +     
  bias[j]                                                                       
  - Fused so you don't materialize int32 intermediates more than necessary.
  - Output is BF16 — the rest of the network (LN, GELU, residual) stays as it is
   today.

6. Per-arch capability gating
 - INT8 tensor cores need SM ≥ 7.5 (Turing+). Maxwell/Pascal don't have them.  
  - At startup, check cudaDeviceProp.major/minor. If unsupported, either refuse
  to launch or fall back to BF16 path.                                          
  - RTX 5080 (SM 12.0) and Jetson Xavier NX (SM 7.2 — Volta, has DP4A but not
  real tensor-core INT8) need different decisions — Xavier may need to refuse   
  since SM 7.2 lacks IMMA.
  Roey - here if SM does not support INT8 refuse with a log message and exit.)

7. Build system 

  - New out/gpu/int8/ directory parallel to out/gpu/.                           
  - make gpu int8 <size> target — sets -DGPT2_USE_INT8 (or whatever flag), links
   the two new kernels, selects the _quant8.bin file in the loader.             
  - setup.sh already needs the download branch you noted (after HF upload).
