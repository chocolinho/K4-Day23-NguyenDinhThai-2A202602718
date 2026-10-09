# Efficient Inference and Small Language Models: A Comprehensive Survey

## TL;DR
- Small language models (SLMs) in the 1B–4B parameter range, such as Phi-3 and Gemma 3, achieve strong on-device capabilities while requiring rigorous quantization and architecture engineering [1][2].
- Post-training quantization (PTQ) methods like GPTQ and AWQ show that sub-7B models are highly sensitive to aggressive 4-bit compression compared to massive LLMs, often requiring 8-bit precision or Quantization-Aware Training (QAT) to prevent catastrophic accuracy degradation [3].
- Novel 1-bit and 1.58-bit architectures (e.g., BitNet b1.58) quantize weights to ternary values $\{-1, 0, 1\}$, delivering massive speedups and energy reductions without losing perplexity [4][5].
- Inference serving engines and memory optimization kernels (vLLM with PagedAttention, FlashAttention-2 and FlashAttention-3) eliminate memory fragmentation and leverage hardware asynchrony to achieve up to 24x higher throughput [6][7][8].
- Advanced KV cache compression techniques like KIVI (2-bit quantization) and attention sink strategies (StreamingLLM) allow unbounded sequence generation with minimal memory footprints [9][10].
- Speculative decoding paradigms (Medusa, EAGLE, Recurrent Drafter) and device-aware engines (SpecMemo) overcome the memory bandwidth wall on edge and NPU hardware, yielding substantial end-to-end latency speedups [11][12][13].

## Background
The rapid deployment of Large Language Models (LLMs) across cloud datacenters and resource-constrained edge devices has catalyzed intensive research into efficient inference and compact architectures. While multi-hundred-billion parameter models dominate frontier reasoning tasks, their deployment is severely constrained by high memory bandwidth requirements, massive VRAM footprints, and substantial energy consumption. Autoregressive token generation is fundamentally memory-bound: loading model weights and key-value (KV) caches from high-bandwidth memory (HBM) for every generated token leaves arithmetic logic units (ALUs) underutilized.

To address these limitations, the research community has pursued a dual track: designing inherently compact Small Language Models (SLMs) optimized through distillation and QAT [1][2], and developing advanced runtime algorithms—including memory paging [6], low-bit KV cache compression [9], kernel-level parallelism [7][8], and speculative execution [11][12]. This survey synthesizes foundational and recent advances across these domains, examining how architectural innovations, quantization trade-offs, serving engines, and speculative decoding interact to enable real-time, low-power AI inference from edge devices to enterprise datacenters.

## SLM Architectures, Distillation, and Extreme Quantization
Building compact language models capable of rivaling larger models requires rethinking architectural design and training paradigms. Recent breakthroughs demonstrate that models in the 1B–4B parameter range can achieve remarkable proficiency through careful data curation, knowledge distillation, and structural modifications. For instance, Gemma 3 introduces an open model family ranging from 1B to 27B parameters with support for 128K contexts, utilizing knowledge distillation and QAT to produce highly compressed INT4 and FP8 weight variants [1]. Similarly, Phi-3-mini (3.8B parameters) achieves 82.5 on GSM8K and 68.8 on MMLU by training on 3.3T tokens, fitting comfortably on mobile devices with a 4-bit memory footprint of roughly 1.8 GB [2].

Despite these successes, compression trade-offs vary significantly across model scales. Evaluating four Post-Training Quantization (PTQ) methods (GPTQ, AWQ, SmoothQuant, and FP8) across 9 model families reveals that 4-bit quantization on smaller models (e.g., Llama-3.2-1B) causes severe accuracy drops—such as a 25.32% degradation on GSM8K and 16.01% on IFEval when using GPTQ [3]. This indicates that 1B-scale models require 8-bit precision (FP8 or SmoothQuant) or Quantization-Aware Training (QAT) to preserve performance, whereas AWQ consistently outperforms GPTQ in weight-only quantization across architectures [3].

To push compression limits further, 1-bit and 1.58-bit architectures have emerged. BitNet b1.58 replaces standard linear layers with ternary weight representations ($\{-1, 0, 1\}$), matching full-precision Transformer performance starting at the 3B parameter scale while achieving 2.71x faster execution and a 71.4x reduction in arithmetic energy consumption on 7nm chips [4]. Furthermore, 1.58-bit QAT has been successfully scaled down to tiny networks (100K to 48M parameters) using absmedian quantization and tuned learning rates [5].

## KV Cache Memory Optimization and Quantization
During autoregressive inference, the KV cache grows linearly with sequence length and batch size, frequently consuming up to 60%–80% of total GPU memory and causing severe fragmentation. To mitigate this memory explosion, Gemma 3 interleaves local attention layers (with a 1024-token span) with global attention layers at a 5:1 ratio, and scales RoPE base frequency to 1M on global layers [1]. Phi-3-small employs blocksparse attention layers alternating with dense layers to restrict KV cache growth during long-context retrieval [2].

At the algorithmic level, KIVI introduces a tuning-free asymmetric 2-bit quantization method for KV caches without requiring model fine-tuning [9]. By recognizing that key caches require per-channel quantization while value caches require per-token quantization, KIVI splits cache tensors into grouped parts and full-precision residuals, achieving a 2.6x peak memory reduction on LLaMA-2-7B and up to 3.47x higher throughput [9]. Complementarily, StreamingLLM resolves the infinite generation challenge by identifying the "attention sink" phenomenon—where initial tokens accumulate disproportionately high attention scores due to Softmax normalization [10]. By retaining initial attention sink tokens alongside a rolling window for recent tokens, StreamingLLM enables continuous generation without performance degradation or memory overflow [10].

## High-Throughput Inference Engines and Kernel Acceleration
Efficient serving frameworks bridge the gap between model weights and hardware execution. The vLLM serving engine introduced PagedAttention, an operating-system-inspired memory management algorithm that partitions KV caches into fixed-size physical blocks (e.g., 16 tokens) managed via block tables [6]. This eliminates internal and external memory fragmentation, reduces memory waste to under 4%, supports Copy-on-Write memory sharing for parallel beam search, and delivers up to 24x higher throughput than HuggingFace Transformers [6]. Modern inference architectures build upon this foundation with KV cache block managers, 16-token chunk prefix hashing (`long_prefix`) for prompt reuse, continuous batching, and CUDA Graph capture to eliminate kernel launch overheads [14].

At the hardware kernel level, FlashAttention-2 optimizes thread block work partitioning and non-matmul FLOP reduction to improve parallelism across sequence lengths during prefill and decoding [7]. FlashAttention-3 further exploits Hopper GPU (H100) hardware asynchrony by splitting producers and consumers into separate warp-specialized routines using Tensor Memory Accelerator (TMA) and WGMMA instructions, while introducing block quantization and Hadamard-based incoherent processing to stabilize FP8 forward passes [8].

## Speculative Decoding and On-Device Edge Deployment
Autoregressive generation on edge devices and NPUs (such as mobile chips and Huawei Ascend NPUs) is bounded by memory bandwidth limits. Speculative decoding bypasses this bottleneck by employing a lightweight draft model or multi-head prediction head to propose multiple candidate tokens, which are verified in parallel by the target model in a single forward pass. Medusa and Recurrent Drafter reduce external draft model overheads by utilizing multi-head architectures or recurrent dependency layers embedded directly within the model [12][13].

On Ascend NPUs, deploying speculative inference requires adapting static graph execution constraints, utilizing zero-copy lookup tables (`retrieve_indices`) and on-chip Gather operations to reconstruct accepted token sequences without CPU synchronization, yielding up to 1.35x speedups for short sequences [11]. Similarly, SpecMemo models memory footprints theoretically to prevent out-of-memory errors on memory-constrained devices (such as 8GB mobile hardware), pruning candidate tree branches and reducing runtime memory usage by 65% while preserving 96% of speculative throughput [12].

## Trends and Open Problems
Despite significant advancements in small language models and efficient inference, several open challenges remain:
1. **Ultra-Low-Bit Quantization Degradation:** While 1.58-bit and 2-bit quantization (BitNet, KIVI) perform well on medium-to-large models, sub-1B parameter models suffer severe accuracy degradation under aggressive compression, necessitating new QAT formulations tailored for micro-models.
2. **Dynamic Speculative Overhead:** Speculative decoding speedups degrade on long generation sequences or complex reasoning tasks where draft acceptance rates drop, creating non-linear computational overheads that can offset latency gains.
3. **Hardware-Software Co-Design for NPUs:** Edge NPUs impose rigid static graph and memory allocation constraints, complicating dynamic KV cache paging and irregular tree verification algorithms common in advanced speculative decoding.
4. **Long-Context Memory Bottlenecks:** Although local attention, sliding windows, and attention sinks mitigate KV cache growth, maintaining robust multi-document reasoning over 128K+ contexts on edge hardware remains an active frontier.

## References
[1] Gemma 3 Technical Report. arxiv. https://arxiv.org/abs/2503.19786 (2025-03-25)
[2] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arxiv. https://arxiv.org/abs/2404.14219 (2024-04-23)
[3] Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in Large Language Models From Edge to Giant. arxiv. https://arxiv.org/abs/2409.11055 (2024-09-17)
[4] The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arxiv. https://arxiv.org/abs/2402.17764 (2024-02-27)
[5] BitNet b1.58 Reloaded: State-of-the-Art Performance Also on Smaller Networks. arxiv. https://arxiv.org/abs/2407.09527 (2024-07-12)
[6] vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention. web. https://vllm.ai/blog/2023-06-20-vllm (2023-06-20)
[7] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arxiv. https://arxiv.org/abs/2307.08691 (2023-07-17)
[8] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. arxiv. https://arxiv.org/abs/2407.08608 (2024-07-11)
[9] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arxiv. https://arxiv.org/abs/2402.02750 (2024-02-04)
[10] Efficient Streaming Language Models with Attention Sinks. arxiv. https://arxiv.org/abs/2309.17453 (2023-09-29)
[11] Accelerating OpenPangu Inference on NPU via Speculative Decoding. arxiv. https://arxiv.org/abs/2603.03383 (2026-03-03)
[12] SpecMemo: Speculative Decoding is in Your Pocket. hf-search. https://huggingface.co/papers/2506.01986 (2025-05-16)
[13] Recurrent Drafter for Fast Speculative Decoding in Large Language Models. hf-search. https://huggingface.co/papers/2403.09919 (2024-03-14)
[14] Anatomy of a High-Throughput LLM Inference System. web. https://vllm.ai/blog/2025-09-05-anatomy-of-vllm (2025-09-05)
