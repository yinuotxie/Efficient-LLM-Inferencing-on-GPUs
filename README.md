
# Efficient Large Language Model (LLM) Inferencing on GPUs 

## Introduction
Large language models (LLMs) like ChatGPT or Llama have recently gained a lot of attention. However, operating them is still quite costly. The expense of generating a single response, which might be around $0.01 for a brief interaction using an 8xA100 instance on AWS at the moment, can become substantial when considering billions of users engaging in multiple interactions daily. Certain tasks, such as code auto-completion which activates with each new character typed, are particularly resource-intensive. As LLMs are increasingly employed in various applications, even minor improvements in generation efficiency can lead to significant overall cost reductions.

The process of LLM inference, or "decoding", is sequential, with tokens being produced **one at a time**. To generate complete sentences of `N` tokens, the model must go through `N` iterations. The good news is that it's possible to store previously computed tokens, meaning that each step of generation isn't influenced by the total length of the context. The exception is the attention mechanism, which doesn’t scale as efficiently with the length of the context.

---

## Project Objective
Our goal is to investigate and integrate advanced acceleration methods for LLM decoding. We plan to evaluate these methods using GPU benchmarks, drawing insights from several key papers:

1. **[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135.pdf)**
2. **[FlashAttention-2: Faster Attention with Enhanced Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691.pdf)**
3. **[FlashDecoding++: Accelerated Large Language Model Inference on GPUs](https://arxiv.org/pdf/2311.01282.pdf)**
4. **[PagedAttention: Optimized Memory Management in LLM Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf)**

Our ultimate goal is to synthesize the best features from these studies to create an innovative, more efficient decoding algorithm.

---

## LLM Inference Pipeline
The conventional LLM decoding algorithm depends significantly on the self-attention mechanism, which, while effective, is a primary source of computational inefficiency in LLMs. The overall LLM inference pipeleine is shown below:

![LLM Inference Pipeline](media/llm_inferece_dataflow.png) (Image Source: [FlashDecoding++](https://arxiv.org/pdf/2311.01282.pdf))

The prefill phase mainly involves the GEMM operation, while the decode phase mainly
involves the GEMV/Flat GEMM operation. 

From the figure, we can also observer that there are mainly three parts in the LLM inference pipeline:
* **Prefill**: The prefill phase, which is the attention part, is responsible to generate the first token and the KV cache for the users' input query.
* **KV Cache**: The KV cache is responsible for storing the key and value matrices, which are used to avoid redundant computation in the decode phase.
* **Decode**: The decode phase, which is the another attention part, is responsible to generate the next token and update the KV cache for the users' input query.

Therefore, in order to optimize the LLM inference pipeline, we can try to optimize the three parts respectively.

Before diving into the optimization, we need to first dive into the details of the attnetion mechanism and the KV cache.

---

### Attention Mechanism

The attention mechanism is crucial in neural networks, especially for natural language processing. It dynamically concentrates on certain parts of the input sequence to produce each part of the output, similar to selective human attention.

Here's a basic formula for self-attention calculation:

![Self-Attention Mechanism](media/attn_equation.png)
    
where `Q`, `K`, and `V` are query, key, and value matrices, respectively. The attention score is calculated by multiplying the query and key matrices, and the weighted sum is calculated by multiplying the attention score and the value matrix.

---

### KV Cache 
[explain the kv cache]

---

After understanding the attention mechanism and the KV cache, we can now dive into the optimization of each steps in the LLM inference pipeline.


## Prefill Optimization

### FlashAttention-1
FlashAttention-1 introduces an IO-aware optimization to the attention mechanism. It significantly reduces memory usage and enhances processing speed by optimizing data flow between memory and processing units.

### FlashAttention-2
Building upon FlashAttention-1, this version further improves parallel processing and workload partitioning. It aims to maximize GPU utilization and reduce latency in LLM decoding.

## KV Cache Optimization
### PagedAttention
This is an attention algorithm inspired by virtual memory and paging techniques in operating systems. This approach divides a request's key-value (KV) cache into blocks, each containing attention keys and values for a fixed number of tokens. Unlike traditional methods, these blocks are not stored in contiguous space, allowing for more flexible memory management similar to the operating system's virtual memory. By avoiding contiguous space caching, we reduce both internal and external fragmentation in GPU memory. This leads to more efficient memory utilization, enabling the handling of larger batch sizes and consequently achieving higher throughput.

## FlashDecoding
FlashDecoding focuses on optimizing the entire LLM decoding pipeline for GPUs. It restructures traditional decoding algorithms to better suit the parallel nature of GPU architectures.

The images below shows the difference between the flash attention and the flash decoding.

![Flash Attention](media/flash_attn.gif) (Image Source: [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/))
![Flash Decoding](media/flash_decoding_demo.gif) (Image Source: [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/))

### FlashDecoding++
FlashDecoding++ is an advanced version that further refines GPU utilization strategies. It introduces novel techniques for managing large-scale language model inference, thereby boosting decoding speeds significantly.

## Performance Evaluation 

### Throughput

### Perplexity

### Memory Usage
# PagedAttention Algorithm
This is an attention algorithm inspired by virtual memory and paging techniques in operating systems. This approach divides a request's key-value (KV) cache into blocks, each containing attention keys and values for a fixed number of tokens. Unlike traditional methods, these blocks are not stored in contiguous space, allowing for more flexible memory management similar to the operating system's virtual memory. By avoiding contiguous space caching, we reduce both internal and external fragmentation in GPU memory. This leads to more efficient memory utilization, enabling the handling of larger batch sizes and consequently achieving higher throughput.

---
