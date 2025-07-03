# Research Proposal: Learning to Optimize GPU Kernels through Profiler-Guided Reinforcement Learning

## Abstract

We propose a novel approach to automated GPU kernel optimization that fundamentally reimagines the task as a diagnostic and debugging process rather than pure code generation. By integrating NVIDIA Nsight Compute (NCU) profiler output directly into the reinforcement learning environment and providing access to modern GPU documentation through filesystem tools, we enable AI agents to learn the same iterative optimization workflow used by human experts. Our approach addresses critical limitations in existing methods that rely solely on binary performance feedback, potentially achieving both superior performance and explainable optimization strategies.

## 1. Introduction

### 1.1 Current State of Automated Kernel Engineering

Recent advances in automated kernel engineering have shown promising results:
- **METR's KernelAgent** achieves 1.8x average speedup using parallel tree search
- **Cognition AI's Kevin-32B** demonstrates 65% correctness with multi-turn RL
- **Sakana AI's CUDA Engineer** claims 10-100x improvements through evolutionary optimization

However, all existing approaches share a fundamental limitation: they operate with minimal feedback, essentially conducting blind search in the space of possible kernels.

### 1.2 The Insight

Expert GPU programmers don't randomly modify code hoping for improvements. They:
1. Profile the kernel to identify specific bottlenecks
2. Consult documentation for optimization strategies
3. Apply targeted fixes based on profiler insights
4. Iterate until performance goals are met

We propose teaching AI agents this exact workflow through reinforcement learning.

## 2. Technical Approach

### 2.1 Core Innovation: Profiler-Guided RL

Our key innovation is integrating rich profiler feedback directly into the RL environment:

```python
# Traditional approach
State: PyTorch code → Action: Generate CUDA → Reward: Is it faster? (binary)

# Our approach  
State: PyTorch code + NCU output + Docs → Action: Generate CUDA → Reward: Multi-factor
                                      ↑                                    (speedup + bottleneck reduction)
                                      |
                          Rich diagnostic information:
                          - Memory bandwidth utilization
                          - Occupancy metrics  
                          - Stall reasons
                          - Cache hit rates
```

_I assume we're going to have to be iterative, kinda like in prior work_

### 2.2 Environment Design: "Poor Man's RAG"

Rather than building complex retrieval systems, we provide documentation and profiler output as text files within the RL environment:

```
/workspace/
├── kernel.cu          # Current kernel attempt
├── ncu_output.txt     # Latest profiler results
├── blackwell_manual.txt    # GPU architecture docs
└── optimization_guide.txt  # CUDA best practices
```

The agent has access to standard Unix tools (ripgrep, grep, head, tail) to search and analyze these files, learning through RL when and how to consult different information sources.

_I am open to better ideas for this if need be._

### 2.3 Model Architecture

We propose using DeepSeek-R1-0528-Qwen3-8B with unsloth as the base model due to its feasibility
to fine-tune on a single 5090 and fairly strong reasoning capabilities.

_Do you think this is a good idea?_

### 2.4 Training Strategy

I had an AI make the following suggestions:
```
1. **Curriculum Learning**: Start with simple bottlenecks (e.g., uncoalesced memory access) before complex scenarios
2. **Reward Shaping**: 
   - Primary: Kernel speedup
   - Secondary: Successful bottleneck identification
   - Penalty: Excessive documentation searches
3. **Demonstration Augmentation**: Seed training with expert optimization trajectories
```

I don't know if this is a good idea, mostly because r1 0528 probably has a lot of this already?
We could have some distillation from the CoT of like claude 4 opus if we really want to,
but prior work just seems to let it rip? I feel like we should just let it rip and see what happens.

_Let me know what you think?_

## 3. Advantages Over Existing Approaches (ai wrote this)

| Aspect | Current SOTA | Our Approach |
|--------|--------------|--------------|
| **Feedback Granularity** | Binary (faster/slower) | Detailed performance metrics |
| **Optimization Strategy** | Black box search | Systematic bottleneck resolution |
| **Explainability** | None | Full reasoning traces |
| **Architecture Awareness** | Limited | Learns from architecture docs |
| **Debugging Capability** | Generate & pray | Diagnose & fix |

## 4. Evaluation Plan

### 4.1 Benchmarks
- KernelBench (250 problems across 4 difficulty levels)

### 4.2 Metrics
- **Performance**: Standard speedup metrics (mostly speedup over pytorch / baseline)

Supplemental metrics that an AI suggested:
- **Diagnostic Accuracy**: Does the agent correctly identify bottlenecks?
- **Tool Utilization**: Efficiency of documentation/profiler consultation
- **Explanation Quality**: Human evaluation of optimization rationales

## 5. Implementation Timeline

| Week | Milestone |
|------|-----------|
| 1-2 | Prototype with PyTorch profiler (simpler output format) |
| 3-4 | Integrate basic NCU metrics (memory bandwidth only) |
| 5-6 | Full NCU integration with all performance counters |
| 7-8 | Add Blackwell documentation and modern GPU features |
| 9-10 | RL training and hyperparameter tuning |
| 11-12 | Evaluation and paper writing |

## 6. Expected Contributions

1. **First profiler-guided RL approach** to GPU kernel optimization
2. **Demonstration that diagnostic understanding** outperforms blind search
3. **Explainable optimization strategies** through reasoning traces
4. **Open-source framework** for profiler-integrated kernel optimization
5. **Insights into emergent tool use** in technical domains

## 7. Risks and Mitigations

### 7.1 Technical Risks
- **NCU Output Variability**: Different metrics across GPU generations
  - *Mitigation*: Multi-architecture training, metric normalization, pairing it with a trace of the system and driver version etc.

- **Training Efficiency**: Compilation and profiling are slow
  - *Mitigation*: Caching! Every result should be cached. Maybe parallel environments?

- **Reward Hacking**: Agent might game the metrics
  - *Mitigation*: Careful reward design, held-out validation, and likely some iteration

### 7.2 Research Risks
- **Baseline Improvements**: Existing methods might improve during our research
  - *Mitigation*: Focus on orthogonal contribution (diagnostic approach)

## 10. Success Criteria

The project succeeds if we demonstrate any of the following above baseline:
1. **Superior performance** to Kevin-32B on KernelBench (>1.5x speedup)
2. **Explainable optimizations** that match human expert strategies
3. **Generalization** to unseen kernel patterns
4. **Efficient tool use** (agent learns when to consult profiler/docs)

## References

1. KernelBench: Can LLMs Write Efficient GPU Kernels? (Stanford, 2025)
2. Measuring Automated Kernel Engineering (METR, 2025)
3. Kevin-32B: Multi-Turn RL for Writing CUDA Kernels (Cognition AI, 2025)
4. The AI CUDA Engineer (Sakana AI, 2025)
5. DeepSeek-R1: Advancing Reasoning in Open-Source LLMs (DeepSeek, 2025)

## Appendix A: Preliminary Results

Initial experiments with PyTorch profiler show promising patterns:
- Agent learns to check "Self CUDA %" to identify bottleneck operations
- Successfully identifies memory-bound vs compute-bound kernels
- Develops consistent search patterns in documentation

## Appendix B: Example Optimization Trace

```
Agent: Let me check the profiler output for bottlenecks.
> rg "Metric.*%" ncu_output.txt | head -5
Memory Throughput: 78.3% of peak
SM Occupancy: 33.4%
L2 Cache Hit Rate: 12.1%

Agent: Low L2 cache hit rate suggests poor data locality. Searching for tiling strategies.
> rg -A 10 "shared memory tiling" optimization_guide.txt
[Retrieves section on tile-based optimization]

Agent: Implementing 32x32 tiled matrix multiplication with shared memory...
```

This trace demonstrates the diagnostic reasoning we aim to develop through RL training.

Steps completed:
 - Integrated Litellm to the codebase (easier to call)
 - Validated it on deepseek 0528-qwen3-8b (single run)

Todo:
 - Finish running 5090 baseline sweep (timing folder): did this by running `python3 scripts/generate_baseline_time.py`
 - Run r1 0528-qwen3-8b on a full set of problems (fp8 and nvfp4) EDIT: can't figure out how to nvfp4, lets just do vanilla unsloth for now
 - Run r1 0528 on (full set? subset? decide) to find out what the headroom looks like
 - Plot the different baseline times (timing folder)
 - (Assuming 0528-qwen3-8b looks similar to r1 0528, then) setup the training loop
 - Create train/test splits
 - Run the training loop
 - Evaluate the results
 - Write the paper
 - Write the blog post