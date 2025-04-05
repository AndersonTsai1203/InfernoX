# 🔥 FastNet: Lightweight C++ CUDA AI Inference Engine
A blazing-fast C++ AI inference engine powered by CUDA — purpose-built for minimal latency, modular design, and full GPU acceleration.

## 📚 Background & Motivation
The demand for real-time AI inference continues to accelerate across edge devices, robotics, autonomous systems, and high-throughput analytics. While major frameworks like TensorFlow and PyTorch offer extensive training capabilities, they introduce significant overhead during deployment, especially for performance-critical applications.

InfernoX was built from the ground up to explore:

- Low-level AI execution mechanics
- Custom CUDA kernel development
- Memory-efficient GPU inference pipelines

This project demonstrates how a handcrafted C++/CUDA engine can rival traditional frameworks in performance while offering full control over execution, memory, and optimization.

## ❓ Why This Project?
InfernoX is a systems-first, framework-agnostic AI inference engine. It serves as a portfolio-grade project to showcase:
1. Strong grasp of GPU programming and CUDA toolchains
2. Advanced C++17/C++20 design patterns
3. Understanding of deep learning internals
4. Ability to architect modular, high-performance systems
  
If you're aiming to break into accelerated computing, embedded AI, or infrastructure engineering, this is a proof of concept that speaks directly to those domains.

## 🎯 Project Goals
1. Design a modular AI inference engine in C++ with a clear layer abstraction
2. Implement key neural network layers using custom CUDA kernels
3. Build an efficient memory and tensor engine with host-device management
4. Achieve real-time performance on GPU (≤10ms latency per inference)
5. Benchmark against CPU-based inference for performance validation

## 🧱 Project Structure
```text
InfernoX/
├── include/               # Public headers
│   ├── layers/            # Conv2D, ReLU, MaxPool, Dense
│   ├── engine/            # Core inference engine
│   └── utils/             # Tensor, profiler, logger
├── src/                   # Implementations
│   ├── layers/
│   ├── engine/
│   └── main.cpp
├── models/                # Sample models (custom or ONNX)
├── data/                  # Input images / test cases
├── tests/                 # Unit/integration tests
├── benchmarks/            # Performance evaluations
├── CMakeLists.txt
└── README.md
```

## ⚙️ Core Features
- Custom CUDA Kernels for Conv2D, ReLU, and Pooling layers
- Lightweight Tensor Engine for memory allocation and shape management
- Modular Graph Execution Engine with static scheduling
- CLI Inference Pipeline for fast end-to-end testing
- Performance Profiling Tools using cudaEvent_t and nvprof

## 🧪 Project and Environment Setup
### ✅ Prerequisites
- CUDA Toolkit ≥ 11.4
- CMake ≥ 3.15
- GPU with Compute Capability ≥ 6.1
- GCC/G++ ≥ 9.3

### 🧰 Build Instructions
```bash
git clone https://github.com/yourusername/InfernoX.git
cd InfernoX
mkdir build && cd build
cmake ..
make -j$(nproc)
./infernoX --model models/mobilenet.bin --image data/sample.jpg
```

## 🗂 Milestones
| Milestone	| Description |	Status |
| --------- | ----------- | ------ |
| Tensor & Memory Engine | Host/device memory, tensor abstraction |	🔄 In Progress |
| Conv2D & ReLU Kernels | CUDA kernels for basic layers	| ⏳ Planned |
|Inference Engine	| Static layer execution graph | ⏳ Planned |
|Model Loader |	Load binary/ONNX models |	⏳ Planned |
|CLI Inference + Demo	| Command-line interface with image input |	⏳ Planned |
|Benchmarks & Profiling | Run tests and compare to CPU baseline |	⏳ Planned |

### 🚧 Roadmap
#### 🔹 Milestone 1: Foundation Layer – Tensor Engine & CUDA Runtime Abstraction
Establish a robust, reusable C++ interface for handling GPU/CPU tensors and managing CUDA resources.

##### 🔧 Deliverables:
- Tensor class (shape, dtype, memory layout)
- Host ↔ Device memory transfer
- RAII wrappers for CUDA allocations
- Error handling macros for CUDA API
- Logging & profiling utilities (CUDA Events)

#### 🔹 Milestone 2: Custom CUDA Kernels – Conv2D, ReLU, MaxPool
Build performant, handcrafted CUDA kernels for basic CNN layers.

##### 🔧 Deliverables:
- ```conv2d_forward<<<>>>``` (shared memory + tiling optimization)
- ```relu_forward<<<>>>``` (element-wise ops)
- ```maxpool2d_forward<<<>>>``` (windowed kernel with stride)
- Unit tests with synthetic tensors
- Timing benchmarks vs CPU baseline

#### 🔹 Milestone 3: Layer Abstraction & Execution Graph
Create a polymorphic interface to model layers and sequence them in a static computation graph.

##### 🔧 Deliverables:
- ```ILayer``` abstract class
- Concrete layer implementations (Conv2DLayer, ReLULayer, MaxPoolLayer)
- SequentialEngine or GraphExecutor
- Input/output tensor chaining
- Memory reuse where possible

#### 🔹 Milestone 4: Model Loader
Allow external model weights and structure to be loaded dynamically.

##### 🔧 Options:
- Phase 1: Simple custom JSON config + binary weights
- Phase 2 (Optional): ONNX runtime or custom ONNX parser

##### 🔧 Deliverables:
- Model parser
- Weight loader with tensor deserialization
- Layer mapping/instantiation based on config

#### 🔹 Milestone 5: CLI Inference Pipeline
Build an intuitive command-line interface for running models with real input data.

##### 🔧 Deliverables:
- Command-line parser (CLI11 / manual args)
- Image preprocessor (OpenCV or stb_image)
- Inference loop with timer
- Result decoding + print Top-1 class

Example:
```bash
./infernoX --model models/mobilenet.json --weights models/mobilenet.bin --image data/cat.jpg
```

#### 🔹 Milestone 6: Benchmarking & Profiling Suite
Measure inference speed, memory usage, and layer performance on GPU vs CPU.

##### 🔧 Deliverables:
- CUDA events and stream-based profiling
- FPS, latency, per-layer execution time
- Optional: CSV or JSON output for analysis
- Compare with OpenCV DNN or PyTorch C++ baseline

#### 🔹 Milestone 7: Modular Integration & Build Refinement
Polish the codebase to support extensibility and platform portability.

##### 🔧 Deliverables:
- CMake refactor for modular builds
- Header-only components where applicable
- C++17/20 language feature cleanup
- Add unit tests using Catch2 or GoogleTest
- GitHub Actions for CI

#### 🔹 Milestone 8: Optimization & Advanced CUDA Features
Push the engine's performance ceiling.

##### 🔧 Deliverables:
- CUDA Streams for async layer execution
- Kernel fusion (e.g., Conv + ReLU)
- Tiled memory access optimization
- Shared memory & L2 caching efficiency

#### 🔹 Milestone 9: Optional Extensions
Features to differentiate and impress in a portfolio or showcase.

##### 🧩 Options:
- INT8 quantization
- Jetson Xavier/Nano deployment pipeline
- Web UI with WebSocket inference results
- TensorRT backend swappable layer support
- WASM or Rust bindings for edge inference

## 💡 Extension Ideas
- [ ] ONNX Loader Integration with operator mapping
- [ ] Jetson Deployment for edge inference on low-power platforms
- [ ] INT8 Quantization Support for model size & perf
- [ ] Asynchronous Execution using CUDA streams
- [ ] Layer Fusion for reducing kernel launch overhead

## 🚀 Extended Features (Optional)
1. Web-based dashboard to visualize inference pipeline and performance
2. REST API microservice via C++ REST SDK or gRPC
3. Integration with TensorRT as a plug-and-play backend
4. ZeroMQ-based distributed inference cluster prototype

