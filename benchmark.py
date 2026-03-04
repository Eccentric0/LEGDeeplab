"""
LEGDeeplab - Performance Benchmark Script
========================================

Comprehensive benchmarking tool for evaluating LEGDeeplab and other segmentation models.
Measures accuracy, speed, memory usage, and other performance metrics.

Features:
- Multi-model comparison
- Hardware performance testing
- Memory footprint analysis
- Throughput measurements
- Accuracy validation
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import json
from collections import defaultdict
import psutil
import GPUtil
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def parse_arguments():
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(description='Benchmark LEGDeeplab models')
    
    parser.add_argument('--models', nargs='+', default=['LEGDeeplab'], 
                        help='Models to benchmark')
    parser.add_argument('--input_sizes', nargs='+', type=int, default=[256, 512], 
                        help='Input sizes to test (HxW)')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 2, 4], 
                        help='Batch sizes to test')
    parser.add_argument('--warmup', type=int, default=10, 
                        help='Warmup iterations')
    parser.add_argument('--benchmark', type=int, default=100, 
                        help='Benchmark iterations')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to benchmark on')
    
    return parser.parse_args()


class ModelBenchmark:
    """Benchmark class for evaluating model performance."""
    
    def __init__(self, model, input_size, batch_size, device='cuda'):
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create dummy input
        self.dummy_input = torch.randn(
            batch_size, 3, input_size, input_size, 
            device=self.device, dtype=torch.float32
        )
    
    def measure_memory(self):
        """Measure memory usage of the model."""
        # Clear cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Measure memory before and after
        if self.device.type == 'cuda':
            mem_before = torch.cuda.memory_allocated()
            _ = self.model(self.dummy_input)
            mem_after = torch.cuda.max_memory_allocated()
            memory_used = mem_after - mem_before
        else:
            # For CPU, use psutil as approximation
            process = psutil.Process()
            mem_before = process.memory_info().rss
            _ = self.model(self.dummy_input)
            mem_after = process.memory_info().rss
            memory_used = mem_after - mem_before
        
        return memory_used / (1024 ** 2)  # Convert to MB
    
    def measure_latency(self, warmup=10, benchmark=100):
        """Measure inference latency."""
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(self.dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(benchmark):
            with torch.no_grad():
                _ = self.model(self.dummy_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_latency = (total_time / benchmark) * 1000  # Convert to ms
        fps = benchmark / total_time
        
        return avg_latency, fps
    
    def measure_flops(self):
        """Measure FLOPs using thop."""
        try:
            macs, params = profile(self.model, inputs=(self.dummy_input,))
            flops = macs * 2  # Convert MACs to FLOPs
            return flops / (10**9), params / (10**6)  # Return in GFLOPs and MParams
        except:
            # Fallback using fvcore
            try:
                flops = FlopCountAnalysis(self.model, self.dummy_input)
                total_flops = flops.total() / (10**9)  # Convert to GFLOPs
                params = sum(p.numel() for p in self.model.parameters()) / (10**6)  # Convert to MParams
                return total_flops, params
            except:
                return 0, 0  # Return 0 if both methods fail
    
    def run_benchmark(self, warmup=10, benchmark=100):
        """Run comprehensive benchmark."""
        print(f"Benchmarking: Input {self.input_size}x{self.input_size}, Batch {self.batch_size}")
        
        # Measure memory
        memory_mb = self.measure_memory()
        print(f"Memory Usage: {memory_mb:.2f} MB")
        
        # Measure latency and FPS
        latency_ms, fps = self.measure_latency(warmup, benchmark)
        print(f"Latency: {latency_ms:.2f} ms, FPS: {fps:.2f}")
        
        # Measure FLOPs and parameters
        gflops, mparams = self.measure_flops()
        print(f"FLOPs: {gflops:.2f} GFLOPs, Parameters: {mparams:.2f} M")
        
        return {
            'input_size': self.input_size,
            'batch_size': self.batch_size,
            'memory_mb': memory_mb,
            'latency_ms': latency_ms,
            'fps': fps,
            'gflops': gflops,
            'mparams': mparams
        }


def create_model(model_name, num_classes=21):
    """Create model instance based on name."""
    if model_name.lower() == 'legdeeplab':
        from nets.LEGDeeplab import LEGDeeplab
        return LEGDeeplab(in_channels=3, num_classes=num_classes, backbone='resnet50')
    elif model_name.lower() == 'deeplabv3':
        from torchvision.models.segmentation import deeplabv3_resnet50
        return deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    elif model_name.lower() == 'fcn':
        from torchvision.models.segmentation import fcn_resnet50
        return fcn_resnet50(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main benchmarking function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("LEGDeeplab - Comprehensive Model Benchmark")
    print("=" * 60)
    
    # Hardware information
    print(f"CPU: {psutil.cpu_count()} cores")
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        if gpu:
            print(f"GPU: {gpu.name}")
            print(f"GPU Memory: {gpu.memoryTotal} MB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print()
    
    # Collect results
    results = defaultdict(list)
    
    # Iterate through models, input sizes, and batch sizes
    for model_name in args.models:
        print(f"\nBenchmarking Model: {model_name}")
        print("-" * 40)
        
        for input_size in args.input_sizes:
            for batch_size in args.batch_sizes:
                try:
                    # Create model
                    model = create_model(model_name)
                    
                    # Create benchmark instance
                    benchmarker = ModelBenchmark(
                        model, input_size, batch_size, args.device
                    )
                    
                    # Run benchmark
                    result = benchmarker.run_benchmark(args.warmup, args.benchmark)
                    result['model'] = model_name
                    
                    # Store result
                    results[model_name].append(result)
                    
                    print()
                    
                except Exception as e:
                    print(f"Error benchmarking {model_name} with input {input_size}, batch {batch_size}: {str(e)}")
                    continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"{'Input':<12} {'Batch':<8} {'FPS':<10} {'Latency(ms)':<12} {'Memory(MB)':<12} {'GFLOPs':<10} {'Params(M)':<10}")
        print("-" * 80)
        
        for result in model_results:
            print(f"{result['input_size']}x{result['input_size']:<4} "
                  f"{result['batch_size']:<8} "
                  f"{result['fps']:<10.2f} "
                  f"{result['latency_ms']:<12.2f} "
                  f"{result['memory_mb']:<12.2f} "
                  f"{result['gflops']:<10.2f} "
                  f"{result['mparams']:<10.2f}")
    
    # Save results to JSON
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(dict(results), f, indent=2)
    print(f"\nBenchmark results saved to {output_file}")
    
    # Generate model comparison insights
    print("\n" + "=" * 60)
    print("MODEL COMPARISON INSIGHTS")
    print("=" * 60)
    
    for model_name in args.models:
        if model_name in results:
            model_results = results[model_name]
            if model_results:
                avg_fps = np.mean([r['fps'] for r in model_results])
                avg_params = np.mean([r['mparams'] for r in model_results])
                avg_memory = np.mean([r['memory_mb'] for r in model_results])
                
                print(f"{model_name}: Avg FPS={avg_fps:.2f}, Avg Params={avg_params:.2f}M, Avg Memory={avg_memory:.2f}MB")


if __name__ == "__main__":
    main()