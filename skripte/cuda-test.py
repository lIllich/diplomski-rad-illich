import torch

def get_gpu_info():
    print(f"PyTorch verzija: {torch.__version__}")
    print(f"CUDA dostupno: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA verzija: {torch.version.cuda}")
        print(f"Broj GPU-ova: {torch.cuda.device_count()}\n")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # u GB
            print(f"GPU {i}: {gpu_name}")
            print(f"  - Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
            print(f"  - Ukupna memorija: {total_memory:.2f} GB")
            print(f"  - Trenutno zauzeće: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB / {total_memory:.2f} GB")
    else:
        print("CUDA nije dostupna. Provjerite instalaciju NVIDIA drajvera i PyTorch s CUDA podrškom.")

if __name__ == "__main__":
    get_gpu_info()