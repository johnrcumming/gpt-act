import torch

def main():
    for id in range(torch.cuda.device_count()):
        print('CUDA:{id}: {name}'.format(id=id, name=torch.cuda.get_device_name(id)))

if __name__ == "__main__":
    main()