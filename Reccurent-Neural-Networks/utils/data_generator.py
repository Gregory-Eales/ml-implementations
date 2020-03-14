import torch

def generate_data(n_samples=105, seq_length=5):
    x = torch.zeros(n_samples-seq_length, seq_length)
    y = torch.zeros(n_samples-seq_length, 1)
    sample = torch.sin((torch.tensor(range(n_samples)).float() - (n_samples/2)) / ((n_samples/2)/3.14159))

    for i in range(n_samples-seq_length):
        x[i] = sample[i:i+seq_length]
        y[i] = sample[i+seq_length]

    return x, y, sample

def main():

    x, y = generate_data()
    print(x[0])
    print(y[0])

if __name__ == "__main__":
    main()
