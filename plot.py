import numpy as np
import matplotlib.pyplot as plt
import json

def main():
    with open("./output/lora_8_llama/ppl_tracking.json") as f:
        ppl = json.load(f)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(ppl)), ppl)
    plt.xlabel("Training Epochs")
    plt.ylabel("Perplexity")
    plt.title("Learning Curve")
    plt.show()

if __name__ == "__main__":
    main()