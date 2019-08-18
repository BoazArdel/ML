import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    results_train_loss = [1.499,1.327,1.211,1.140,1.064,1.001,0.957,0.913,0.890,0.841,0.838,0.808,0.783,0.773,0.753,0.698,0.701,0.688,0.691,0.622]
    results_valid_loss = [1.477,1.391,1.252,1.117,1.184,1.154,1.134,1.156,1.173,1.175,1.216,1.228,1.280,1.352,1.325,1.364,1.431,1.501,1.443,1.559]

    t = range(20)
    plt.interactive(False)
    plt.plot(t, results_train_loss, 'r')  # plotting t, a - normal dist
    plt.plot(t, results_valid_loss, 'b')  # plotting t, b - softmax prob
    red_patch = mpatches.Patch(color='red', label='Train')
    blue_patch = mpatches.Patch(color='blue', label='Validation')
    plt.legend(handles=[red_patch, blue_patch])
    plt.title('Loss for 20 epochs')
    plt.ylabel('Average Loss')
    plt.xlabel('epochs')
    plt.show(block=True)


if __name__ == "__main__":
    main()