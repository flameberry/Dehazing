import matplotlib.pyplot as plt

def display(in_img, out_img, clear_img):
    _, axes = plt.subplots(1, 3)
    axs = axes.flatten()
    axs[0].imshow(in_img)
    axs[0].set_xlabel('Hazy Image')

    axs[1].imshow(out_img)
    axs[1].set_xlabel('Dehazed Image')

    axs[2].imshow(clear_img)
    axs[2].set_xlabel('Clear Image')

    plt.show()
