import matplotlib.pyplot as plt

def Performance_Visualization(history, title):
    fig, axs = plt.subplots(1, 2,figsize=(16,4))
    
    # LOSS
    axs[0].plot(history['loss'], label='train')
    axs[0].plot(history['val_loss'], label='validation')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[0].legend(loc='upper right')

    # ACCURACY
    axs[1].plot(history['acc'], label='train')
    axs[1].plot(history['val_acc'], label='validation')
    axs[1].set(xlabel='epochs', ylabel='metrics')
    axs[1].legend(loc='lower right')

    plt.savefig(title)