import matplotlib.pyplot as plt
import numpy as np


def main():

    train_loss = [13.934168990670818, 10.127379962044401, 8.831391263407296, 7.9122066837467555, 7.285163250143669, 6.620031223558399, 6.0665657670530555, 5.287704917321285, 4.802964104546441, 4.815188108078421, 4.490151693904418, 3.7780739221761577, 3.6315624419956993, 3.188915561141852, 3.532238297266503, 3.3836777145641217, 3.4592553032043316, 3.270596426262703, 3.1252840362183036, 3.0779884532162045, 2.730588551634523, 2.237290512154636, 3.27383238118896, 2.3831295889443638]
    train_loss2 = [672.4512967601759, 448.6434518288622, 359.09387406301283, 271.95128952676663, 236.23452488334576, 198.29694297244743, 159.78571258730722, 127.42594054757733, 106.37824130123609, 116.67807622529237, 96.046842839257, 69.72000591446456, 67.17260772496054, 49.18640372974506, 75.45367696172752, 65.68126714988023, 72.55791571107629, 70.95458535552932, 59.95323460954872, 60.13219666909228, 39.22127852911637, 19.808480911603258, 83.4669653115933, 30.82126263918942]

    lentr = len(train_loss)+1  
    """
    plt.plot(np.arange(1,lentr),train_loss,label='L1Loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig("lossPlot.png")
    """
    plt.plot(np.arange(1,lentr),train_loss2,label='MSELoss')
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.legend(loc="upper right", fontsize=16)
    plt.savefig("lossPlot2.png")
    #plt.show()


if __name__ == "__main__":
    main()
