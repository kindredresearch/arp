from arp.arp import ARProcess
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 12))
for i, p in enumerate([1, 2, 3, 5]):
    for j, alpha in enumerate([0.0, 0.5, 0.8, 0.9]):
        plt.subplot(4, 4, 4 * i + j + 1)
        arp = ARProcess(p, alpha)
        seeds = [12345, 23456, 34567]
        for s, seed in enumerate(seeds):
            noise = []
            arp.reset(seed)
            # # warm up
            # for _ in range(1000):
            #     arp.step()
            for _ in range(100):
                noise.append(arp.step())
            if s == 0:
                plt.plot(noise)
            else:
                plt.plot(noise, alpha=0.5)
        plt.title("p = {}, alpha = {}".format(p, alpha))
plt.tight_layout()
plt.show(True)