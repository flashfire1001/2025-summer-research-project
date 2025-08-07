some key parameters : noisy pointless

trial 1:

tdim =40
ydim =40
channel = [32, 64, 128]
batch_size=128
epoch = 2000
eta = 0.1
eta = 0.1
lr = 10e-4

trial 2: some improve outline is correct

tdim =40
ydim =40
channel = [32, 64, 128]
batch_size=128
epoch = 5000
eta = 0.1
lr = 10e-4


trial 3:

tdim =64
ydim =64
channel = [128, 256, 512]
batch_size=128
epoch = 1000
eta = 0.2 #label_drop_prob
lr = 1e-4

trail 4:
try:
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
/# Update learning rate at end of epoch
scheduler.step()



trail5:

some updates!

ema for sampling(generation process)

use 2 scheduler for better Lr

enlarge the num_workloader for speed.

tdim =64
ydim =64
channel = [128, 256, 256,256]
batch_size=128
epoch = 10000
eta = 0.2 #label_drop_prob
lr = 1e-4

