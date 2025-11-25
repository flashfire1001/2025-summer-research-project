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