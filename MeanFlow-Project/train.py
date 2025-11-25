from model import MFDiT
import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from meanflow import MeanFlow
from accelerate import Accelerator
import time
import os
from utils import basic_visualize, draw_loss_curve, compute_psnr, should_test
from datasets import get_noised_mnist_dataloader
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

if __name__ == '__main__':
    n_steps = 300000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    os.makedirs('results/mf_img_restore/images', exist_ok=True)
    os.makedirs('results/mf_img_restore/checkpoints', exist_ok=True)
    accelerator = Accelerator(mixed_precision='fp16')

    def cycle(iterable):
        while True:
            for i in iterable:
                yield i

    train_dataloader = get_noised_mnist_dataloader(noise_func_arg=0.3)
    train_dataloader = cycle(train_dataloader)

    # visualize the test data
    test_dataloader = get_noised_mnist_dataloader(noise_func_arg=0.3, train=False)
    for test_data in test_dataloader:
        test_noised_img, test_clean_img, test_label = test_data
        break

    label_to_index = {}
    for i, label in enumerate(test_label):
        l = label.item()
        if l not in label_to_index:
            label_to_index[l] = i
        if len(label_to_index) == 10:
            break

    indices = [label_to_index[i] for i in range(10)]

    test_noised_img = test_noised_img[indices].to(device)
    test_clean_img = test_clean_img[indices].to(device)
    test_label = test_label[indices].to(device)

    test_clean_img_save_path = f"results/mf_img_restore/images/clean_img.png"
    basic_visualize(test_clean_img, savepath=test_clean_img_save_path, title=f"clean image", max_images=10)
    test_noised_img_save_path = f"results/mf_img_restore/images/noised_img.png"
    basic_visualize(test_noised_img, savepath=test_noised_img_save_path, title=f"noised image", max_images=10)







    n_batches = 200
    batch_list_noise = []
    batch_list_clean = []
    batch_list_label = []

    test_dataloader = get_noised_mnist_dataloader(noise_func_arg=0.3, train=False)
    test_iter = iter(test_dataloader)

    for _ in range(n_batches):
        try:
            noised_img, clean_img, label = next(test_iter)
        except StopIteration:
            # 若不够 100 个 batch，可考虑重新迭代
            test_iter = iter(test_dataloader)
            noised_img, clean_img, label = next(test_iter)

        batch_list_noise.append(noised_img)
        batch_list_clean.append(clean_img)
        batch_list_label.append(label)

    # 合并为一个大 tensor
    all_noised = torch.cat(batch_list_noise, dim=0)    # [12800, 1, 32, 32]
    all_clean = torch.cat(batch_list_clean, dim=0)     # [12800, 1, 32, 32]
    all_label = torch.cat(batch_list_label, dim=0)     # [12800]

    # 随机抽 128 张图像
    perm = torch.randperm(all_noised.size(0))[:128]    # 抽出索引
    psnr_test_noise = all_noised[perm].to(device)
    psnr_test_clean = all_clean[perm].to(device)
    psnr_test_label = all_label[perm].to(device)






    model = MFDiT(
        input_size=32,
        patch_size=2,
        in_channels=1,
        dim=144,
        depth=6,
        num_heads=3,
        num_classes=10,
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10000,
        T_mult=2,
        eta_min=1e-6
    )

    meanflow = MeanFlow(channels=1,
                        image_size=32,
                        num_classes=10,
                        flow_ratio=0.50,
                        time_dist=['lognorm', -0.4, 1.0],
                        cfg_ratio=0.10,
                        cfg_scale=2.0,
                        # experimental
                        cfg_uncond='u')

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    losses = 0.0
    mse_losses = 0.0
    loss_history = []

    log_step = 200

    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        for step in pbar:
            data = next(train_dataloader)
            noised_img, clean_img, c = data
            noised_img = noised_img.to(accelerator.device)
            clean_img = clean_img.to(accelerator.device)
            c = c.to(accelerator.device)

            loss, mse_val = meanflow.loss(model, clean_img, noised_img, c)

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()
            loss_history.append(loss.item())
            mse_losses += mse_val.item()

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / log_step:.6f}    MSE_Loss: {mse_losses / log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    psnr_test_restored_img = meanflow.simulate(model, psnr_test_noise.clone(), psnr_test_label.clone(), sample_steps=1)
                    psnr = compute_psnr(psnr_test_clean, psnr_test_restored_img)
                    psnr_info = f'Test-PSNR: {psnr.item()}'

                    log_message = f'{current_time}\n{batch_info}    {loss_info}    {lr_info}    {psnr_info}\n\n'

                    with open('log.txt', mode='a') as n:
                        n.write(log_message)

                    losses = 0.0
                    mse_losses = 0.0
                    model.train()

            if should_test(global_step):
                if accelerator.is_main_process:
                    z = meanflow.simulate(model, test_noised_img.clone(), test_label.clone(), sample_steps=1)
                    img_save_path = f"results/mf_img_restore/images/step_{global_step}.png"
                    basic_visualize(z, savepath=img_save_path, title=f"Step {global_step}, 1 NFE", max_images=10)

                accelerator.wait_for_everyone()
                model.train()
                
    if accelerator.is_main_process:
        ckpt_path = f"results/mf_img_restore/checkpoints/step_{global_step}.pth"
        accelerator.save(model.state_dict(), ckpt_path)
        draw_loss_curve(loss_history, n_steps, batch_size, "mf_img_restore")
