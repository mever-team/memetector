import numpy as np
import os

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
resdir = f'{drive}PycharmProjects/MediaVerse/MemeDetection/TextDetection/'
files = np.array([x for x in os.listdir(resdir) if 'accuracy' in x and 'old' not in x])[[0, 2, 6, 4, 5]]

with open(f'{resdir}accuracy VGG16.npy', 'rb') as f:
    accuracy_vgg = np.load(f)

with open(f'{resdir}accuracy ResNet50.npy', 'rb') as f:
    accuracy_resnet = np.load(f)

with open(f'{resdir}accuracy EfficientNetB5.npy', 'rb') as f:
    accuracy_effnet = np.load(f)

with open(f'{resdir}accuracy ViT.npy', 'rb') as f:
    accuracy_vit = np.load(f)

with open(f'{resdir}accuracy ViT trainable vector concatenate attention.npy', 'rb') as f:
    accuracy_vit_att_conc = np.load(f)

for file in files:
    with open(f'{resdir}{file}', 'rb') as f:
        accuracy = np.load(f)
        print(file)
        print(np.round(accuracy * 100, 1))
        print('argmax across train scenarios:', np.argmax(accuracy, axis=0))
        print()

n = accuracy_vgg.shape[0] * accuracy_vgg.shape[1]

print(f'Average accuracy:\nVGG16: {np.mean(accuracy_vgg) * 100:1.2f}%\nResNet50: {np.mean(accuracy_resnet) * 100:1.2f}%\nEfficientNetB5: {np.mean(accuracy_effnet) * 100:1.2f}%\nViT: {np.mean(accuracy_vit) * 100:1.2f}%\nViT Att.: {np.mean(accuracy_vit_att_conc) * 100:1.2f}%')
print(f'\nScenarios fraction with greater accuracy (ViT Att.):')
print(f'VGG16: {np.sum(accuracy_vit_att_conc > accuracy_vgg)}/{n}={np.sum(accuracy_vit_att_conc > accuracy_vgg) / n * 100:1.2f}%. Average difference: {(np.mean(accuracy_vit_att_conc) - np.mean(accuracy_vgg)) * 100: 1.2f}%')
print(f'ResNet50: {np.sum(accuracy_vit_att_conc > accuracy_resnet)}/{n}={np.sum(accuracy_vit_att_conc > accuracy_resnet) / n * 100:1.2f}%. Average difference: {(np.mean(accuracy_vit_att_conc) - np.mean(accuracy_resnet)) * 100: 1.2f}%')
print(f'EfficientNetB5: {np.sum(accuracy_vit_att_conc > accuracy_effnet)}/{n}={np.sum(accuracy_vit_att_conc > accuracy_effnet) / n * 100:1.2f}%. Average difference: {(np.mean(accuracy_vit_att_conc) - np.mean(accuracy_effnet)) * 100: 1.2f}%')
print(f'ViT: {np.sum(accuracy_vit_att_conc > accuracy_vit)}/{n}={np.sum(accuracy_vit_att_conc > accuracy_vit) / n * 100:1.2f}%. Average difference: {(np.mean(accuracy_vit_att_conc) - np.mean(accuracy_vit)) * 100: 1.2f}%')
print('\nVPU comparison on test set scenarios w/o VPU:')
print(f'VGG16 w/o VPU:\t\t', np.max(accuracy_vgg[9:, 9:], axis=0) * 100, np.mean(accuracy_vgg[9:, 9:]) * 100)
print(f'ResNet50 w/o VPU:\t\t', np.max(accuracy_resnet[9:, 9:], axis=0) * 100, np.mean(accuracy_resnet[9:, 9:]) * 100)
print(f'EfficientNetB5 w/o VPU:\t', np.max(accuracy_effnet[9:, 9:], axis=0) * 100, np.mean(accuracy_effnet[9:, 9:]) * 100)
print(f'ViT w/o VPU:\t\t\t', np.max(accuracy_vit[9:, 9:], axis=0) * 100, np.mean(accuracy_vit[9:, 9:]) * 100)
print(f'ViT Att. w/o VPU:\t\t\t', np.max(accuracy_vit_att_conc[9:, 9:], axis=0) * 100, np.mean(accuracy_vit_att_conc[9:, 9:]) * 100)
print(f'ViT Att. w/ VPU:\t\t', accuracy_vit_att_conc[8, -4:] * 100, np.mean(accuracy_vit_att_conc[8, -4:]) * 100)
