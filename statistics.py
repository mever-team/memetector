from split import split
from PIL import Image
import numpy as np

TRAIN, _, _ = split()
scenarios = list(TRAIN.keys())

mean = {}
var = {}

for scenario in scenarios:
    count_x = 0
    count_y = 0
    count_z = 0
    sum_x = 0
    sum_y = 0
    sum_z = 0
    sum_x2 = 0
    sum_y2 = 0
    sum_z2 = 0
    for x in TRAIN[scenario]:
        img = np.array(Image.open(x[0])) / 255

        count_x += img[:, :, 0].reshape(-1).shape[0]
        sum_x += np.sum(img[:, :, 0])
        sum_x2 += np.sum(img[:, :, 0] ** 2)
        mean_x = sum_x / count_x
        var_x = sum_x2 / count_x - mean_x**2

        count_y += img[:, :, 1].reshape(-1).shape[0]
        sum_y += np.sum(img[:, :, 1])
        sum_y2 += np.sum(img[:, :, 1] ** 2)
        mean_y = sum_y / count_y
        var_y = sum_y2 / count_y - mean_y**2

        count_z += img[:, :, 2].reshape(-1).shape[0]
        sum_z += np.sum(img[:, :, 2])
        sum_z2 += np.sum(img[:, :, 2] ** 2)
        mean_z = sum_z / count_z
        var_z = sum_z2 / count_z - mean_z**2

    mean[scenario] = np.array([mean_x, mean_y, mean_z])
    var[scenario] = np.array([var_x, var_y, var_z])

    print(scenario)
    print(
        f"mean: [{mean_x:1.3f},{mean_y:1.3f},{mean_z:1.3f}], var: [{var_x:1.3f},{var_y:1.3f},{var_z:1.3f}]"
    )
    print()

# ConcepCap 0.00 TXT 0.00
# mean: [0.508,0.474,0.451], var: [0.104,0.101,0.104]
#
# ConcepCap 0.33 TXT 0.00
# mean: [0.502,0.475,0.448], var: [0.099,0.096,0.101]
#
# ConcepCap 0.33 TXT 0.33
# mean: [0.504,0.475,0.449], var: [0.101,0.098,0.103]
#
# ConcepCap 0.33 TXT 0.67
# mean: [0.509,0.479,0.453], var: [0.103,0.099,0.104]
#
# ConcepCap 0.33 TXT 1.00
# mean: [0.512,0.482,0.457], var: [0.106,0.102,0.106]
#
# ConcepCap 0.67 TXT 0.00
# mean: [0.500,0.478,0.450], var: [0.095,0.092,0.099]
#
# ConcepCap 0.67 TXT 0.33
# mean: [0.504,0.480,0.453], var: [0.099,0.095,0.101]
#
# ConcepCap 0.67 TXT 0.67
# mean: [0.508,0.481,0.454], var: [0.103,0.099,0.105]
#
# ConcepCap 0.67 TXT 1.00
# mean: [0.513,0.485,0.461], var: [0.107,0.103,0.108]
#
# ConcepCap 1.00 TXT 0.00
# mean: [0.496,0.475,0.443], var: [0.092,0.089,0.098]
#
# ConcepCap 1.00 TXT 0.33
# mean: [0.504,0.482,0.454], var: [0.097,0.094,0.101]
#
# ConcepCap 1.00 TXT 0.67
# mean: [0.508,0.483,0.457], var: [0.103,0.099,0.105]
#
# ConcepCap 1.00 TXT 1.00
# mean: [0.514,0.487,0.463], var: [0.107,0.103,0.108]
