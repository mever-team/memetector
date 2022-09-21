import os
import numpy as np
import pandas as pd
import random


def split(showtxt=True):
    random.seed(0)

    web_tp_dir = "./data/web/text_presence"
    web_ta_dir = "./data/web/text_absence"
    cropdir = "./data/cropped"
    memedir = "./data/meme"

    files_concep_cap_wtxt = [
        (os.path.join(web_tp_dir, x), 0, "CC-wtxt") for x in os.listdir(web_tp_dir)
    ]
    files_concep_cap_wotxt = [
        (os.path.join(web_ta_dir, x), 0, "CC-wotxt") for x in os.listdir(web_ta_dir)
    ]
    files_hateful_memes_regular = [
        (os.path.join(cropdir, x), 0, "HM-cropped") for x in os.listdir(cropdir)
    ]
    files_hateful_memes_regular.sort(key=lambda x: x[0])
    files_hateful_memes_memes = [
        (os.path.join(memedir, x), 1, "HM")
        for x in os.listdir(memedir)
        if f"{cropdir}/cropped_{x}" in [x[0] for x in files_hateful_memes_regular]
    ]
    files_hateful_memes_memes.sort(key=lambda x: x[0])

    n = len(files_hateful_memes_memes)

    ntrain = int(0.85 * n)
    nval = int(0.9 * n)
    ntest = int(1.0 * n)

    files_concep_cap_wtxt_train = files_concep_cap_wtxt[:ntrain]
    files_concep_cap_wtxt_val = files_concep_cap_wtxt[ntrain:nval]
    files_concep_cap_wtxt_test = files_concep_cap_wtxt[nval:ntest]

    files_concep_cap_wotxt_train = files_concep_cap_wotxt[:ntrain]
    files_concep_cap_wotxt_val = files_concep_cap_wotxt[ntrain:nval]
    files_concep_cap_wotxt_test = files_concep_cap_wotxt[nval:ntest]

    files_hateful_memes_regular_train = files_hateful_memes_regular[:ntrain]
    files_hateful_memes_regular_val = files_hateful_memes_regular[ntrain:nval]
    files_hateful_memes_regular_test = files_hateful_memes_regular[nval:ntest]

    files_hateful_memes_memes_train = files_hateful_memes_memes[:ntrain]
    files_hateful_memes_memes_val = files_hateful_memes_memes[ntrain:nval]
    files_hateful_memes_memes_test = files_hateful_memes_memes[nval:ntest]

    train = {}
    val = {}
    test = {}

    p_conc = 0.0
    p_txt = 0.0
    files_train = files_hateful_memes_regular_train + files_hateful_memes_memes_train
    files_val = files_hateful_memes_regular_val + files_hateful_memes_memes_val
    files_test = files_hateful_memes_regular_test + files_hateful_memes_memes_test

    train[f"ConcepCap{p_conc: 1.2f} TXT{p_txt: 1.2f}"] = files_train
    val[f"ConcepCap{p_conc: 1.2f} TXT{p_txt: 1.2f}"] = files_val
    test[f"ConcepCap{p_conc: 1.2f} TXT{p_txt: 1.2f}"] = files_test

    for p_conc in [1 / 3, 2 / 3, 3 / 3]:
        for p_txt in [0, 1 / 3, 2 / 3, 3 / 3]:
            n_cc_wtxt_train = int(ntrain * p_conc * p_txt)
            n_cc_wotxt_train = int(ntrain * p_conc * (1 - p_txt))
            n_cropped_train = int(ntrain * (1 - p_conc))
            diff = ntrain - (n_cc_wtxt_train + n_cc_wotxt_train + n_cropped_train)
            if n_cc_wtxt_train != 0:
                n_cc_wtxt_train += diff
            elif n_cc_wotxt_train != 0:
                n_cc_wotxt_train += diff
            elif n_cropped_train != 0:
                n_cropped_train += diff
            files_train = (
                files_concep_cap_wtxt_train[:n_cc_wtxt_train]
                + files_concep_cap_wotxt_train[:n_cc_wotxt_train]
                + files_hateful_memes_regular_train[:n_cropped_train]
                + files_hateful_memes_memes_train
            )
            random.shuffle(files_train)

            n_cc_wtxt_val = int((nval - ntrain) * p_conc * p_txt)
            n_cc_wotxt_val = int((nval - ntrain) * p_conc * (1 - p_txt))
            n_cropped_val = int((nval - ntrain) * (1 - p_conc))
            diff = nval - ntrain - (n_cc_wtxt_val + n_cc_wotxt_val + n_cropped_val)
            if n_cc_wtxt_val != 0:
                n_cc_wtxt_val += diff
            elif n_cc_wotxt_val != 0:
                n_cc_wotxt_val += diff
            elif n_cropped_val != 0:
                n_cropped_val += diff
            files_val = (
                files_concep_cap_wtxt_val[:n_cc_wtxt_val]
                + files_concep_cap_wotxt_val[:n_cc_wotxt_val]
                + files_hateful_memes_regular_val[:n_cropped_val]
                + files_hateful_memes_memes_val
            )
            random.shuffle(files_val)

            n_cc_wtxt_test = int((ntest - nval) * p_conc * p_txt)
            n_cc_wotxt_test = int((ntest - nval) * p_conc * (1 - p_txt))
            n_cropped_test = int((ntest - nval) * (1 - p_conc))
            diff = ntest - nval - (n_cc_wtxt_test + n_cc_wotxt_test + n_cropped_test)
            if n_cc_wtxt_test != 0:
                n_cc_wtxt_test += diff
            elif n_cc_wotxt_test != 0:
                n_cc_wotxt_test += diff
            elif n_cropped_test != 0:
                n_cropped_test += diff
            files_test = (
                files_concep_cap_wtxt_test[:n_cc_wtxt_test]
                + files_concep_cap_wotxt_test[:n_cc_wotxt_test]
                + files_hateful_memes_regular_test[:n_cropped_test]
                + files_hateful_memes_memes_test
            )
            random.shuffle(files_test)

            train[f"ConcepCap{p_conc: 1.2f} TXT{p_txt: 1.2f}"] = files_train
            val[f"ConcepCap{p_conc: 1.2f} TXT{p_txt: 1.2f}"] = files_val
            test[f"ConcepCap{p_conc: 1.2f} TXT{p_txt: 1.2f}"] = files_test

    if showtxt:
        scenarios = list(train.keys())
        source = ["CC-wtxt", "CC-wotxt", "HM-cropped", "HM"]
        for scenario in scenarios:
            counts = []
            nmeme = 0
            nreg = 0
            for s in source:
                traincount = len([x for x in train[scenario] if x[2] == s])
                valcount = len([x for x in val[scenario] if x[2] == s])
                testcount = len([x for x in test[scenario] if x[2] == s])
                total = traincount + valcount + testcount
                counts.append([traincount, valcount, testcount, total])
            counts.append(np.sum(counts, axis=0).tolist())
            counts = pd.DataFrame(
                counts,
                index=source + ["total"],
                columns=["training", "validation", "test", "total"],
            )

            nmeme += (
                len([x for x in train[scenario] if x[1] == 1])
                + len([x for x in val[scenario] if x[1] == 1])
                + len([x for x in test[scenario] if x[1] == 1])
            )
            nreg += (
                len([x for x in train[scenario] if x[1] == 0])
                + len([x for x in val[scenario] if x[1] == 0])
                + len([x for x in test[scenario] if x[1] == 0])
            )

            print(scenario)
            print(counts)
            print(f"meme: {nmeme}, regular: {nreg}")
            print()

    return train, val, test
