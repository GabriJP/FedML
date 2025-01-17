import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_top_k_labels(data_dir, top_k=5):
    data_path = "Groundtruth/AllLabels"
    label_counts = dict()
    for filename in os.listdir(os.path.join(data_dir, data_path)):
        file = os.path.join(data_dir, data_path, filename)
        if os.path.isfile(file):
            label = file[:-4].split("_")[-1]
            df = pd.read_csv(file)
            df.columns = ['label']
            label_counts[label] = (df[df['label'] == 1].shape[0])
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    selected = [k for (k, v) in label_counts[:top_k]]
    return selected


def get_labeled_data_with_2_party(data_dir, selected_labels, n_samples, dtype="Train"):
    # get labels
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_labels:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        df = pd.read_csv(file, header=None)
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    if len(selected_labels) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels

    # get XA, which are image low level features
    features_path = "Low_Level_Features"
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ")
            df.dropna(axis=1, inplace=True)
            print(f"{file} datasets features {len(df.columns)}")
            dfs.append(df)
    data_xa = pd.concat(dfs, axis=1)
    data_xa_selected = data_xa.loc[selected.index]
    print("XA shape:", data_xa_selected.shape)  # 634 columns

    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t")
    tagsdf.dropna(axis=1, inplace=True)
    data_xb_selected = tagsdf.loc[selected.index]
    print("XB shape:", data_xb_selected.shape)
    if n_samples != -1:
        return data_xa_selected.values[:n_samples], data_xb_selected.values[:n_samples], selected.values[:n_samples]
    else:
        # load all data
        return data_xa_selected.values, data_xb_selected.values, selected.values


def get_labeled_data_with_3_party(data_dir, selected_labels, n_samples, dtype="Train"):
    xa, xb, y = get_labeled_data_with_2_party(data_dir=data_dir, selected_labels=selected_labels, n_samples=n_samples,
                                              dtype=dtype)
    n_tags = xb.shape[1]
    half_n_tags = int(0.5 * n_tags)
    return xa, xb[:, :half_n_tags], xb[:, half_n_tags:], y


def nus_wide_load_two_party_data(data_dir, selected_labels, neg_label=-1, n_samples=-1):
    print("# load_two_party_data")

    xa, xb, y = get_labeled_data_with_2_party(data_dir=data_dir,
                                              selected_labels=selected_labels,
                                              n_samples=n_samples)

    scale_model = StandardScaler()
    xa = scale_model.fit_transform(xa)
    xb = scale_model.fit_transform(xb)

    y_ = list()
    pos_count = 0
    neg_count = 0
    for i in range(y.shape[0]):
        # the first label in y as the first class while the other labels as the second class
        if y[i, 0] == 1:
            y_.append(1)
            pos_count += 1
        else:
            y_.append(neg_label)
            neg_count += 1

    print("pos counts:", pos_count)
    print("neg counts:", neg_count)

    y = np.expand_dims(y_, axis=1)

    print("xa shape:", xa.shape)
    print("xb shape:", xb.shape)
    print("y shape:", y.shape)

    n_train = int(0.8 * xa.shape[0])
    print("# of train samples:", n_train)
    # print("# of test samples:", n_test)

    xa_train, xb_train = xa[:n_train], xb[:n_train]
    xa_test, xb_test = xa[n_train:], xb[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("xa_train.shape:", xa_train.shape)
    print("xb_train.shape:", xb_train.shape)
    print("xa_test.shape:", xa_test.shape)
    print("xb_test.shape:", xb_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)
    return [xa_train, xb_train, y_train], [xa_test, xb_test, y_test]


def nus_wide_load_three_party_data(data_dir, selected_labels, neg_label=-1, n_samples=-1):
    print("# load_three_party_data")
    xa, xb, xc, y = get_labeled_data_with_3_party(data_dir=data_dir, selected_labels=selected_labels,
                                                  n_samples=n_samples)

    scale_model = StandardScaler()
    xa = scale_model.fit_transform(xa)
    xb = scale_model.fit_transform(xb)
    xc = scale_model.fit_transform(xc)

    y_ = list()
    pos_count = 0
    neg_count = 0
    for i in range(y.shape[0]):
        # the first label in y as the first class while the other labels as the second class
        if y[i, 0] == 1:
            y_.append(1)
            pos_count += 1
        else:
            y_.append(neg_label)
            neg_count += 1

    print("pos counts:", pos_count)
    print("neg counts:", neg_count)

    y = np.expand_dims(y_, axis=1)

    n_train = int(0.8 * xa.shape[0])
    xa_train, xb_train, xc_train = xa[:n_train], xb[:n_train], xc[:n_train]
    xa_test, xb_test, xc_test = xa[n_train:], xb[n_train:], xc[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print("xa_train.shape:", xa_train.shape)
    print("xb_train.shape:", xb_train.shape)
    print("xc_train.shape:", xc_train.shape)
    print("Xa_test.shape:", xa_test.shape)
    print("xb_test.shape:", xb_test.shape)
    print("xc_test.shape:", xc_test.shape)
    print("y_train.shape:", y_train.shape)
    print("y_test.shape:", y_test.shape)
    return [xa_train, xb_train, xc_train, y_train], [xa_test, xb_test, xc_test, y_test]


def prepare_party_data(src_data_folder, des_data_folder, selected_labels, neg_label, n_samples, is_three_party=False):
    print("# preparing data ...")

    train_data_list, test_data_list = nus_wide_load_three_party_data(src_data_folder, selected_labels,
                                                                     neg_label=neg_label, n_samples=n_samples) \
        if is_three_party else nus_wide_load_two_party_data(src_data_folder, selected_labels,
                                                            neg_label=neg_label, n_samples=n_samples)

    train_data_file_name_list = ["Xa_train", "Xb_train", "Xc_train", "y_train"] if is_three_party \
        else ["Xa_train", "Xb_train", "y_train"]

    test_data_file_name_list = ["Xa_test", "Xb_test", "Xc_test", "y_test"] if is_three_party \
        else ["Xa_test", "Xb_test", "y_test"]

    for train_data, train_data_name in zip(train_data_list, train_data_file_name_list):
        print(f"{train_data_name} shape: {train_data.shape}")

    for test_data, test_data_name in zip(test_data_list, test_data_file_name_list):
        print(f"{test_data_name} shape: {test_data.shape}")

    ext = "vfl_cnn_lr_00001_async_True_L_33_B_256_R_140_20190820155141_3.csv"
    train_data_full_name_list = [des_data_folder + file_name + ext for file_name in train_data_file_name_list]
    test_data_full_name_list = [des_data_folder + file_name + ext for file_name in test_data_file_name_list]

    for train_data, train_data_full_name in zip(train_data_list, train_data_full_name_list):
        np.savetxt(fname=train_data_full_name, X=train_data, delimiter=',')

    for test_data, test_data_full_name in zip(test_data_list, test_data_full_name_list):
        np.savetxt(fname=test_data_full_name, X=test_data, delimiter=',')

    print("# prepare data finished!")


def get_data_folder_name(sel_lbls, is_three_party):
    folder_name = sel_lbls[0]
    for idx, lbl in enumerate(sel_lbls):
        if idx == 0:
            folder_name = lbl
        else:
            folder_name += "_" + lbl
    appendix = "_three_party" if is_three_party else "_two_party"
    return folder_name + appendix


def load_prepared_parties_data(data_dir, sel_lbls, load_three_party):
    print("# load prepared {0} party data".format("three" if load_three_party else "two"))
    folder_name = get_data_folder_name(sel_lbls, is_three_party=load_three_party)
    print("folder name: {0}".format(folder_name))
    data_folder_full_name = data_dir + folder_name + "/"
    ext = ".csv"
    train_data_name_list = ["Xa_train", "Xb_train", "Xc_train", "y_train"] if load_three_party \
        else ["Xa_train", "Xb_train", "y_train"]
    test_data_name_list = ["Xa_test", "Xb_test", "Xc_test", "y_test"] if load_three_party \
        else ["Xa_test", "Xb_test", "y_test"]
    train_data_path_list = list()
    for train_data_name in train_data_name_list:
        train_data_path = data_folder_full_name + train_data_name + ext
        train_data_path_list.append(train_data_path)
    test_data_path_list = list()
    for test_data_name in test_data_name_list:
        test_data_path = data_folder_full_name + test_data_name + ext
        test_data_path_list.append(test_data_path)

    train_data_list = list()
    for train_data_name, train_data_path in zip(train_data_name_list, train_data_path_list):
        print(f"load {train_data_name}")
        train_data_list.append(np.loadtxt(fname=train_data_path, delimiter=','))

    test_data_list = list()
    for test_data_name, test_data_path in zip(test_data_name_list, test_data_path_list):
        print(f"load {test_data_name}")
        test_data_list.append(np.loadtxt(fname=test_data_path, delimiter=','))

    return train_data_list, test_data_list


if __name__ == '__main__':
    main_data_dir = "../../../data/NUS_WIDE/"

    # sel = get_top_k_labels(data_dir=data_dir, top_k=10)
    # print("sel", sel)
    # ['sky', 'clouds', 'person', 'water', 'animal', 'grass', 'buildings', 'window', 'plants', 'lake']

    # sel_lbls = ['person', 'water', 'animal', 'grass', 'buildings']
    main_sel_lbls = ["person", "animal"]
    # if no prepare three party data, then it is going to prepare two party data
    prepare_three_party = False
    print(f"prepare {'three' if prepare_three_party else 'two'} party data")
    main_folder_name = get_data_folder_name(main_sel_lbls, is_three_party=prepare_three_party)
    folder_full_name = main_data_dir + main_folder_name + "/"
    print(f"folder_full_name:{folder_full_name}")
    if not os.path.exists(folder_full_name):
        os.mkdir(folder_full_name)
    prepare_party_data(src_data_folder=main_data_dir, des_data_folder=folder_full_name, selected_labels=main_sel_lbls,
                       neg_label=0, n_samples=20000, is_three_party=prepare_three_party)
