import pandas as pd


def load_data_csv(path):
    data = pd.read_csv(path)

    ll = data.values.tolist()
    for i3 in range(len(ll)):
        split_string = ll[i3][0].split("/", 2)
        mm = split_string[0]
        dd = split_string[1]
        yy = split_string[2]
        ff_strr = yy + "-" + mm + "-" + dd
        ll[i3][0] = ff_strr

    from pandas import DataFrame

    data_tmp = DataFrame(ll, columns=["DateTime","Open","High","Low","Close","Volume","Range","RangeP","zscore","z1cat"])
    data_tmp = data_tmp.set_index('DateTime')
    data_tmp.index = pd.to_datetime(data_tmp.index)

    return data_tmp