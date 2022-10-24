import pandas as pd
import numpy as np



def find_all_index(sequence_folder):
    sync_path = sequence_folder + 'time_stamps_synced.html'
    df_sync = pd.read_html(sync_path)[0]
    converters = {c: lambda x: str(x) for c in df_sync.columns}
    df_sync = pd.read_html(sync_path, converters=converters)[0]


    size_c = df_sync.shape[1] - 1
    index_all = []

    for num_index in range(size_c):
        if (num_index < 10):
            num_index_str = '00000' + str(num_index)

        elif (num_index >= 100):
            num_index_str = '000' + str(num_index)
        else:
            num_index_str = '0000' + str(num_index)
        flag = 1
        for i in range(17,23):
            if (df_sync[num_index_str][i] != df_sync[num_index_str][i]):
                flag = 0

        if(flag == 1):
            index_all.append(num_index)

    return np.array(index_all)




sequence_folder = './data/20220520_161851/'

index_all = find_all_index(sequence_folder)



# if (df_sync['000001'][16] != df_sync['000001'][16]):
#     print('yes nan')
#
# print(df_sync['000001'][17])