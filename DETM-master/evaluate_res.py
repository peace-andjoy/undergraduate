# %%
import pickle
import numpy as np
import pandas as pd

# %%
result = []

for num_docs in [100, 200]:
    for len_docs in [50, 100]:
        for num_topics in [10, 20]:
            for num_time_slices in [5, 10, 20]:
                train_ppls = []
                val_ppls = []
                test_ppls = []
                tds = []
                css = []
                for sim in range(20):
                    with open('./res/res_M{}_N{}_K{}_T{}_{}.pkl'.format(num_docs, len_docs, num_topics, num_time_slices, sim), 'rb') as file:
                        res = pickle.load(file)
                    train_ppls.append(res['train_ppl'])
                    val_ppls.append(res['val_ppl'])
                    test_ppls.append(res['test_ppl'])
                    tds.append(res['td'])
                    css.append(res['cs'])
                train_ppl_mean, train_ppl_std = np.mean(train_ppls), np.std(train_ppls)
                val_ppl_mean, val_ppl_std = np.mean(val_ppls), np.std(val_ppls)
                test_ppl_mean, test_ppl_std = np.mean(test_ppls), np.std(test_ppls)
                td_mean, td_std = np.mean(tds), np.std(tds)
                cs_mean, cs_std = np.mean(css), np.std(css)

                result.append([num_docs, len_docs, num_topics, num_time_slices, train_ppl_mean, train_ppl_std, val_ppl_mean, val_ppl_std, test_ppl_mean, test_ppl_std, td_mean, td_std, cs_mean, cs_std])

# %%
pd.DataFrame(result, columns=['num_docs', 'len_docs', 'num_topics', 'num_time_slices', 'train_ppl_mean', 'train_ppl_std', 'val_ppl_mean', 'val_ppl_std', 'test_ppl_mean', 'test_ppl_std', 'td_mean', 'td_std', 'cs_mean', 'cs_std'])
