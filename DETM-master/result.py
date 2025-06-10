# %%
import pickle
import pandas as pd

# %%
results = []
for T in [5, 10, 20]:
    for M in [100, 200]:
        for K in [10, 20]:
            for N in [50, 100]:
                for sim in range(20):
                    with open('./res_old/res_M{}_N{}_K{}_T{}_{}.pkl'.format(M, N, K, T, sim), 'rb') as file:
                        res = pickle.load(file)
                    res['T'] = T
                    res['M'] = M
                    res['K'] = K
                    res['N'] = N
                    res['sim'] = sim
                    
                    with open('./res/res_M{}_N{}_K{}_T{}_{}.pkl'.format(M, N, K, T, sim), 'rb') as file:
                        res2 = pickle.load(file)
                    res['train_ppl2'] = res2['train_ppl']
                    res['val_ppl2'] = res2['val_ppl']
                    res['test_ppl2'] = res2['test_ppl']
                    results.append(res)

results = pd.DataFrame(results)

# %%
results = results.groupby(['T', 'M', 'K', 'N']).aggregate(['mean', 'std']).reset_index()

# %%
results.to_csv('simulation_results.csv')

# %%
with open('./res/res_M{}_N{}_K{}_T{}_{}.pkl'.format(M, N, K, T, sim), 'rb') as file:
