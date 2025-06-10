#!/bin/bash
# m_sets=(100 200)
m_sets=(200)
len_docss=(50 100)
num_topicss=(10 20)
t_sets=(5 10 20)

for m_set in "${m_sets[@]}"; do
  for len_docs in "${len_docss[@]}"; do
    for num_topics in "${num_topicss[@]}"; do
      for t_set in "${t_sets[@]}"; do
        for i in {0..19}; do
          python main_simulation.py --dataset acl --data_path ./data_acl_largev --emb_path ./embeddings/acl/skipgram_emb_300d.txt --min_df 10 --num_topics $num_topics --m_set $m_set --t_set $t_set --len_docs $len_docs --lr 0.0001 --epochs 1000 --mode train --idx_sim $i
        done
      done
    done
  done
done