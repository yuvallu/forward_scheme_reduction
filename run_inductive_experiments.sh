#!/usr/bin/env bash
for r in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    echo Genes ${r}
    python forward_inductive.py --allowed_scheme_list Sorted_schemes/models_genes_EK_3_100_500_10_50000_0experiment_distribution_var_0  --classifier SVM --data_name genes --kernel EK --depth 2 --dim 100 --num_samples 1000 --epochs 20 --batch_size 10000 --train_ratio ${r}

    echo Mutagenesis ${r}
    python forward_inductive.py --allowed_scheme_list Sorted_schemes/models_mutagenesis_EK_4_100_500_10_50000_0experiment_distribution_var_0 --classifier SVM --data_name mutagenesis --kernel EK --depth 3 --dim 100 --num_samples 5000 --epochs 20 --batch_size 50000 --train_ratio ${r}

    echo Mondial ${r}
    python forward_inductive.py --allowed_scheme_list Sorted_schemes/models_mondial_EK_3_100_500_10_50000_0experiment_distribution_var_0  --classifier SVM --data_name mondial --kernel EK --depth 3 --dim 100 --num_samples 5000 --epochs 20 --batch_size 50000 --train_ratio ${r}
done;

