from felix.experimental_setup import run_experiment, table_final_results


def trec19_top_k():
    experiment_results = {}
    (experiment_results["PL ranker"], listnet_predicted_candidates, _) = run_experiment(
        data_path="/data/TREC2019/features/fold",
        label_prediction="ListNet",
        plackett_luce=True,
        top_k=10,
        min_item_num=10,
        upsample=True,
        upsample_number=50,
        repeat=5,
    )
    (experiment_results["PL uniform"], _, _) = run_experiment(
        top_k=10,
        uniform_stochastic_policy=True,
        predicted_test_candidates_per_fold=listnet_predicted_candidates,
    )
    (experiment_results["Deterministic ranking"], _, _) = run_experiment(
        top_k=10,
        deterministic_ranker=True,
        predicted_test_candidates_per_fold=listnet_predicted_candidates,
    )

    # Linear Programming approach
    (experiment_results["FELIX top-k"], _, mrp_matrices) = run_experiment(
        decomposition_method="vanilla_BvN",
        lp_outlier_objective=False,
        predicted_test_candidates_per_fold=listnet_predicted_candidates,
        top_k=10,
    )
    # LP outlier resample
    (experiment_results["FELIX outlier resample"], _, _) = run_experiment(
        decomposition_method="outlier_resample",
        predicted_test_candidates_per_fold=listnet_predicted_candidates,
        lp_outlier_objective=False,
        top_k=10,
        mrp_matrices=mrp_matrices,
        number_of_resamples=20,
    )
    # omit (with std method)
    (experiment_results["OMIT"], _, mrp_matrices) = run_experiment(
        decomposition_method="vanilla_BvN",
        lp_outlier_objective=True,
        predicted_test_candidates_per_fold=listnet_predicted_candidates,
        top_k=10,
    )
    table_final_results(
        experiment_results,
        file_name="trec19_topk_experiment_resultsv",
        raw_data_file_name="trec19_topk_experiment_results.pkl",
    )
