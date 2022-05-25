from tabulate import tabulate
import random
import pickle
import os
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from collections import defaultdict
from felix.src.algorithms.listnet import ListNet
from felix.src.candidate_creator.ranking_candidates_loader import RankingCandidatesLoader
from felix.src.algorithms.run_foeir import runFOEIR
from felix.src.measures.run_metrics import runMetrics
from felix.src.algorithms.plackett_luce.rankers import (
    PLRanker,
    DeterministicRanker,
)
import copy

"""
Parts of this code refer to https://github.com/MilkaLichtblau/BA_Laura
"""

path = os.getcwd()
result_path = path + "/results/"


def get_predicted_test_candidates_per_fold(
    data_path,
    label_prediction,
    upsample=False,
    min_item_num=None,
    upsample_number=100,
    normalize_scores=True,
    normalization_epsilon=1e-4,
    repeat=1,
):
    """
    Returns the test ranking candidates, with either oracle labels or listnet predicted scores,
    trained on training candidates.
    """

    predicted_test_candidates_per_fold = {}
    for dirpath, dirnames, files in os.walk(data_path):

        if "fold" in dirpath:
            test_path, train_path = (
                dirpath + "/" + "test.csv",
                dirpath + "/" + "train.csv",
            )
            train_candidate_loader = RankingCandidatesLoader(train_path)
            (
                train_rankings_as_lists,
                train_rankings,
                train_queryNumbers,
            ) = train_candidate_loader.load_learning_candidates(min_item_num=None)

            test_candidate_loader = RankingCandidatesLoader(test_path)
            (
                test_rankings_as_lists,
                test_rankings,
                test_queryNumbers,
            ) = test_candidate_loader.load_learning_candidates(
                min_item_num=min_item_num
            )

            if upsample:
                # Upsample with random items from train queries, labeled as non-relevant.
                upsample_candidates = copy.deepcopy(train_rankings_as_lists)
                upsample_candidates = [
                    c for c in upsample_candidates if c.originalQualification == 0
                ]
                for candidate in upsample_candidates:
                    candidate.qualification = 0
                    candidate.originalQualification = 0
                for query in test_rankings:
                    num_candidates = len(test_rankings[query].candidates)
                    if num_candidates < upsample_number:
                        indices = np.random.randint(
                            0,
                            len(upsample_candidates),
                            size=upsample_number - num_candidates,
                        )
                        for i in indices:
                            new_candidate = copy.deepcopy(upsample_candidates[i])
                            new_candidate.query = query
                            test_rankings[query].candidates.append(new_candidate)

            for split in range(repeat):
                num_train_rankings = len(train_rankings)
                size_train_set_split = int(1 * num_train_rankings)
                train_set = {
                    key: train_rankings[key]
                    for key in random.sample(
                        train_rankings.keys(), size_train_set_split
                    )
                }

                to_remove_k = []
                for k in train_rankings.keys():
                    lv = [i.originalQualification for i in train_rankings[k].candidates]
                    if sum(lv) == 0:
                        to_remove_k.append(k)
                for k in to_remove_k:
                    del train_set[k]
                if label_prediction == "ListNet":
                    agent = ListNet(
                        verbose=50,
                        max_iter=10,
                        val_ratio=0.2,
                        n_thres_cand=100,
                        batch_size=128,
                    )
                    agent.fit(
                        candidate_dict=train_set,
                    )

                    # Estimate scores for the test data:
                    test_rankings_predicted = copy.deepcopy(test_rankings)
                    test_rankings_predicted = agent.test(
                        candidates=test_rankings_predicted, noscore=True
                    )
                elif label_prediction == "oracle":
                    test_rankings_predicted = copy.deepcopy(test_rankings)
                    # use the original labels as scores
                else:
                    ValueError(
                        "Please choose one of the implemented label prediction options"
                    )
                for query in test_rankings_predicted:
                    cand = test_rankings_predicted[query].candidates
                    if normalize_scores:
                        learned_scores = [candidate.learnedScores for candidate in cand]
                        min_score = np.min(learned_scores)
                        max_score = np.max(learned_scores)
                        for candidate in cand:
                            candidate.learnedScores = (
                                candidate.learnedScores - min_score
                            ) * (1 - normalization_epsilon) / (
                                max_score - min_score
                            ) + normalization_epsilon

                predicted_test_candidates_per_fold[
                    dirpath + str(split)
                ] = test_rankings_predicted
    return predicted_test_candidates_per_fold


def run_experiment(
    data_path="/data/TREC2020/features/fold",
    label_prediction="oracle",
    normalize_scores=True,
    normalization_epsilon=1e-4,
    plackett_luce=False,
    deterministic_ranker=False,
    uniform_stochastic_policy=False,
    lp_outlier_objective=False,
    fairness_constraint="individual_fairness",
    decomposition_method="outlier_resample",
    number_of_resamples=10,
    top_k=None,
    max_num_items=60,
    min_item_num=None,
    upsample=False,
    upsample_number=100,
    outlier_top_k=10,
    repeat=5,
    predicted_test_candidates_per_fold={},
    mrp_matrices=None,
):
    """
    Runs a fair ranking experiment. Returns results in form of a list of dictionaries, candidates with
    predicted ranking scores and marginal rank probability matrices that can be used in next experiment.

    data_path: str
        path to data,
    label_prediction: "oracle" or "ListNet"
        method to predict labels,
    normalize_scores: bool
        if true predicted scores are normalized,
    normalization_epsilon: float
        epsilon used in normalization for more stability,
    plackett_luce: bool
        if true, plackett-luce model is used as probabilistic ranker,
    deterministic_ranker: bool
        if true, deterministic ranker is used,
    uniform_stochastic_policy:bool
        if true stochastic policy with uniform distribution over items is used as ranker,
    lp_outlier_objective: bool
        if true linear programming approach to fairness is used,
    fairness_constraint: "individual_fairness" or "group_fairness"
        kind of fairness being used (only if lp_outlier_objective is true),
    decomposition_method: "outlier_resample" or vanilla_BvN
        kind of decomposition method used (only if lp_outlier_objective is true),
    number_of_resamples: int
        number of resampling iterations to use if decomposition_method is 'resample',
    top_k: None or int
        length of the rankings if None all items are ranked,
    max_num_items: int
        maximum number of items to be used in the experiments,
    min_item_num: int
        minimum number of items that a query has to possess to be considered in experiment,
    upsample: bool
        if true number of candidates gets upsampled,
    upsample_number: int
        number to which candidates get upsampled if upsample is true,
    outlier_top_k: int
        number of top candidates in ranking to consider for outlier metrics,
    repeat: int
        number of times to run the experiment with different train/val splits,
    predicted_test_candidates_per_fold: dict
        if non-empty, predetermined labels are used instead of re-determining the labels,
    mrp_matrices: None or dict
        if not None, predetermined stochastic matrices are used instead of re-determining them,
    """
    results = []
    if mrp_matrices is None:
        mrp_matrices = defaultdict(dict)

    startTime = datetime.datetime.now()

    # If there have not been any predicted candidates inputted, get them and predict their score.
    if not predicted_test_candidates_per_fold:
        predicted_test_candidates_per_fold = get_predicted_test_candidates_per_fold(
            path + data_path,
            label_prediction,
            upsample,
            min_item_num=min_item_num,
            upsample_number=upsample_number,
            normalize_scores=normalize_scores,
            normalization_epsilon=normalization_epsilon,
            repeat=repeat,
        )
        print(path + data_path)

    #################################################################
    for fold in predicted_test_candidates_per_fold:
        mrp_matrices_fold = mrp_matrices[fold]
        test_rankings = predicted_test_candidates_per_fold[fold]

        queryNumbers = test_rankings.keys()

        evalResults = []

        progress_monitor = 0

        for query in queryNumbers:
            print("Query number: ", query)
            progress_monitor += 1
            print(
                "************ ",
                progress_monitor,
                " / ",
                len(queryNumbers),
                " ************",
            )

            query_candidates = test_rankings[query].candidates

            # sorting the ranking in accordance with its original scores
            # to be able to only use the top_k items.
            query_candidates.sort(
                key=lambda candidate: candidate.qualification, reverse=True
            )
            query_candidates = query_candidates[:max_num_items]

            num_protected = sum(
                [candidate.isProtected for candidate in query_candidates]
            )

            # If we work with group fairness check whether there is at least
            # one protected/nonprotected item in the ranking.
            if fairness_constraint == "group_fairness" and num_protected in [
                0,
                len(query_candidates),
            ]:
                continue

            if plackett_luce:
                pl_ranker = PLRanker(query_candidates, 1000)
                stochastic_policy = pl_ranker.get_stochastic_policy(
                    top_k=top_k, query=query
                )

            elif deterministic_ranker:
                ranker = DeterministicRanker(query_candidates)
                stochastic_policy = ranker.get_stochastic_policy(
                    top_k=top_k, query=query
                )

            elif uniform_stochastic_policy:
                pl_ranker = PLRanker(query_candidates, 1000, uniform_scores=True)
                stochastic_policy = pl_ranker.get_stochastic_policy(
                    top_k=top_k, query=query
                )

            elif fairness_constraint:
                stochastic_policy, isDTC = runFOEIR(
                    query_candidates,
                    outlier_objective=lp_outlier_objective,
                    individual_fairness=(fairness_constraint == "individual_fairness"),
                    top_k=top_k,
                    decomposition_method=decomposition_method,
                    number_of_resamples=number_of_resamples,
                    mrp_matrix=mrp_matrices_fold[query]
                    if query in mrp_matrices_fold.keys()
                    else None,
                )
                if (
                    query not in mrp_matrices_fold.keys()
                    and stochastic_policy is not None
                ):
                    mrp_matrices_fold[query] = stochastic_policy.compute_mrp()

            if stochastic_policy is None:
                print(
                    "No stochastic policy could be calculated for query: ", query
                )
            else:
                evalResults.append(
                    runMetrics(
                        stochastic_policy,
                        outlier_top_k=outlier_top_k,
                    )
                )
        mrp_matrices[fold] = mrp_matrices_fold
        listResults = evalResults

        results.append(listResults)

    endTime = datetime.datetime.now()
    print("Total time of execution: " + str(endTime - startTime))
    return results, predicted_test_candidates_per_fold, mrp_matrices


def table_final_results(results, file_name=None, raw_data_file_name=None):
    result_df = {}
    Path(result_path).mkdir(parents=True, exist_ok=True)
    if raw_data_file_name:
        a_file = open(result_path + raw_data_file_name, "wb")
        pickle.dump(results, a_file)
        a_file.close()

    for experiment in results:
        fold_dict = defaultdict(list)
        for fold_results in results[experiment]:
            fold_results = pd.DataFrame(fold_results)

            fold_dict["ndcg1"].append(fold_results["ndcg1"].mean())
            fold_dict["ndcg5"].append(fold_results["ndcg5"].mean())
            fold_dict["ndcg10"].append(fold_results["ndcg10"].mean())
            fold_dict["outlierness_omit"].append(
                fold_results["outlierness_omit"].mean()
            )
            fold_dict["outlier_count"].append(fold_results["outlier_count"].mean())
            fold_dict["outlier_probability"].append(
                fold_results["outlier_probability"].mean()
            )
            fold_dict["EEL"].append(fold_results["EEL"].mean())
        result_df[experiment] = {
            "ndcg1": str(round(np.mean(fold_dict["ndcg1"]), 4))
            + "+-"
            + str(round(np.std(fold_dict["ndcg1"]), 6)),
            "ndcg5": str(round(np.mean(fold_dict["ndcg5"]), 4))
            + "+-"
            + str(round(np.std(fold_dict["ndcg5"]), 6)),
            "ndcg10": str(round(np.mean(fold_dict["ndcg10"]), 4))
            + "+-"
            + str(round(np.std(fold_dict["ndcg10"]), 6)),
            "outlierness_omit": str(round(np.mean(fold_dict["outlierness_omit"]), 4))
            + "+-"
            + str(round(np.std(fold_dict["outlierness_omit"]), 4)),
            "outlier_count": str(round(np.mean(fold_dict["outlier_count"]), 4))
            + "+-"
            + str(round(np.std(fold_dict["outlier_count"]), 4)),
            "outlier_probability": str(
                round(np.mean(fold_dict["outlier_probability"]), 4)
            )
            + "+-"
            + str(round(np.std(fold_dict["outlier_probability"]), 4)),
            "EEL": str(round(np.mean(fold_dict["EEL"]), 4))
            + "+-"
            + str(round(np.std(fold_dict["EEL"]), 4)),
        }
    result_df = pd.DataFrame(result_df)
    print(tabulate(result_df, headers="keys", tablefmt="psql"))
    if file_name is not None:
        path = result_path + file_name
        result_df.to_csv(path)
