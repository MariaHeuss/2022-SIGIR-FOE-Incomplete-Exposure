### Simulated experiments
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from felix.experimental_setup import table_final_results
from felix.experimental_setup import run_experiment
from felix.src.candidate_creator.candidate import get_test_candidate
from felix.src.candidate_creator.ranking_candidates_loader import RankingCandidates

""""
    Here we run the sensitivity analysis in a simulated set-up. We simulate the candidate 
    items with the 'get_mocked_candidates_per_query_per_fold' function. The relevance scores 
    are sampled from a distribution that has to be defined with the argument 
    'outlier_score_distribution'. The function 'candidate_set_size_experiment' is running an 
    experiment with individual fairness and the disparate treatment constraint, looking 
    at the impact of the number of candidates on the performance of the new method FELIX. 
    The function 'number_of_resample_experiment' is running the same experiment but with 
    a varying number of resampling  iterations. 
"""


path = os.getcwd()
result_path = path + "/results/"
figure_path = path + "/figures/"


def sample_normal_distribution(sample_size=10, std=1, mean=0):
    return np.random.normal(loc=mean, scale=std, size=sample_size)


def sample_powerlaw_distribution(sample_size=10, alpha=1):
    exp = -1 / alpha
    return np.array([u ** exp for u in np.random.random(sample_size)])


def sample_log_normal_distribution(sample_size=10, std=1, mean=0):
    normal_scores = np.random.normal(loc=mean, scale=std, size=sample_size)
    log_normal_scores = np.exp(normal_scores)
    return log_normal_scores


def sample_uniform_distribution(sample_size=10):
    return np.random.uniform(low=0, high=1, size=sample_size)


def get_mocked_candidates(
    num_candidates, relevance_distribution, outlier_score_distribution
):
    relevance_scores = relevance_distribution(sample_size=num_candidates)
    outlier_scores = outlier_score_distribution(sample_size=num_candidates)
    candidates = [
        get_test_candidate(relevance_scores[i], outlier_scores[i])
        for i in range(len(relevance_scores))
    ]
    return candidates


def get_mocked_candidates_per_query_per_fold(
    num_queries, num_candidates, relevance_distribution, outlier_score_distribution
):
    cand = {}
    for query in range(num_queries):
        query_candidates = get_mocked_candidates(
            num_candidates, relevance_distribution, outlier_score_distribution
        )
        query_candidates = RankingCandidates(
            query_number=query,
            candidates=query_candidates,
            protected=None,
            non_protected=None,
        )
        cand[query] = query_candidates
    return {"fold1": cand}


def candidate_set_size_experiment(
    num_queries,
    num_candidates_list,
    outlier_threshold=2.5,
    file_name="candidate_set_size",
):
    experiment_results = {}

    top_k = 10
    distributions = {
        "powerlaw": sample_powerlaw_distribution,
        "log_normal": sample_log_normal_distribution,
        "gaussian": sample_normal_distribution,
        "uniform": sample_uniform_distribution,
    }
    for distribution in distributions:
        for num_candidates in num_candidates_list:
            candidates = get_mocked_candidates_per_query_per_fold(
                num_queries=num_queries,
                num_candidates=num_candidates,
                relevance_distribution=sample_uniform_distribution,
                outlier_score_distribution=distributions[distribution],
            )
            (
                experiment_results[
                    "Vanilla_" + distribution + "_" + str(num_candidates)
                ],
                _,
                mrp_matrices,
            ) = run_experiment(
                fairness_constraint="individual_fairness",
                decomposition_method="vanilla_BvN",
                predicted_test_candidates_per_fold=candidates,
                top_k=top_k,
                max_num_items=None,
                upsample=False,
                normalize_scores=False,
            )

            (
                experiment_results[
                    "Resample_" + distribution + "_" + str(num_candidates)
                ],
                _,
                mrp_matrices,
            ) = run_experiment(
                fairness_constraint="individual_fairness",
                decomposition_method="outlier_resample",
                number_of_resamples=20,
                predicted_test_candidates_per_fold=candidates,
                top_k=top_k,
                max_num_items=None,
                upsample=False,
                mrp_matrices=mrp_matrices,
                outlier_threshold=outlier_threshold,
                normalize_scores=False,
            )
        table_final_results(
            experiment_results,
            file_name=file_name + ".csv",
            raw_data_file_name=file_name + ".pkl",
        )


def number_of_resample_experiment(
    num_queries, resample_num_list, file_name="number_of_resamples_experiment"
):
    num_candidates = 100
    experiment_results = {}

    top_k = 10
    distributions = {
        "powerlaw": sample_powerlaw_distribution,
        "log_normal": sample_log_normal_distribution,
        "gaussian": sample_normal_distribution,
        "uniform": sample_uniform_distribution,
    }
    for distribution in distributions:
        candidates = get_mocked_candidates_per_query_per_fold(
            num_queries=num_queries,
            num_candidates=num_candidates,
            relevance_distribution=sample_uniform_distribution,
            outlier_score_distribution=distributions[distribution],
        )
        (
            experiment_results["Vanilla_" + distribution + "_" + str(num_candidates)],
            _,
            mrp_matrices,
        ) = run_experiment(
            fairness_constraint="individual_fairness",
            decomposition_method="vanilla_BvN",
            predicted_test_candidates_per_fold=candidates,
            top_k=top_k,
            max_num_items=None,
            upsample=False,
            normalize_scores=False,
        )
        for resample in resample_num_list:
            (
                experiment_results["Resample_" + distribution + "_" + str(resample)],
                _,
                _,
            ) = run_experiment(
                fairness_constraint="individual_fairness",
                decomposition_method="outlier_resample",
                predicted_test_candidates_per_fold=candidates,
                top_k=top_k,
                max_num_items=None,
                upsample=False,
                outlier_top_k=top_k,
                number_of_resamples=resample,
                mrp_matrices=mrp_matrices,
                normalize_scores=False,
            )

    table_final_results(
        experiment_results,
        file_name=file_name + ".csv",
        raw_data_file_name=file_name + ".pkl",
    )


def plot_candidate_set_experiment_results(path_to_csv, candidate_list):
    # Read the csv with the results and format the dataframe
    results = pd.read_csv(os.path.join(result_path, path_to_csv))
    results = results.T
    results = results.rename(columns=results.iloc[0])
    results = results.drop(index="Unnamed: 0")
    # Plot outlier_probability against number of candidates
    fig1 = plt.gcf()
    for distribution in ["powerlaw", "log_normal", "gaussian", "uniform"]:
        subexperiment = results[results.index.str.contains(distribution)]
        vanilla = subexperiment[subexperiment.index.str.contains("Vanilla")]
        resample = subexperiment[subexperiment.index.str.contains("Resample")]
        for name, subresults in zip(["vanilla", "resample"], [vanilla, resample]):
            outlier_probability = subresults["outlier_probability"].tolist()
            outlier_probability = [float(o.split("+")[0]) for o in outlier_probability]
            plt.plot(
                candidate_list, outlier_probability, label=distribution + "_" + name
            )

    plt.xlabel("Number of Candidates")
    plt.ylabel("Outlier probability")
    plt.title("Impact of the number of available candidates on FELIX")
    plt.legend()
    plt.show()
    fig1.savefig(figure_path + "available_candidates.png", format="png")

    improvement_df = {"num_candidates": candidate_list}

    # Plot the improvement over no resamples against number of available candidates
    fig2 = plt.gcf()
    for distribution in ["powerlaw", "log_normal", "gaussian", "uniform"]:
        subexperiment = results[results.index.str.contains(distribution)]
        vanilla = subexperiment[subexperiment.index.str.contains("Vanilla")]
        resample = subexperiment[subexperiment.index.str.contains("Resample")]
        vanilla_outlier_probability = vanilla["outlier_probability"].tolist()
        vanilla_outlier_probability = [
            float(o.split("+")[0]) for o in vanilla_outlier_probability
        ]
        resample_outlier_probability = resample["outlier_probability"].tolist()
        resample_outlier_probability = [
            float(o.split("+")[0]) for o in resample_outlier_probability
        ]
        improvement = [
            (resample_outlier_probability[i] - vanilla_outlier_probability[i])
            / vanilla_outlier_probability[i]
            * 100
            if vanilla_outlier_probability[i] != 0
            else 0
            for i in range(len(vanilla_outlier_probability))
        ]
        plt.plot(candidate_list, improvement, label=distribution)
        improvement_df[distribution] = improvement

    improvement_df = pd.DataFrame(improvement_df)
    improvement_df.to_csv(figure_path + "candidates.csv")

    plt.xlabel("Number of Candidates")
    plt.ylabel("Relative improvement in Outlier Probability")
    plt.title("Impact of the number of available candidates on FELIX")
    plt.legend()
    plt.show()
    plt.draw()
    fig2.savefig(figure_path + "available_candidates_relative.png", format="png")


def plot_number_of_resamples_experiment_results(path_to_csv, number_of_resamples):
    number_of_resamples = [r + 1 for r in number_of_resamples]
    # Read the csv with the results and format the dataframe
    results = pd.read_csv(os.path.join(result_path, path_to_csv))
    results = results.T
    results = results.rename(columns=results.iloc[0])
    results = results.drop(index="Unnamed: 0")

    fig1 = plt.gcf()
    # Plot outlier_probability against number of resamples
    for distribution in ["powerlaw", "log_normal", "gaussian", "uniform"]:
        subexperiment = results[results.index.str.contains(distribution)]
        resample = subexperiment[subexperiment.index.str.contains("Resample")]

        outlier_probability = resample["outlier_probability"].tolist()
        outlier_probability = [float(o.split("+")[0]) for o in outlier_probability]
    plt.plot(number_of_resamples, outlier_probability, label=distribution)
    plt.xscale("log")
    plt.xlabel("Number of resampling iterations")
    plt.ylabel("Outlier probability")
    plt.title("Impact of the choice of the resampling parameter on FELIX")
    plt.legend()
    plt.show()
    fig1.savefig(figure_path + "resampling_parameter.png", format="png")

    improvement_df = {"number_of_resamples": number_of_resamples}
    fig2 = plt.gcf()
    # Plot the improvement over no resamples against number of resamples
    for distribution in ["powerlaw", "log_normal", "gaussian", "uniform"]:
        subexperiment = results[results.index.str.contains(distribution)]
        vanilla = subexperiment[subexperiment.index.str.contains("Vanilla")]
        resample = subexperiment[subexperiment.index.str.contains("Resample")]
        vanilla_outlier_probability = vanilla["outlier_probability"].tolist()
        vanilla_outlier_probability = [
            float(o.split("+")[0]) for o in vanilla_outlier_probability
        ]
        resample_outlier_probability = resample["outlier_probability"].tolist()
        resample_outlier_probability = [
            float(o.split("+")[0]) for o in resample_outlier_probability
        ]
        improvement = [
            (resample_outlier_probability[i] - vanilla_outlier_probability[0])
            / vanilla_outlier_probability[0]
            * 100
            if vanilla_outlier_probability[0] != 0
            else 0
            for i in range(len(resample_outlier_probability))
        ]
        improvement_df[distribution] = improvement
        plt.plot(number_of_resamples, improvement, label=distribution)
    improvement_df = pd.DataFrame(improvement_df)
    improvement_df.to_csv(figure_path + "resample_experiment.csv")
    plt.xlabel("Number of resampling iterations")
    plt.ylabel("Relative improvement in Outlier Probability")
    plt.title("Impact of the choice of the resampling parameter on FELIX")
    plt.legend()
    plt.show()
    fig2.savefig(figure_path + "resampling_parameter_relative.png", format="png")


def run_analysis_experiments():
    number_of_resamples = [0, 1, 2, 4, 8, 16, 32]
    number_of_resample_experiment(
        100, number_of_resamples, file_name="number_of_resamples"
    )
    plot_number_of_resamples_experiment_results(
        "number_of_resamples.csv", number_of_resamples
    )

    num_candidates_list = [10, 15, 20, 40, 60, 80, 100, 140, 200]
    candidate_set_size_experiment(
        100, num_candidates_list, file_name="candidate_set_size"
    )
    plot_candidate_set_experiment_results(
        "candidate_set_size.csv", num_candidates_list
    )
