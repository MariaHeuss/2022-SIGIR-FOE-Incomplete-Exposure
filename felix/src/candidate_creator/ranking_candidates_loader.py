import pandas as pd
import numpy as np
from felix.src.candidate_creator.candidate import Candidate

"""

Create objects of type candidate to use for the ranking algorithms

Parts of this code refer to https://github.com/MilkaLichtblau/FA-IR_Ranking and 
https://github.com/MilkaLichtblau/BA_Laura

"""


class RankingCandidates:
    def __init__(self, query_number, candidates, protected, non_protected):
        self.candidates = candidates
        self.query_number = query_number
        self.protected = protected
        self.non_protected = non_protected


class RankingCandidatesLoader:
    def __init__(self, filename):
        self.filename = filename
        self.queries = []
        self.rankings = {}

    def load_learning_candidates(self, min_item_num=None):
        """

        @param filename: Path of input file. Assuming preprocessed CSV file:

            sensitive_attribute | session | label as index value | feature_1 | ... | feature_n

            sensitive_attribute: is either 0 for non-protected or 1 for protected
            session: indicates the query identifier of the file
            score: we assume that score is given indirectly as enumeration, therefore we normalize
            the score with 1 - score/len(query)

        return    a list with candidate objects from the inputed document, might contain multiple queries

        """

        self.rankings = {}
        rankings_as_list = []

        try:
            data = pd.read_csv(self.filename)
        except FileNotFoundError:
            raise FileNotFoundError(
                "File could not be found. Something must have gone wrong during preprocessing."
            )

        queryNumbers = data["session"]

        queryNumbers = queryNumbers.drop_duplicates()
        self.queries = queryNumbers

        for query in queryNumbers:
            dataQuery = data.loc[data.session == query]
            if min_item_num is not None and len(dataQuery) < min_item_num:
                continue
            nonProtected = []
            protected = []
            for row in dataQuery.itertuples():
                features = np.asarray(row[4:])
                # access second row of .csv with protected attribute 'H' = nonprotected group and 'L' = protected group
                if row[1] == "H":
                    nonProtected.append(
                        Candidate(
                            qualification=float(row[3]),
                            originalQualification=float(
                                row[3]
                            ),  # TODO why do we set the qualification and the original qualification here?
                            protectedAttributes=[],
                            query=row[2],
                            features=features,
                            outlier_feature=float(features[-1]),
                        )
                    )
                else:
                    protected.append(
                        Candidate(
                            qualification=float(row[3]),
                            originalQualification=float(row[3]),
                            protectedAttributes="protectedGroup",
                            query=row[2],
                            features=features,
                            outlier_feature=float(features[-1]),
                        )
                    )

            queryRanking = nonProtected + protected

            # sort candidates by credit scores
            protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
            nonProtected.sort(
                key=lambda candidate: candidate.qualification, reverse=True
            )

            # creating a color-blind ranking which is only based on scores
            queryRanking.sort(
                key=lambda candidate: candidate.qualification, reverse=True
            )
            ranking_candidates = RankingCandidates(
                query_number=queryNumbers,
                candidates=queryRanking,
                protected=protected,
                non_protected=nonProtected,
            )
            self.rankings[query] = ranking_candidates
            rankings_as_list += queryRanking
        self.queries = list(self.rankings.keys())
        return rankings_as_list, self.rankings, self.queries
