"""
Most parts of this code refer to https://github.com/MilkaLichtblau/FA-IR_Ranking
"""

import uuid


class Candidate(object):
    """
    represents a candidate in a set that is passed to a search algorithm
    a candidate composes of a qualification and a list of protected attributes (strings)
    if the list of protected attributes is empty/null this is a candidate from a non-protected group
    """

    def __init__(
        self,
        qualification,
        originalQualification,
        protectedAttributes,
        query,
        features,
        outlier_feature,
    ):
        """
        qualification : float
            describes the relevance of an item to the query,
        protectedAttributes: list of strings
            represent the protected attributes,
        index: int
            index of a candidate in a ranking,
        query: int
            query number,
        features: numpy array
            with feature vector for a candidate inside,
        outlier_feature: float
            number that will be used in the outlier detection,
        """
        self.qualification = qualification
        self.protectedAttributes = protectedAttributes
        # keeps the candidate's initial qualification for evaluation purposes
        self.originalQualification = originalQualification
        self.learnedScores = originalQualification
        self.uuid = uuid.uuid4()
        # query number for more than one data set
        self.query = query
        # numpy array with features inside
        self.features = features

        self.outlier_feature = outlier_feature

    @property
    def isProtected(self):
        """
        true if the list of ProtectedAttribute elements actually contains anything
        false otherwise
        """
        return not self.protectedAttributes == []


def get_test_candidate(relevance=0.0, outlier_feature=0):
    candidate = Candidate(
        qualification=relevance,
        originalQualification=relevance,
        protectedAttributes=None,
        query=None,
        features=None,
        outlier_feature=outlier_feature,
    )
    return candidate
