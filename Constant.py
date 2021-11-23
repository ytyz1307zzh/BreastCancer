FEATURE_LIST = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
                "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
                "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
                "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]

MEAN_FEATURES = ['radius_mean', 'texture_mean', 'perimeter_mean',
                 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

WORST_FEATURES = ['radius_worst', 'texture_worst',
                  'perimeter_worst', 'area_worst', 'smoothness_worst',
                  'compactness_worst', 'concavity_worst', 'concave points_worst',
                  'symmetry_worst', 'fractal_dimension_worst']

LABEL = 'diagnosis'

# print(len(FEATURE_LIST))
ID2FEATURE = {i: FEATURE_LIST[i] for i in range(len(FEATURE_LIST))}
FEATURE2ID = {FEATURE_LIST[i]: i for i in range(len(FEATURE_LIST))}
