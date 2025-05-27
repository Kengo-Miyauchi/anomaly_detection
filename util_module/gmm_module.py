from sklearn.mixture import GaussianMixture
import numpy as np

def calc_AnomalyScore(n_components, covariance_type, feature_train, feature_test, out_dir):
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42, n_init=10, max_iter=500)
    gmm.fit(feature_train)
    log_likelihood = -gmm.score_samples(feature_train)
    threshold = np.percentile(log_likelihood,99.9)
    anomaly_score = -gmm.score_samples(feature_test)
    img_path = out_dir + "/gmm_" + str(n_components) + "components.png"
    # os.makedirs(img_path, exist_ok=True)
    
    return anomaly_score, threshold, img_path