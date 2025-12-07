import numpy as np

class NaiveBayesClassifier:
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        n_features = X.shape[1]
        
        # Priors
        for c in self.classes:
            count_c = np.sum(y == c)
            self.class_priors[c] = (count_c + self.alpha) / (n_samples + self.alpha * len(self.classes))
            
        # Likelihoods
        self.feature_probs = {}
        for feat_idx in range(n_features):
            self.feature_probs[feat_idx] = {}
            unique_vals = np.unique(X[:, feat_idx])
            
            for c in self.classes:
                self.feature_probs[feat_idx][c] = {}
                X_c = X[y == c]
                count_c = len(X_c)
                
                for val in unique_vals:
                    count_val = np.sum(X_c[:, feat_idx] == val)
                    prob = (count_val + self.alpha) / (count_c + self.alpha * len(unique_vals))
                    self.feature_probs[feat_idx][c][val] = prob

    def predict_proba(self, X):
        probs = []
        
        for x in X:
            class_scores = {}
            
            for c in self.classes:
                log_prob = np.log(self.class_priors[c])
                
                for feat_idx, val in enumerate(x):
                    if val in self.feature_probs[feat_idx][c]:
                        log_prob += np.log(self.feature_probs[feat_idx][c][val])
                    else:
                        log_prob += np.log(
                            self.alpha / (np.sum(list(self.feature_probs[feat_idx][c].values())) * len(self.feature_probs[feat_idx][c]) + self.alpha)
                        )
                
                class_scores[c] = log_prob
            
            max_score = max(class_scores.values())
            exp_scores = {c: np.exp(score - max_score) for c, score in class_scores.items()}
            total = sum(exp_scores.values())
            probs.append([exp_scores[c] / total for c in self.classes])
            
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
