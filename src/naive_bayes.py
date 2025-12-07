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
        
        # Priors ( (Count of Class + alpha) \ Total Samples + (alpha *Number of Classes) )
        for c in self.classes:
            count_c = np.sum(y == c)
            self.class_priors[c] = (count_c + self.alpha) / (n_samples + self.alpha * len(self.classes))
            
        # Likelihoods
        self.feature_probs = {}
        for feat_idx in range(n_features): #loop over all features
            self.feature_probs[feat_idx] = {}
            unique_vals = np.unique(X[:, feat_idx]) #get unique values for each feature
            
            for c in self.classes: # hena ana loop over all classes
                self.feature_probs[feat_idx][c] = {}
                X_c = X[y == c] # get all samples belonging to class c (like group by)
                count_c = len(X_c)
                
                for val in unique_vals:
                    count_val = np.sum(X_c[:, feat_idx] == val)  # count of feature value in class c(how many times this value appears in this class)
                    prob = (count_val + self.alpha) / (count_c + self.alpha * len(unique_vals)) #likelihood with laplace smoothing
                    self.feature_probs[feat_idx][c][val] = prob

    def predict_proba(self, X):
        probs = []
        
        for x in X: #loop 3la kol test sample
            class_scores = {}
            
            for c in self.classes: #loop 3la kol class
                log_prob = np.log(self.class_priors[c]) #start with prior
                
                for feat_idx, val in enumerate(x): #loop over features
                    if val in self.feature_probs[feat_idx][c]: #check if feature value exists in training data for this class
                        log_prob += np.log(self.feature_probs[feat_idx][c][val]) #add log likelihood
                    else: #handle unseen feature values with laplace smoothing
                        log_prob += np.log( 
                            self.alpha / (np.sum(list(self.feature_probs[feat_idx][c].values())) * len(self.feature_probs[feat_idx][c]) + self.alpha)
                        )
                
                class_scores[c] = log_prob #store log prob for class
            
            max_score = max(class_scores.values()) 
            exp_scores = {c: np.exp(score - max_score) for c, score in class_scores.items()} # for numerical stability(34an 2hrb mn underflow)
            total = sum(exp_scores.values()) 
            probs.append([exp_scores[c] / total for c in self.classes]) 
            
        return np.array(probs)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
