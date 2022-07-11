class BoostingOutput():
    
    def __init__(self):
        
        self.original_dataset = None
        self.problem_type = None
        self.model = None
        self.residues_per_iteration = None
        self.prediction_rules = None
        self.predictions_per_iteration = None
        self.learning_rate = None

    
    def predict(self,data):
        data = np.array(data)
        if data.ndim == 1:
                data = data.reshape(1, -1)
        overall_pred = 0
        for rule in self.prediction_rules:
            boosted = self.learning_rate * rule.predict(data)
            overall_pred += boosted
        
        if self.problem_type == 'regression':
            return overall_pred
        elif self.problem_type == 'classification':
            return list(map(math.ceil,overall_pred)) 
    
    def loss(self):
        
        data = np.array(self.original_dataset[:, :-1])
        if data.ndim == 1:
                data = data.reshape(1, -1)
        overall_pred = 0
        for rule in self.prediction_rules:
            boosted = self.learning_rate * rule.predict(data)
            overall_pred += boosted
        
       
        if self.problem_type == 'regression':
            temp = [(u-v)**2 for u,v in zip(self.original_dataset[:, -1],overall_pred)]
            return {'MSE : ' + str(sum(temp)/len(temp))} 
        elif self.problem_type == 'classification':
            return {'error rate : ' + str(np.mean(self.original_dataset[:, -1] != list(map(math.ceil,overall_pred)) ))} 
        
   