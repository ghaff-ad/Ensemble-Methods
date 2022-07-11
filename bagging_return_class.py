class BaggingOutput():
    
    def __init__(self):
        
        self.original_dataset = None
        self.problem_type = None
        self.model = None
        self.boostrapped_sets = None
        self.OOB_bootstrapped_bool_list = None
        self.prediction_rules = None

        
    def predict(self,data):
        data = np.array(data)
        
        preds = []
        
        if data.ndim == 1:
                data = data.reshape(1, -1)
          
        for rule in self.prediction_rules:
            preds.append(rule.predict(data))
            
        preds = np.array(preds)
        if self.problem_type == 'classification':
            preds_out = []
            for i in range(len(preds[0])):
                p = preds[:,i].astype(int)
                preds_out.append(np.bincount(p).argmax())
            return preds_out
        elif self.problem_type == 'regression':
            return np.sum(preds,axis=0)/len(preds)
        
            
            
    
    def loss(self):   
        original_labels = self.original_dataset[:, -1]

        test_preds = []
        for i in range(len(self.original_dataset)):
            bts_preds = []
            for j in range(len(self.boostrapped_sets)):
                bts_preds.append(self.boostrapped_sets[j][i][-1])
            if self.problem_type == 'classification':
                test_preds.append(max(bts_preds, key = bts_preds.count))
            elif self.problem_type == 'regression':
                test_preds.append(sum(bts_preds)/len(bts_preds))

    
        if self.problem_type == 'classification':
            return {'error rate : ' + str(np.mean(test_preds != original_labels) )} 
        elif self.problem_type == 'regression':
            temp = [(u-v)**2 for u,v in zip(test_preds,original_labels)]
            return {'MSE : ' + str(sum(temp)/len(temp))}
                
                
    def OOB_Score(self):
    
        original_labels = self.original_dataset[:, -1]
        OOB_preds = []
        active_original_labels = []
        for i in range(len(self.original_dataset)):
            bt_preds = []
            for j in range(len(self.OOB_bootstrapped_bool_list)):
                if self.OOB_bootstrapped_bool_list[j][i] == True:
                    bt_preds.append(self.boostrapped_sets[j][i][-1])
                    
            if bt_preds:
                active_original_labels.append(original_labels[i])
                if self.problem_type == 'classification':
                    OOB_preds.append(max(bt_preds, key = bt_preds.count))
                elif self.problem_type == 'regression':
                    OOB_preds.append(sum(bt_preds)/len(bt_preds))
                    
        if self.problem_type == 'classification':
            active_original_labels = np.array(active_original_labels)
            return {'OOB_Score : ' + str(np.mean(OOB_preds != active_original_labels))} 
        elif self.problem_type == 'regression':
            temp = [(u-v)**2 for u,v in zip(OOB_preds,active_original_labels)]
            return {'OOB_MSE : ' + str(sum(temp)/len(temp))}
        