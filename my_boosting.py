def boosting(dataset,labels, model, problem_type, learning_rate = 0.5, B = 100):
    

    residue = np.array(merger(dataset, labels))
    prediction_rules = []
    residues_per_iteration = []
    predictions_per_iteration = []
    
   
    for i in range(B):
        
        learners = {'DecisionTree':[tree.DecisionTreeClassifier(random_state =  1, max_depth = 2),
                                tree.DecisionTreeRegressor(random_state = 1, max_depth = 2)],
                
                'SVM':[svm.SVC(), svm.SVR()],
                
                'Ridge':[linear_model.RidgeClassifier(), linear_model.Ridge(alpha = 1.0)]
                }
        
        new_model = learners[model][0] if problem_type == 'classification' else learners[model][1]
        
    
        prediction_rules.append(new_model.fit(residue[:, :-1],residue[:, -1]))
       
        ppt = np.array([prediction_rules[-1].predict(t[:-1].reshape(1, -1)) for t in residue])
        predictions_per_iteration.append([v[0] for v in ppt])
        
        if problem_type == 'regression':
            residue = np.array([np.append(t[:-1],[t[-1] - (learning_rate *prediction_rules[-1].predict(t[:-1].reshape(1, -1)))]) for t in residue])
        elif problem_type == 'classification':
            residue = np.array([np.append(t[:-1],1) 
                                if t[-1] - (learning_rate*prediction_rules[-1].predict(t[:-1].reshape(1, -1))) > 0.5
                                else np.append(t[:-1],0) for t in residue])
       
        residues_per_iteration.append([t[-1] for t in residue ])
    
    new_boosting_output = BoostingOutput()
    new_boosting_output.original_dataset = np.array(merger(dataset, labels))
    new_boosting_output.problem_type = problem_type
    new_boosting_output.model = model
    new_boosting_output.prediction_rules = np.array(prediction_rules)
    new_boosting_output.residues_per_iteration =np.array(residues_per_iteration)
    new_boosting_output.predictions_per_iteration =np.array(predictions_per_iteration)
    new_boosting_output.learning_rate = learning_rate

    return new_boosting_output

        
        

