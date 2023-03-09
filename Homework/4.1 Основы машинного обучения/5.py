optimal_clf = KNeighborsClassifier(n_neighbors=4)
optimal_clf.fit(train_feature_matrix, train_labels)
pred_prob = optimal_clf.predict_proba(test_feature_matrix)
round(pred_prob[:, 2].mean(), 2)
