import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz2.44.1/bin/'
from sklearn import datasets, metrics, model_selection, tree
import numpy as np
import matplotlib.pyplot as plt
import graphviz


breastCancer = datasets.load_breast_cancer()

numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

model = tree.DecisionTreeClassifier(criterion = "gini", max_depth=3)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 0)

model.fit(x_train,y_train)

y_predicted = model.predict(x_test)

print("Recall Score:")
print(metrics.recall_score(y_test, y_predicted ))
print("Precision Score:")
print(metrics.precision_score(y_test, y_predicted))
print("Accuracy Score:")
print(metrics.accuracy_score(y_test, y_predicted))
print("F1 Score:")
print(metrics.f1_score(y_test, y_predicted))

y_predicted_train = model.predict(x_train)

print("Recall Score:")
print(metrics.recall_score(y_train, y_predicted_train))
print("Precision Score:")
print(metrics.precision_score(y_train, y_predicted_train))
print("Accuracy Score:")
print(metrics.accuracy_score(y_train, y_predicted_train))
print("F1 Score:")
print(metrics.f1_score(y_train, y_predicted_train))


'''
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=breastCancer.feature_names,  
                   class_names=breastCancer.target_names,
                   filled=True)

fig.savefig("decision_tree.png")
'''
dot_data = tree.export_graphviz(model, out_file=None, feature_names=breastCancer.feature_names[:numberOfFeatures],
class_names=breastCancer.target_names, filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data) 
graph.render("breastCancerTreePlot") 
