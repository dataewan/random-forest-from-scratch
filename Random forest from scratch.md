
Create a random forest from scratch, based on an exercise in chapter 6 of the excellent book [Hands on Machine Learning with Scikit-learn and Tensorflow](http://shop.oreilly.com/product/0636920052289.do).

## Train and fine tune a Decision Tree for the moons dataset

> a. Generate a moons dataset using `make_moons(n_samples=10000, noise=0.4)`

Reading the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) for the `make_moons` function tells us the following things:

 - It is for making play datasets with for clustering and classification
 - The `n_samples` parameter suggested controls the number of datapoints that it will return
 - `noise` is random noise added to the dataset
 - It is going to return two things: an array `X` containing the samples and an array `y` containing their class


```python
from sklearn.datasets import make_moons

moons_X, moons_y = make_moons(n_samples=10000, noise=0.4)
```

I like to take a look at the dataset before getting stuck in.
I could try printing it out, or can use the plotting functions in matplotlib.
Plotting is nicer, lets try that.


```python
%matplotlib inline
from matplotlib import pyplot as plt

figure = plt.figure(figsize=(10, 10))

plt.scatter(
    x=moons_X[:, 0], y=moons_X[:, 1], c=moons_y, alpha=0.5
);
```


![png](Random%20forest%20from%20scratch_files/Random%20forest%20from%20scratch_3_0.png)


Okay, that makes sense.
There are two classes there, which I'm going to try separating with the decision tree.
There is a bit of overlap between the classes, which is going to make things more difficult for the classifier.
Good. We wouldn't want it to have everything easy now 🙂

Just out of interest,
what happens if we reduced that noise parameter?
I'll create a temporary dataset that I'll throw away after.


```python
nonoise = make_moons(10000, noise=0.1)

figure = plt.figure(figsize=(10, 10))

plt.scatter(
    x=nonoise[0][:, 0], y=nonoise[0][:, 1], c=nonoise[1], alpha=0.5
);
```


![png](Random%20forest%20from%20scratch_files/Random%20forest%20from%20scratch_5_0.png)


Okay, makes sense.

> b. Split it into a training set and a test set using `train_test_split()`

Splitting the dataset into a holdout group that we'll use for evaluating the model (_test_ datasets),
and a dataset that we'll use for training the model (_train_ datasets).

Checking the results of your model on a dataset that you didn't use for training is important.
This is a good way to make sure that you're not overfitting your model.
If your model does well on the training dataset,
but poorly on the testing dataset, 
then it probably means that you've overfit your model.

The [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) for the `train_test_split` function tells us that it does the following things:

 - You give it parameters like this `train_test_split(*arrays, **options)`. What the single `*` means is that you can pass it in multiple arrays separated by commas, and internally in the function it will treat `arrays` like a list. The double `**` means that you can pass named parameters into the function, separated by commas, and it will treat `options` like a dictionary. I always forget exactly what this means when writing functions myself, but there's a good description on the [saltycrane](http://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/) blog.
 
 - One of the options that you can give to it is `test_size`, which defaults to $0.25$. This is the proportion of the datasets that get held back for the testing dataset.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(moons_X, moons_y)
```

> c. Use grid search with cross-validation (with the help of the `GridSearchCV` class) to find good hyperparameter values for a `DecisionTreeClassifier`. Hint: try various values for `max_leaf_nodes`.

Okay, this is where we start training a model over our data.
The `DecisionTreeClassifier` is the model we're going to use.
There are a bunch of hyperparameters you can select for the decision tree model.
We're only going to be looking at one parameter - the `max_leaf_nodes` parameter.

`GridSearchCV` searches through all the combinations of hyperparameter you give it,
and it finds out which is the best one.
It finds out the best combination is by applying cross validation.
The metric that it [optimises for by default](http://scikit-learn.org/stable/modules/grid_search.html#specifying-an-objective-metric) for a classification algorithm is accuracy
(for regression problems it uses the `r2_score`).

So it runs every combination of hyperparameter,
calculates a cross validated accuracy score (since we're using a classifier),
and finds out which combination is most accuract.
You tell the grid search which combinations of parameter to use by passing it a [param_grid](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
In this case I'll be passing it a dictionary with a range of values from 2 to 10 for the `max_leaf_nodes` parameter.
If I wanted to try more parameters, then I'd add them into that dictionary.


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier()
param_grid = {'max_leaf_nodes': range(2, 10)}

cv = GridSearchCV(tree, param_grid)
cv.fit(X=X_train, y=y_train)

print(cv.best_params_)
print(cv.best_score_)
```

    {'max_leaf_nodes': 9}
    0.858266666667


> d. Train it on the full training set using these hyperparameters, and measure your model's performance on the test set. You should get roughly $85\%$ to $87\%$ accuracy.

So now take the hyperparameters that were taken from above,
train a model with those hyperparameters,
and then compare it against that holdout group from above.


```python
from sklearn.metrics import accuracy_score

tree = DecisionTreeClassifier(max_leaf_nodes=4)
tree.fit(X=X_train, y=y_train)

y_pred = tree.predict(X=X_test)

accuracy_score(y_test, y_pred)
```




    0.85360000000000003


