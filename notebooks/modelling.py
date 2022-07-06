from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator



def regression_model():

    # # Load training data
    # training = spark.read.format("libsvm")\
    #     .load("data/mllib/sample_linear_regression_data.txt")

    # lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # # Fit the model
    # lrModel = lr.fit(training)

    # # Print the coefficients and intercept for linear regression
    # print("Coefficients: %s" % str(lrModel.coefficients))
    # print("Intercept: %s" % str(lrModel.intercept))

    # # Summarize the model over the training set and print out some metrics
    # trainingSummary = lrModel.summary
    # print("numIterations: %d" % trainingSummary.totalIterations)
    # print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    # trainingSummary.residuals.show()
    # print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    # print("r2: %f" % trainingSummary.r2)


def GeneralizedLinearRegression():
    # Load training data
    dataset = spark.read.format("libsvm")\
        .load("data/mllib/sample_linear_regression_data.txt")

    glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)

    # Fit the model
    model = glr.fit(dataset)

    # Print the coefficients and intercept for generalized linear regression model
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    # Summarize the model over the training set and print out some metrics
    summary = model.summary
    print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    print("T Values: " + str(summary.tValues))
    print("P Values: " + str(summary.pValues))
    print("Dispersion: " + str(summary.dispersion))
    print("Null Deviance: " + str(summary.nullDeviance))
    print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("Deviance: " + str(summary.deviance))
    print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("AIC: " + str(summary.aic))
    print("Deviance Residuals: ")
    summary.residuals().show()



def DecisionTreeRegressor():
    # Load the data stored in LIBSVM format as a DataFrame.
    data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Train a DecisionTree model.
    dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

    # Chain indexer and tree in a Pipeline
    pipeline = Pipeline(stages=[featureIndexer, dt])

    # Train model.  This also runs the indexer.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    treeModel = model.stages[1]
    # summary only
    print(treeModel)