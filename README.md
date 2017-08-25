# Example: Linear Regression
In this example we cover some basic Tensorflow models and support for different SSE function types. It's based on the [Column Summing example](https://github.com/qlik-oss/server-side-extension/tree/master/examples/python/ColumnOperations) for Qlik's Server Side Extensions (SSE).

To learn more about Tensorflow, here is their [Getting Started Guide](https://www.tensorflow.org/get_started/get_started) and [Installation Instructions](https://www.tensorflow.org/install/). Please note that this project was created with Tensorflow v1.3.0 and Python v3.5.4

## Content
* [Defined functions](#defined-functions)
    * [`simple` function](#simple-function)
    * [`estimator` function](#estimator-function)
    * [`polynomial` function](#polynomial-function)
* [Sense document](#sense-document)
* [Run the example!](#run-the-example)
* [Conclusion](#conclusion)
* [Future Work](#future-work)

## Defined functions
This plugin has three user defined functions, `simple`, `estimator` and `polynomial`, all operating on numerical data. The `ExecuteFunction` method in the `ExtensionService` class is the same for any of the example plugins, but the JSON file and the `functions` method are different. The JSON file for this plugin includes the following information:

| __Function Name__ | __Id__ | __Type__ | __ReturnType__ | __Parameters__ |
| ----- | ----- | ----- | ------ | ----- |
| simple | 0 | 2 (tensor) | 1 (numeric) | __name:__ 'col1', __type:__ 1 (numeric); __name:__ 'col2', __type:__ 1(numeric) |
| estimator | 0 | 2 (tensor) | 1 (numeric) | __name:__ 'col1', __type:__ 1 (numeric); __name:__ 'col2', __type:__ 1(numeric) |
| polynomial | 0 | 2 (tensor) | 1 (numeric) | __name:__ 'col1', __type:__ 1 (numeric); __name:__ 'col2', __type:__ 1(numeric) |

The ID is mapped to the implemented function name in the `functions` method, below:
```python
import ServerSideExtension_pb2 as SSE

class ExtensionService(SSE.ConnectorServicer):
    ...
    @property
    def functions(self):
        return {
            0: '_simple',
            1: '_estimator',
            2: '_polynomial'
        }
```

### `simple` function
The first function is a Tensorflow function that manually creates a session to run the linear regression expression y=Wx + b. We iterate over the `BundledRows` and extract the numerical values, which we then use with placeholders to train our Tensorflow model using a Gradient Descent Optimizer. After training, we once again run through all x values to derive the predicted y values. Finally, we transform the results into the desired form in order to return them to the Sense client.

```python
    @staticmethod
    def _simple(request):
        """
        use basic tensorflow session to run linear regression expression: y = Wx + b
        :param request: an iterable sequence of RowData
        :return: the same iterable sequence of row data as received
        """

        # Iterate over bundled rows
        for request_rows in request:
            response_rows = []
            xSense = []
            ySense = []
            # Iterating over rows
            for row in request_rows.rows:
                # Retrieve the numerical value of the parameters
                # Two columns are sent from the client, hence the length of params will be 2
                xSense.append(row.duals[0].numData)
                ySense.append(row.duals[1].numData)
                params = [d.numData for d in row.duals]

            # Model parameters
            W = tf.Variable([2.5], dtype=tf.float32)
            b = tf.Variable([2.5], dtype=tf.float32)
            # Model input and output
            x = tf.placeholder(tf.float32)
            linear_model = W * x + b
            y = tf.placeholder(tf.float32)

            # loss
            loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
            # optimizer
            optimizer = tf.train.GradientDescentOptimizer(0.000001)
            train = optimizer.minimize(loss)

            # training loop
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init) # reset values to wrong
            for i in range(100):
                sess.run(train, feed_dict={x: xSense, y: ySense})

            # evaluate training accuracy
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: xSense, y: ySense})
            print("Simple W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

            # Iterating over rows
            for i in range(len(xSense)):
                predY = sess.run(linear_model, feed_dict={x: xSense[i]})
                result = predY[0]

                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])

                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)
```

### `estimator` function
The second function uses Tensorflow's built-in estimator to help automate training, testing and predicting. We iterate over the `BundledRows` again and retrieve the numerical values, define our model, train, evaluate and finally predict. We then iterate over all the predicted y values and then return the result as bundled rows.

```python
    @staticmethod
    def _estimator(request):
        """
        use tensorflow estimator with gradient descent optimizer to run linear regression expression: y = Wx + b
        :param request: an iterable sequence of RowData
        :return: the same iterable sequence of row data as received
        """

        # Iterate over bundled rows
        for request_rows in request:
            response_rows = []
            xSense = []
            ySense = []
            # Iterating over rows
            for row in request_rows.rows:
                # Retrieve the numerical value of the parameters
                # Two columns are sent from the client, hence the length of params will be 2
                xSense.append(row.duals[0].numData)
                ySense.append(row.duals[1].numData)
                params = [d.numData for d in row.duals]

            def model_fn(features, labels, mode):
                """
                Declare list of features, only have one real-valued feature
                """
                # Build a linear model and predict values
                W = tf.get_variable("W", [1], dtype=tf.float64)
                b = tf.get_variable("b", [1], dtype=tf.float64)
                y = W * features['x'] + b
                loss = None
                train = None
                if labels is not None:
                    # Loss sub-graph
                    loss = tf.reduce_sum(tf.square(y - labels))
                    # Training sub-graph
                    global_step = tf.train.get_global_step()
                    optimizer = tf.train.GradientDescentOptimizer(0.00001)
                    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

                # EstimatorSpec connects subgraphs we built to the
                # appropriate functionality.
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"y":y},
                    loss=loss,
                    train_op=train)

            estimator = tf.estimator.Estimator(model_fn=model_fn)
            # define our data sets
            x_train = np.array(xSense)
            y_train = np.array(ySense)
            x_eval = np.array(xSense)
            y_eval = np.array(ySense)
            x_pred = np.array(xSense)

            input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_train}, y_train, batch_size=4, num_epochs=100, shuffle=False)
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                {"x": x_eval}, y_eval, batch_size=4, num_epochs=100, shuffle=False)
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": x_pred}, num_epochs=1, shuffle=False)

            # train
            estimator.train(input_fn=input_fn, steps=100)
            # Here we evaluate how well our model did.
            train_metrics = estimator.evaluate(input_fn=train_input_fn)
            eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
            print("Estimator train metrics: %r"% train_metrics)

            predictions = estimator.predict(input_fn=predict_input_fn, predict_keys=['y'])
            for i,p in enumerate(predictions):
                # Retrieve value from tensor
                result = p["y"]
                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])
                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)
```

### `polynomial` function
The third function uses Tensorflow (this time without an Estimator) to run training and predict a polynomial linear regression expression. We iterate over the `BundledRows` again and retrieve the numerical values, build our model, train, evaluate and finally predict. We then iterate over all the predicted y values and then return the result as bundled rows.

```python
    @staticmethod
    def _polynomial(request):
        """
        use tensorflow with gradient descent optimizer to run polynomial linear regression expression: y = (Wx + b)i^5
        :param request: an iterable sequence of RowData
        :return: the same iterable sequence of row data as received
        """

        # Iterate over bundled rows
        for request_rows in request:
            response_rows = []
            xSense = []
            ySense = []
            # Iterating over rows
            for row in request_rows.rows:
                # Retrieve the numerical value of the parameters
                # Two columns are sent from the client, hence the length of params will be 2
                xSense.append(row.duals[0].numData)
                ySense.append(row.duals[1].numData)
                params = [d.numData for d in row.duals]

            # %% tf.placeholders for the input and output of the network. Placeholders are
            # variables which we need to fill in when we are ready to compute the graph.
            X = tf.placeholder(tf.float32)
            Y = tf.placeholder(tf.float32)

            # %% Instead of a single factor and a bias, we'll create a polynomial function
            # of different polynomial degrees.  We will then learn the influence that each
            # degree of the input (X^0, X^1, X^2, ...) has on the final output (Y).
            Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
            for pow_i in range(1, 5):
                W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
                Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

            # %% Loss function will measure the distance between our observations
            # and predictions and average over them.
            cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (len(xSense) - 1)

            # %% if we wanted to add regularization, we could add other terms to the cost,
            # e.g. ridge regression has a parameter controlling the amount of shrinkage
            # over the norm of activations. the larger the shrinkage, the more robust
            # to collinearity.
            # cost = tf.add(cost, tf.mul(1e-6, tf.global_norm([W])))

            # %% Use gradient descent to optimize W,b
            # Performs a single step in the negative gradient
            learning_rate = 0.00000002
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

            # %% We create a session to use the graph
            n_epochs = 1000
            with tf.Session() as sess:
                # Here we tell tensorflow that we want to initialize all
                # the variables in the graph so we can use them
                sess.run(tf.global_variables_initializer())

                # Fit all training data
                prev_training_cost = 0.0
                for epoch_i in range(n_epochs):
                    for (x, y) in zip(xSense, ySense):
                        sess.run(optimizer, feed_dict={X: x, Y: y})

                    training_cost = sess.run(
                        cost, feed_dict={X: xSense, Y: ySense})
                    # print("training_cost %s"% training_cost)

                    # if epoch_i % 10 == 0:
                        # yPrediction = Y_pred.eval(feed_dict={X: xSense}, session=sess)
                        # print(yPrediction)

                    # Allow the training to quit if we've reached a minimum
                    if np.abs(prev_training_cost - training_cost) < 0.000001:
                        break
                    prev_training_cost = training_cost

                print("Polynomial training_cost %s"% training_cost)
                yPrediction = Y_pred.eval(feed_dict={X: xSense}, session=sess)
                print("Polynomial yPrediction %s"% yPrediction)

            for p in enumerate(yPrediction):
                # Retrieve value from tensor
                result = p[1]
                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])
                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)
```

## Sense document
I built this Sense app originally to validate a Natural Language Generation solutions from Qlik partner Narrative Science. There are are six sheets. The first three show all of the various data trends along with some NLG examples (you'll need to download & install the Narratives For Qlik extension to see those. The last three sheets contain the SSE examples. Each sheet has a filter to select various days and four line charts showing the three custom SSE functions along with a comparison using Qlik's built-in linear regression expression. Each of the three sheets runs all four techniques on different time-series trend metrics (Slow Increase, Alternating Increase and Alternating)

The user defined function calls are straightforward as implemented. Ensure you set the plugin name to Linear in your Settings.ini file for Qlik Sense Desktop. Then the functions are called with `Linear.simple(A,B)`, `Linear.estimator(A,B)` or `Linear.polynomial(A,B)`. You may notice that I subtract the date dimension so that it starts at zero to make my models easier to work with, though this shouldn't be necessary.

## Run the example!
To run this example, follow the instructions in [Getting started with the SSE Python examples](https://github.com/qlik-oss/server-side-extension/blob/master/examples/python/GetStarted.md).

## Conclusion
This has been a great learning experience to become better acquainted with Qlik Server Side Extensions and Tensorflow. However, the models I've built function very poorly and Qlik's built-in linear regression is much more robust and speedy. Most of the models only function OK with certain data values. 

## Future work
There is surely quite a lot I could do to improve these linear regression models with Tensorflow - which I may continue to do. Though my real interest is in using Tensorflow's Recurrent Neural Networks such as Long Short Term Memory (LSTM) networks. I believe this should be idealy suited for classifying time-series data.
