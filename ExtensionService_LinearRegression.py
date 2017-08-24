#! /usr/bin/env python3
import argparse
import json
import logging
import logging.config
import os
import sys
import time
from concurrent import futures
import numpy as np
import tensorflow as tf

# Add Generated folder to module path.
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PARENT_DIR, 'Generated'))

import ServerSideExtension_pb2 as SSE
import grpc
from ScriptEval_LinearRegression import ScriptEval
from SSEData_LinearRegression import FunctionType

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ExtensionService(SSE.ConnectorServicer):
    """
    A simple SSE-plugin created for Linear Regression with Tensorflow.
    """

    def __init__(self, funcdef_file):
        """
        Class initializer.
        :param funcdef_file: a function definition JSON file
        """
        self._function_definitions = funcdef_file
        self.scriptEval = ScriptEval()
        if not os.path.exists('logs'):
            os.mkdir('logs')
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logger.config')
        logging.config.fileConfig(log_file)
        logging.info('Logging enabled')

    @property
    def function_definitions(self):
        """
        :return: json file with function definitions
        """
        return self._function_definitions

    @property
    def functions(self):
        """
        :return: Mapping of function id and implementation
        """
        return {
            0: '_simple',
            1: '_estimator',
            2: '_polynomial'
        }

    """
    Implementation of added functions.
    """

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

            print("xSense: %r"% xSense)
            print("ySense: %r"% ySense)

            # Model parameters
            W = tf.Variable([2.5], dtype=tf.float32)
            b = tf.Variable([2.5], dtype=tf.float32)
            # Model input and output
            x = tf.placeholder(tf.float32)
            linear_model = W * x + b
            y = tf.placeholder(tf.float32)
#            W = tf.Print(W, [W], "W: ")
#            b = tf.Print(b, [b], "b: ")

            # loss
            loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
            # loss = tf.Print(loss, [loss], "loss: ")
            # optimizer
            optimizer = tf.train.GradientDescentOptimizer(0.000001)
            train = optimizer.minimize(loss)

            # training data
            x_train = [1, 2, 3, 4, 5, 6, 7]
            y_train = [1, 2, 3, 4, 5, 6, 7]
            # training loop
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init) # reset values to wrong
            for i in range(100):
#                sess.run(train, {x: x_train, y: y_train})
                sess.run(train, feed_dict={x: xSense, y: ySense})


            # evaluate training accuracy
#            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: xSense, y: ySense})
            print("Simple W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

            # test
            test_x = 5
            test_y = W * test_x + b
            #print(sess.run(test_y))

            # Iterating over rows
            for i in range(len(xSense)):
                predY = sess.run(linear_model, feed_dict={x: xSense[i]})
                result = predY[0]
                # print("predY %r"% result)

                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])

                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)

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

            print("xSense: %r"% xSense)
            print("ySense: %r"% ySense)

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
            # x_train = np.array([1., 2., 3., 4.])
            # y_train = np.array([0., -1., -2., -3.])
            x_train = np.array(xSense)
            y_train = np.array(ySense)
            # x_eval = np.array([2., 5., 8., 1.])
            # y_eval = np.array([-1.01, -4.1, -7, 0.])
            x_eval = np.array(xSense)
            y_eval = np.array(ySense)
            # x_pred = np.array([3., 4., 5., 6.])
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
            # print("Estimator eval metrics: %r"% eval_metrics)

            predictions = estimator.predict(input_fn=predict_input_fn, predict_keys=['y'])
            for i,p in enumerate(predictions):
                # Retrieve value from tensor
                result = p["y"]
                # print("Prediction %s: %s" % (x_pred[i], result))
                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])
                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)

    @staticmethod
    def _polynomial(request):
        """
        use tensorflow estimator with gradient descent optimizer to run polynomial linear regression expression: y = (Wx + b)i^5
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

            print("xSense: %r"% xSense)
            print("ySense: %r"% ySense)

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

            # predictions = estimator.predict(input_fn=predict_input_fn, predict_keys=['y'])
            # predictions = [{"i": 0, "y": 0}, {"i": 1, "y": 1}, {"i": 2, "y": 2}, {"i": 3, "y": 3}]
            for p in enumerate(yPrediction):
                # Retrieve value from tensor
                result = p[1]
                # print("Prediction %s: %s" % (x_pred[i], result))
                # print("Prediction %s: %s" % (i, result))
                # Create an iterable of Dual with a numerical value
                duals = iter([SSE.Dual(numData=result)])
                # Append the row data constructed to response_rows
                response_rows.append(SSE.Row(duals=duals))

            # Yield Row data as Bundled rows
            yield SSE.BundledRows(rows=response_rows)

    @staticmethod
    def _get_function_id(context):
        """
        Retrieve function id from header.
        :param context: context
        :return: function id
        """
        metadata = dict(context.invocation_metadata())
        header = SSE.FunctionRequestHeader()
        header.ParseFromString(metadata['qlik-functionrequestheader-bin'])

        return header.functionId

    """
    Implementation of rpc functions.
    """

    def GetCapabilities(self, request, context):
        """
        Get capabilities.
        Note that either request or context is used in the implementation of this method, but still added as
        parameters. The reason is that gRPC always sends both when making a function call and therefore we must include
        them to avoid error messages regarding too many parameters provided from the client.
        :param request: the request, not used in this method.
        :param context: the context, not used in this method.
        :return: the capabilities.
        """
        logging.info('GetCapabilities')

        # Create an instance of the Capabilities grpc message
        # Enable(or disable) script evaluation
        # Set values for pluginIdentifier and pluginVersion
        capabilities = SSE.Capabilities(allowScript=True,
                                        pluginIdentifier='Linear Regression - Qlik',
                                        pluginVersion='v1.0.0-beta1')

        # If user defined functions supported, add the definitions to the message
        with open(self.function_definitions) as json_file:
            # Iterate over each function definition and add data to the Capabilities grpc message
            for definition in json.load(json_file)['Functions']:
                function = capabilities.functions.add()
                function.name = definition['Name']
                function.functionId = definition['Id']
                function.functionType = definition['Type']
                function.returnType = definition['ReturnType']

                # Retrieve name and type of each parameter
                for param_name, param_type in sorted(definition['Params'].items()):
                    function.params.add(name=param_name, dataType=param_type)

                logging.info('Adding to capabilities: {}({})'.format(function.name,
                                                                     [p.name for p in function.params]))

        return capabilities

    def ExecuteFunction(self, request_iterator, context):
        """
        Call corresponding function based on function id sent in header.
        :param request_iterator: an iterable sequence of RowData.
        :param context: the context.
        :return: an iterable sequence of RowData.
        """
        # Retrieve function id
        func_id = self._get_function_id(context)
        logging.info('ExecuteFunction (functionId: {})'.format(func_id))

        # Disable cache for testing
        md = (('qlik-cache', 'no-store'),)
        context.send_initial_metadata(md)

        return getattr(self, self.functions[func_id])(request_iterator)

    def EvaluateScript(self, request, context):
        """
        Support script evaluation, based on different function and data types.
        :param request:
        :param context:
        :return:
        """
        # Retrieve header from request
        metadata = dict(context.invocation_metadata())
        header = SSE.ScriptRequestHeader()
        header.ParseFromString(metadata['qlik-scriptrequestheader-bin'])

        # Retrieve function type
        func_type = self.scriptEval.get_func_type(header)

        # Verify function type
        if (func_type == FunctionType.Tensor) or (func_type == FunctionType.Aggregation):
            return self.scriptEval.EvaluateScript(request, context, header, func_type)
        else:
            # This plugin does not support other function types than tensor and aggregation.
            # Make sure the error handling, including logging, works as intended in the client
            msg = 'Function type {} is not supported in this plugin.'.format(func_type.name)
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(msg)
            # Raise error on the plugin-side
            raise grpc.RpcError(grpc.StatusCode.UNIMPLEMENTED, msg)

    """
    Implementation of the Server connecting to gRPC.
    """

    def Serve(self, port, pem_dir):
        """
        Server
        :param port: port to listen on.
        :param pem_dir: Directory including certificates
        :return: None
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        SSE.add_ConnectorServicer_to_server(self, server)

        if pem_dir:
            # Secure connection
            with open(os.path.join(pem_dir, 'sse_server_key.pem'), 'rb') as f:
                private_key = f.read()
            with open(os.path.join(pem_dir, 'sse_server_cert.pem'), 'rb') as f:
                cert_chain = f.read()
            with open(os.path.join(pem_dir, 'root_cert.pem'), 'rb') as f:
                root_cert = f.read()
            credentials = grpc.ssl_server_credentials([(private_key, cert_chain)], root_cert, True)
            server.add_secure_port('[::]:{}'.format(port), credentials)
            logging.info('*** Running server in secure mode on port: {} ***'.format(port))
        else:
            # Insecure connection
            server.add_insecure_port('[::]:{}'.format(port))
            logging.info('*** Running server in insecure mode on port: {} ***'.format(port))

        server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', nargs='?', default='50054')
    parser.add_argument('--pem_dir', nargs='?')
    parser.add_argument('--definition-file', nargs='?', default='FuncDefs_LinearRegression.json')
    args = parser.parse_args()

    # need to locate the file when script is called from outside it's location dir.
    def_file = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), args.definition_file)

    calc = ExtensionService(def_file)
    calc.Serve(args.port, args.pem_dir)
