# deep_asteroid_classifier

Initial runs were trying to tune hyper parameters for very small dataset we have

#experiment 1

adjusted learning rate function (which definitely helped smoothen loss curve but it is still rough); will keep the functionality as it may come in handy later on

#experiment 2

Raising epoch num obviously helps convergence and lowers overall loss, but overfitting is something to be aware of with such a small dataset

#experiment 3

Epoch num = 20; learning rate 0.0001;
Loss is lower and seems to be generally going down but necessarily converging;
Will now check accuracy scores to see if accuracy is generally improving

# ADDED ACCURACY TRACKING

#experiment 4

LOWER learning rate, 20 epochs. accuracy does not seem to go generally up or down.