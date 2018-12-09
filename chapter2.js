/*
Start of codes from chapter1.js
 */

// every Unit corresponds to a wire in the diagrams
var Unit = function(value, grad) {
    // value computed in the forward pass
    this.value = value;
    // the derivative of circuit output w.r.t this unit, computed in backward pass
    this.grad = grad;
}

var multiplyGate = function(){ };
multiplyGate.prototype = {
    forward: function(u0, u1) {
        // store pointers to input Units u0 and u1 and output unit utop
        this.u0 = u0;
        this.u1 = u1;
        this.utop = new Unit(u0.value * u1.value, 0.0);
        return this.utop;
    },
    backward: function() {
        // take the gradient in output unit and chain it with the
        // local gradients, which we derived for multiply gate before
        // then write those gradients to those Units.
        this.u0.grad += this.u1.value * this.utop.grad;
        this.u1.grad += this.u0.value * this.utop.grad;
    }
}

var addGate = function(){ };
addGate.prototype = {
    forward: function(u0, u1) {
        this.u0 = u0;
        this.u1 = u1; // store pointers to input units
        this.utop = new Unit(u0.value + u1.value, 0.0);
        return this.utop;
    },
    backward: function() {
        // add gate. derivative wrt both inputs is 1
        this.u0.grad += 1 * this.utop.grad;
        this.u1.grad += 1 * this.utop.grad;
    }
}

/*
End of codes from chapter1.js
 */

// A circuit: it takes 5 Units (x,y,a,b,c) and outputs a single Unit
// It can also compute the gradient w.r.t. its inputs
var Circuit = function() {
    // create some gates
    this.mulg0 = new multiplyGate();
    this.mulg1 = new multiplyGate();
    this.addg0 = new addGate();
    this.addg1 = new addGate();
};
Circuit.prototype = {
    forward: function(x,y,a,b,c) {
        this.ax = this.mulg0.forward(a, x); // a*x
        this.by = this.mulg1.forward(b, y); // b*y
        this.axpby = this.addg0.forward(this.ax, this.by); // a*x + b*y
        this.axpbypc = this.addg1.forward(this.axpby, c); // a*x + b*y + c
        return this.axpbypc;
    },
    backward: function(gradient_top) { // takes pull from above
        this.axpbypc.grad = gradient_top;
        this.addg1.backward(); // sets gradient in axpby and c
        this.addg0.backward(); // sets gradent in ax and by
        this.mulg1.backward(); // sets gradient in b and y
        this.mulg0.backward(); // sets gradient in a and x
    }
}

// SVM class
var SVM = function() {

    // random initial parameter values
    this.a = new Unit(1.0, 0.0);
    this.b = new Unit(-2.0, 0.0);
    this.c = new Unit(-1.0, 0.0);

    this.circuit = new Circuit();
};
SVM.prototype = {
    forward: function(x, y) { // assume x and y are Units
        this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c);
        return this.unit_out;
    },
    backward: function(label) { // label is +1 or -1

        // reset pulls on a,b,c
        this.a.grad = 0.0;
        this.b.grad = 0.0;
        this.c.grad = 0.0;

        // compute the pull based on what the circuit output was
        var pull = 0.0;
        if(label === 1 && this.unit_out.value < 1) {
            pull = 1; // the score was too low; pull up
        }
        if(label === -1 && this.unit_out.value > -1) {
            pull = -1; // the score was too high for a positive example, pull down
        }
        this.circuit.backward(pull); // writes gradient into x,y,a,b,c

        // add regularization pull for parameters: towards zero and proportional to value
        this.a.grad += -this.a.value;
        this.b.grad += -this.b.value;
    },
    learnFrom: function(x, y, label) {
        this.forward(x, y); // forward pass (set .value in all Units)
        this.backward(label); // backward pass (set .grad in all Units)
        this.parameterUpdate(); // parameters respond to tug
    },
    parameterUpdate: function() {
        var step_size = 0.01;
        this.a.value += step_size * this.a.grad;
        this.b.value += step_size * this.b.grad;
        this.c.value += step_size * this.c.grad;
    }
};

var data = []; var labels = [];
data.push([1.2, 0.7]); labels.push(1);
data.push([-0.3, -0.5]); labels.push(-1);
data.push([3.0, 0.1]); labels.push(1);
data.push([-0.1, -1.0]); labels.push(-1);
data.push([-1.0, 1.1]); labels.push(-1);
data.push([2.1, -3]); labels.push(1);
var svm = new SVM();

// a function that computes the classification accuracy
var evalTrainingAccuracy = function() {
    var num_correct = 0;
    for(var i = 0; i < data.length; i++) {
        var x = new Unit(data[i][0], 0.0);
        var y = new Unit(data[i][1], 0.0);
        var true_label = labels[i];

        // see if the prediction matches the provided label
        var predicted_label = svm.forward(x, y).value > 0 ? 1 : -1;
        if(predicted_label === true_label) {
            num_correct++;
        }
    }
    return num_correct / data.length;
};

// the learning loop
for(var iter = 0; iter < 400; iter++) {
    // pick a random data point
    var i = Math.floor(Math.random() * data.length);
    var x = new Unit(data[i][0], 0.0);
    var y = new Unit(data[i][1], 0.0);
    var label = labels[i];
    svm.learnFrom(x, y, label);

    if(iter % 25 === 0) { // every 25 iterations...
        console.log('training accuracy at iter ' + iter + ': ' + evalTrainingAccuracy());
    }
}

var a = 1, b = -2, c = -1; // initial parameters
for(var iter = 0; iter < 400; iter++) {
    // pick a random data point
    var i = Math.floor(Math.random() * data.length);
    var x = data[i][0];
    var y = data[i][1];
    var label = labels[i];

    // compute pull
    var score = a*x + b*y + c;
    var pull = 0.0;
    if(label === 1 && score < 1) pull = 1;
    if(label === -1 && score > -1) pull = -1;

    // compute gradient and update parameters
    var step_size = 0.01;
    a += step_size * (x * pull - a); // -a is from the regularization
    b += step_size * (y * pull - b); // -b is from the regularization
    c += step_size * (1 * pull);
}

// assume inputs x,y
var x = 1.0, y = -2.0;
var n1 = Math.max(0, a1*x + b1*y + c1); // activation of 1st hidden neuron
var n2 = Math.max(0, a2*x + b2*y + c2); // 2nd neuron
var n3 = Math.max(0, a3*x + b3*y + c3); // 3rd neuron
var score = a4*n1 + b4*n2 + c4*n3 + d4; // the score

// random initial parameters
var a1 = Math.random() - 0.5; // a random number between -0.5 and 0.5
// ... similarly initialize all other parameters to randoms
var b1 = Math.random() - 0.5;
var c1 = Math.random() - 0.5;
var a2 = Math.random() - 0.5;
var b2 = Math.random() - 0.5;
var c2 = Math.random() - 0.5;
var a3 = Math.random() - 0.5;
var b3 = Math.random() - 0.5;
var c3 = Math.random() - 0.5;
var a4 = Math.random() - 0.5;
var b4 = Math.random() - 0.5;
var c4 = Math.random() - 0.5;
var d4 = Math.random() - 0.5;
for(var iter = 0; iter < 400; iter++) {
    // pick a random data point
    var i = Math.floor(Math.random() * data.length);
    var x = data[i][0];
    var y = data[i][1];
    var label = labels[i];

    // compute forward pass
    var n1 = Math.max(0, a1*x + b1*y + c1); // activation of 1st hidden neuron
    var n2 = Math.max(0, a2*x + b2*y + c2); // 2nd neuron
    var n3 = Math.max(0, a3*x + b3*y + c3); // 3rd neuron
    var score = a4*n1 + b4*n2 + c4*n3 + d4; // the score

    // compute the pull on top
    var pull = 0.0;
    if(label === 1 && score < 1) pull = 1; // we want higher output! Pull up.
    if(label === -1 && score > -1) pull = -1; // we want lower output! Pull down.

    // now compute backward pass to all parameters of the model

    // backprop through the last "score" neuron
    var dscore = pull;
    var da4 = n1 * dscore;
    var dn1 = a4 * dscore;
    var db4 = n2 * dscore;
    var dn2 = b4 * dscore;
    var dc4 = n3 * dscore;
    var dn3 = c4 * dscore;
    var dd4 = 1.0 * dscore; // phew

    // backprop the ReLU non-linearities, in place
    // i.e. just set gradients to zero if the neurons did not "fire"
    var dn3 = n3 === 0 ? 0 : dn3;
    var dn2 = n2 === 0 ? 0 : dn2;
    var dn1 = n1 === 0 ? 0 : dn1;

    // backprop to parameters of neuron 1
    var da1 = x * dn1;
    var db1 = y * dn1;
    var dc1 = 1.0 * dn1;

    // backprop to parameters of neuron 2
    var da2 = x * dn1;
    var db2 = y * dn2;
    var dc2 = 1.0 * dn2;

    // backprop to parameters of neuron 3
    var da3 = x * dn3;
    var db3 = y * dn3;
    var dc3 = 1.0 * dn3;

    // phew! End of backprop!
    // note we could have also backpropped into x,y
    // but we do not need these gradients. We only use the gradients
    // on our parameters in the parameter update, and we discard x,y

    // add the pulls from the regularization, tugging all multiplicative
    // parameters (i.e. not the biases) download, proportional to their value
    da1 += -a1; da2 += -a2; da3 += -a3;
    db1 += -b1; db2 += -b2; db3 += -b3;
    da4 += -a4; db4 += -b4; dc4 += -b4;

    // finally, do the parameter update
    var step_size = 0.01;
    a1 += step_size * da1;
    b1 += step_size * db1;
    c1 += step_size * dc1;
    a2 += step_size * da2;
    b2 += step_size * db2;
    c2 += step_size * dc2;
    a3 += step_size * da3;
    b3 += step_size * db3;
    c3 += step_size * dc3;
    a4 += step_size * da4;
    b4 += step_size * db4;
    c4 += step_size * dc4;
    d4 += step_size * dd4;
    // wow this is tedious, please use for loops in prod.
    // we're done!
}

var X = [ [1.2, 0.7], [-0.3, 0.5], [3, 2.5] ] // array of 2-dimensional data
var y = [1, -1, 1] // array of labels
var w = [0.1, 0.2, 0.3] // example: random numbers
var alpha = 0.1; // regularization strength

function cost(X, y, w) {
    // L = (\sum_{i=1}^N\max(0, -y_i(w_0x_0 + w_1x_1 + w_2) + 1)) + \alpha(w_0^2 + w_1^2)
    var total_cost = 0.0 // L, in SVM loss function above
    N = X.length;
    for(var i=0;i<N;i++) {
        // loop over all data points and compute their score
        var xi = X[i];
        var score = w[0] * xi[0] + w[1] * xi[1] + w[2];

        // accumulate cost based on how compatible the score is with the label
        var yi = y[i]; // label
        var costi = Math.max(0, - yi * score + 1);
        console.log('example ' + i + ': xi = (' + xi + ') and label = ' + yi);
        console.log('  score computed to be ' + score.toFixed(3));
        console.log('  => cost computed to be ' + costi.toFixed(3));
        total_cost += costi;
    }

    // regularization cost: we want smaller weights
    reg_cost = alpha * (w[0]*w[0] + w[1]*w[1])
    console.log('regularization cost for current model is ' + reg_cost.toFixed(3));
    total_cost += reg_cost;

    console.log('total cost is ' + total_cost.toFixed(3));
    return total_cost;
}