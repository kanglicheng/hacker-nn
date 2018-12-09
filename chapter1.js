var forwardMultiplyGate = function(x, y) {
    return x * y;
};
forwardMultiplyGate(-2, 3); // returns -6. Exciting.

// circuit with single gate for now
var forwardMultiplyGate = function(x, y) { return x * y; };
var x = -2, y = 3; // some input values

// try changing x,y randomly small amounts and keep track of what works best
var tweak_amount = 0.01;
var best_out = -Infinity;
var best_x = x, best_y = y;
for(var k = 0; k < 100; k++) {
    var x_try = x + tweak_amount * (Math.random() * 2 - 1); // tweak x a bit
    var y_try = y + tweak_amount * (Math.random() * 2 - 1); // tweak y a bit
    var out = forwardMultiplyGate(x_try, y_try);
    if(out > best_out) {
        // best improvement yet! Keep track of the x and y
        best_out = out;
        best_x = x_try, best_y = y_try;
    }
}

var x = -2, y = 3;
var out = forwardMultiplyGate(x, y); // -6
var h = 0.0001;

// compute derivative with respect to x
var xph = x + h; // -1.9999
var out2 = forwardMultiplyGate(xph, y); // -5.9997
var x_derivative = (out2 - out) / h; // 3.0

// compute derivative with respect to y
var yph = y + h; // 3.0001
var out3 = forwardMultiplyGate(x, yph); // -6.0002
var y_derivative = (out3 - out) / h; // -2.0

var step_size = 0.01;
var out = forwardMultiplyGate(x, y); // before: -6
x = x + step_size * x_derivative; // x becomes -1.97
y = y + step_size * y_derivative; // y becomes 2.98
var out_new = forwardMultiplyGate(x, y); // -5.87! exciting.

var x = -2, y = 3;
var out = forwardMultiplyGate(x, y); // before: -6
var x_gradient = y; // by our complex mathematical derivation above
var y_gradient = x;

var step_size = 0.01;
x += step_size * x_gradient; // -1.97
y += step_size * y_gradient; // 2.98
var out_new = forwardMultiplyGate(x, y); // -5.87. Higher output! Nice.

var forwardMultiplyGate = function(a, b) {
    return a * b;
};
var forwardAddGate = function(a, b) {
    return a + b;
};
var forwardCircuit = function(x,y,z) {
    var q = forwardAddGate(x, y);
    var f = forwardMultiplyGate(q, z);
    return f;
};

var x = -2, y = 5, z = -4;
var f = forwardCircuit(x, y, z); // output is -12

// initial conditions
var x = -2, y = 5, z = -4;
var q = forwardAddGate(x, y); // q is 3
var f = forwardMultiplyGate(q, z); // f is -12

// gradient of the MULTIPLY gate with respect to its inputs
// wrt is short for "with respect to"
var derivative_f_wrt_z = q; // 3
var derivative_f_wrt_q = z; // -4

// derivative of the ADD gate with respect to its inputs
var derivative_q_wrt_x = 1.0;
var derivative_q_wrt_y = 1.0;

// chain rule
var derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q; // -4
var derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q; // -4

// final gradient, from above: [-4, -4, 3]
var gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

// let the inputs respond to the force/tug:
var step_size = 0.01;
x = x + step_size * derivative_f_wrt_x; // -2.04
y = y + step_size * derivative_f_wrt_y; // 4.96
z = z + step_size * derivative_f_wrt_z; // -3.97

// Our circuit now better give higher output:
var q = forwardAddGate(x, y); // q becomes 2.92
var f = forwardMultiplyGate(q, z); // output is -11.59, up from -12! Nice!

// initial conditions
var x = -2, y = 5, z = -4;

// numerical gradient check
var h = 0.0001;
var x_derivative = (forwardCircuit(x+h,y,z) - forwardCircuit(x,y,z)) / h; // -4
var y_derivative = (forwardCircuit(x,y+h,z) - forwardCircuit(x,y,z)) / h; // -4
var z_derivative = (forwardCircuit(x,y,z+h) - forwardCircuit(x,y,z)) / h; // 3

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

var sigmoidGate = function() {
	// helper function
	this.sig = function(x) { return 1 / (1 + Math.exp(-x)); };
};
sigmoidGate.prototype = {
    forward: function(u0) {
        this.u0 = u0;
        this.utop = new Unit(this.sig(this.u0.value), 0.0);
        return this.utop;
    },
    backward: function() {
        var s = this.sig(this.u0.value);
        this.u0.grad += (s * (1 - s)) * this.utop.grad;
    }
}

// create input units
var a = new Unit(1.0, 0.0);
var b = new Unit(2.0, 0.0);
var c = new Unit(-3.0, 0.0);
var x = new Unit(-1.0, 0.0);
var y = new Unit(3.0, 0.0);

// create the gates
var mulg0 = new multiplyGate();
var mulg1 = new multiplyGate();
var addg0 = new addGate();
var addg1 = new addGate();
var sg0 = new sigmoidGate();

// do the forward pass
var forwardNeuron = function() {
    ax = mulg0.forward(a, x); // a*x = -1
    by = mulg1.forward(b, y); // b*y = 6
    axpby = addg0.forward(ax, by); // a*x + b*y = 5
    axpbypc = addg1.forward(axpby, c); // a*x + b*y + c = 2
    s = sg0.forward(axpbypc); // sig(a*x + b*y + c) = 0.8808
};
forwardNeuron();

console.log('circuit output: ' + s.value); // prints 0.8808

s.grad = 1.0;
sg0.backward(); // writes gradient into axpbypc
addg1.backward(); // writes gradient into axpby and c
addg0.backward(); // writes gradient into ax and by
mulg1.backward(); // writes gradient into b and y
mulg0.backward(); // writes gradient into a and x

var step_size = 0.01;
a.value += step_size * a.grad; // a.grad is -0.105
b.value += step_size * b.grad; // b.grad is 0.315
c.value += step_size * c.grad; // c.grad is 0.105
x.value += step_size * x.grad; // x.grad is 0.105
y.value += step_size * y.grad; // y.grad is 0.210

forwardNeuron();
console.log('circuit output after one backprop: ' + s.value); // prints 0.8825

var forwardCircuitFast = function(a,b,c,x,y) {
    return 1/(1 + Math.exp( - (a*x + b*y + c)));
};
var a = 1, b = 2, c = -3, x = -1, y = 3;
var h = 0.0001;
var a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h;
var y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h;

var x = a * b;
// and given gradient on x (dx), we saw that in backprop we would compute:
var dx = 1;
var da = b * dx;
var db = a * dx;

var x = a + b;
// ->
var da = 1.0 * dx;
var db = 1.0 * dx;

// lets compute x = a + b + c in two steps:
var q = a + b; // gate 1
var x = q + c; // gate 2

// backward pass:
dc = 1.0 * dx; // backprop gate 2
dq = 1.0 * dx;
da = 1.0 * dq; // backprop gate 1
db = 1.0 * dq;

var x = a + b + c;
var da = 1.0 * dx; var db = 1.0 * dx; var dc = 1.0 * dx;

var x = a * b + c;
// given dx, backprop in-one-sweep would be =>
da = b * dx;
db = a * dx;
dc = 1.0 * dx;

// lets do our neuron in two steps:
var q = a*x + b*y + c;
var sig = function(x) {
    return 1 / (1 + Math.exp(-x));
};
var f = sig(q); // sig is the sigmoid function
// and now backward pass, we are given df, and:
var df = 1;
var dq = (f * (1 - f)) * df;
// and now we chain it to the inputs
var da = x * dq;
var dx = a * dq;
var dy = b * dq;
var db = y * dq;
var dc = 1.0 * dq;

var x = a * a;
var da = a * dx; // gradient into a from first branch
da += a * dx; // and add on the gradient from the second branch

// short form instead is:
var da = 2 * a * dx;

var x = a*a + b*b + c*c;
// we get:
var da = 2*a*dx;
var db = 2*b*dx;
var dc = 2*c*dx;

var d = 1;
var x = Math.pow(((a * b + c) * d), 2) // pow(x,2) squares the input JS

var x1 = a * b + c;
var x2 = x1 * d;
var x = x2 * x2; // this is identical to the above expression for x
// and now in backprop we go backwards:
var dx2 = 2 * x2 * dx; // backprop into x2
var dd = x1 * dx2; // backprop into d
var dx1 = d * dx2; // backprop into x1
var da = b * dx1;
var db = a * dx1;
var dc = 1.0 * dx1; // done!

var x = 1.0/a; // division
var da = -1.0/(a*a);

var x = (a + b)/(c + d);
// lets decompose it in steps:
var x1 = a + b;
var x2 = c + d;
var x3 = 1.0 / x2;
var x = x1 * x3; // equivalent to above
// and now backprop, again in reverse order:
var dx1 = x3 * dx;
var dx3 = x1 * dx;
var dx2 = (-1.0/(x2*x2)) * dx3; // local gradient as show above, and chain rule
var da = 1.0 * dx1; // and finally into the original variables
var db = 1.0 * dx1;
var dc = 1.0 * dx2;
var dd = 1.0 * dx2;

var x = Math.max(a, b);
var da = a === x ? 1.0 * dx : 0.0;
var db = b === x ? 1.0 * dx : 0.0;

var x = Math.max(a, 0)
// backprop through this gate will then be:
var da = a > 0 ? 1.0 * dx : 0.0;
