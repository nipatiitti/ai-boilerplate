export class Neuron {
    constructor (options) {
        this.weights = []
        this.bias = options.bias
        this.input = []
        this.output = 0
        this.deltas = []
        this.previousDeltas = []
        this.gradient = options.gradient
        this.momentum = options.momentum
    }

    parse = (input) => {
        let sum = 0
        let len = input.length
		for(let i = 0; i < len; i++) {
			if(!this.weights[i]) {
				this.weights[i] = Math.floor(Math.random()*(1-(-1)+1)+(-1))
			}
			sum += input[i] * this.weights[i];
		}
		sum += this.bias
        this.input = sum
        this.output = ( 1 / (1 + Math.exp(-1 * sum)))
		return this.output
    }

}