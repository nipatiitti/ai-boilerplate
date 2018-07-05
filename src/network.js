import fs from 'fs'

import { Layer } from './layer'
import { Neuron } from './neuron'
import { rejects } from 'assert';

export class NeuralNetwork {
    constructor(neurons, options) {
        this.options = Object.assign({}, {
            iterations: 5000,
            learningRate: 0.3,
            momentum: 0.9,
            errorMargin: 0.0001,
            neurons: {
                bias: 1,
                momentum: 0.7,
                gradient: 0
            }
        }, options)
        this.layers = []

        this.initialize(neurons)
    }

    results = (input) => {
        return this.input(input)
    }

    initialize = (neurons) => {
        const len = neurons.length
        for(let i = 0; i < len; i++) {
            this.layers[i] = new Layer()
            for(let j = 0; j < neurons[i]; j++) {
                this.layers[i].neurons.push(new Neuron(this.options.neurons))
            }
        }
    }

    input = (input) => {
        let result = input.slice()
        let len = this.layers.length

        for(let i = 0; i < len; i++) {
            result = this.layers[i].parse(result)
        }

        return result
    }
    
    train = (inputs, ideals) => {
        let collectiveError = 1
        let i = 0
        const len = inputs.length

        while(collectiveError > this.options.errorMargin && i < this.options.iterations ) {
            let tempError = 0
            inputs.forEach((input, j) => {
                tempError += this.iterate(input, ideals[j])
            })
            collectiveError = tempError / len
            console.log(i, collectiveError)
            i++
        }
    }
    
    iterate = (input, ideal) => {
        this.input(input)
        let sigErr = 0.0
        let len = this.layers.length

        for(let i = len-1; i >= 0; i--) {
            if(i === len-1) {
                this.layers[i].neurons.forEach((neuron, j) => {
                    let output = neuron.output
                    neuron.gradient = output * (1 - output) * (ideal[j] - output)
                    sigErr += Math.pow((ideal[j] - output), 2)
                })
            }
            else {
                this.layers[i].neurons.forEach((neuron, j) => {
                    let output = neuron.output
                    let error = 0.0
                    this.layers[i+1].neurons.forEach((nextNeuron, k) => {
                        error += nextNeuron.weights[j] * nextNeuron.gradient
                    })
                    neuron.gradient = output * (1 - output) * error
                })
            }
        }

        this.layers.forEach((layer, i) => {
            for(let j = 0; j < this.layers[i].neurons.length; j++) {
                let neuron = this.layers[i].neurons[j]
                neuron.bias += this.options.learningRate * neuron.gradient
                for(let k = 0; k < neuron.weights.length; k++) {
                    neuron.deltas[k] = this.options.learningRate * neuron.gradient * (this.layers[i-1] ? this.layers[i-1].neurons[k].output : input[k])
                    neuron.weights[k] += neuron.deltas[k]
                    neuron.weights[k] += this.options.momentum * neuron.previousDeltas[k]
                }
                neuron.previousDeltas = neuron.deltas.slice()
            }
        })

        return sigErr
    }

    save = (fileUrl) => {
        let toWrite = JSON.stringify(this, null, 2)
        fs.writeFile(fileUrl, toWrite, 'utf8', (e) => {
            console.log(e)
        })
    }

    load = (fileUrl) => {
        return new Promise((resolve, reject) => {
            fs.readFile(fileUrl, 'utf8', (err, data) => {
                if (err){
                    reject(err)
                } else {
                    let net = JSON.parse(data)
                    console.log(net.layers.map(layer => (layer.neurons.length)))
                    let newNet = new NeuralNetwork(net.layers.map(layer => (layer.neurons.length)), net.options)
                    newNet.layers.forEach((layer, i) => {
                        for(let j = 0; j < newNet.layers[i].neurons.length; j++) {
                            let neuron = newNet.layers[i].neurons[j]
                            neuron.bias = net.layers[i].neurons[j].bias
                            neuron.weights =  net.layers[i].neurons[j].weights
                            neuron.input = net.layers[i].neurons[j].input
                            neuron.output = net.layers[i].neurons[j].output
                            neuron.deltas = net.layers[i].neurons[j].deltas
                            neuron.previousDeltas = net.layers[i].neurons[j].previousDeltas
                            neuron.gradient = net.layers[i].neurons[j].gradient
                            neuron.momentum = net.layers[i].neurons[j].momentum
                        }
                    })
                    resolve(newNet)
                }
            })
        })
    }
}


