export class Layer {
    constructor() {
        this.neurons = []
    }

    parse = (input) => {
        var result = []
        this.neurons.forEach((neuron, i) => {
            result[i] = neuron.parse(input)
        })
        return result
    }

}