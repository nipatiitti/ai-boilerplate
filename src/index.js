import { NeuralNetwork } from './network'

let net = new NeuralNetwork([2, 3, 1], {
    iterations: 10000,
    learningRate: 0.3,
    momentum: 0.9,
    errorMargin: 0.0001,
    neurons: {
        bias: 1,
        momentum: 0.7,
        gradient: 0.3
    }
})

net.train(
    [[0,0], [0,1], [1,0], [1,1]],
    [[0], [1], [1], [0]]
)

net.load('./network.json')
    .then(newNet => {
        console.log(newNet.results([0,0]))
        console.log(newNet.results([1,0]))
        console.log(newNet.results([0,1]))
        console.log(newNet.results([1,1]))
    })
    .catch(e => {
        console.error(e)
    })
