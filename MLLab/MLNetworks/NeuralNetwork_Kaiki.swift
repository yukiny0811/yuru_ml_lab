//
//  NeuralNetwork_Kaiki.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//

import Foundation

class NeuralNetwork_Kaiki {
    
    let saveData = UserDefaults.standard
    
    let lr: Float = 0.1
    let epoch = 2000
    
    let inputData = Calc.createFloatData(0, Float.pi * 2, 0.1)
    lazy var correctData = inputData.map {
        sin($0)
    }
    lazy var normalizedInputData = inputData.map {
        ($0 - Float.pi) / Float.pi
    }
    
    let midLayer: Kaiki_MiddleLayer
    let outLayer: Kaiki_OutputLayer
    
    public init() {
        
        midLayer = Kaiki_MiddleLayer(upperLayerCount: 1, thisLayerCount: 3)
        outLayer = Kaiki_OutputLayer(upperLayerCount: 3, thisLayerCount: 1)
        
    }
    
    public func train() -> (input: [Float], result: [Float], correct: [Float]) {
        return autoreleasepool {
            var plotX: [Float] = []
            var plotY: [Float] = []
            var plotC: [Float] = []

            for i in 0..<correctData.count{


                /// this is important
//                let x = normalizedInputData[i]
//                let c = correctData[i]
                let tempX = Float.random(in: 0..<Float.pi * 2)
                let x = (tempX - Float.pi) / Float.pi
                let c = sin(tempX)
//
                let tempInput = Matrix([x], 1, 1)
                let tempCorrect = Matrix([c], 1, 1)
//
                midLayer.forward(input: tempInput)
                outLayer.forward(input: midLayer.output)
////
                outLayer.back(correct: tempCorrect)
                midLayer.back(gradOutput: outLayer.gradInput)
//
                autoreleasepool {
                    midLayer.update(lr: lr)
                }

                outLayer.update(lr: lr)
                let res = outLayer.output.array[0]
                plotX.append(x)
                plotY.append(res)
                plotC.append(c)
            }
            return (plotX, plotY, plotC)
        }
    }
    
    public func staticTrain() {
        for k in 0..<epoch {
            autoreleasepool {
                var totalError: Float = 0
                var plotX: [Float] = []
                var plotY: [Float] = []

                for i in 0..<correctData.count{


                        ///this is important
                    let x = normalizedInputData[i]
                    let c = correctData[i]
                    
                    
                    
//
                    let tempInput = Matrix([x], 1, 1)
                    let tempCorrect = Matrix([c], 1, 1)
//
                    midLayer.forward(input: tempInput)
                    outLayer.forward(input: midLayer.output)
////
                    outLayer.back(correct: tempCorrect)
                    midLayer.back(gradOutput: outLayer.gradInput)
//
                    autoreleasepool {
                        midLayer.update(lr: lr)
                    }

                    outLayer.update(lr: lr)
                    
                    
                    if k % 200 == 0{
                        let res = outLayer.output.array[0]
                        totalError += 1.0/2.0 * sqrt(res - correctData[i])
                        plotX.append(x)
                        plotY.append(res)
                        print(plotX)
                        print(plotY)
                        print(totalError)
                        print(k)
                    }


                }
            
                print(k)
            }
            
            
        }
        let arrayDataRaw: [[Float]] = [midLayer.weight.array, midLayer.bias.array, outLayer.weight.array, outLayer.bias.array]
        let rowColDataRaw: [[Float]] = [
            [Float(midLayer.weight.row), Float(midLayer.weight.col)],
            [Float(midLayer.bias.row), Float(midLayer.bias.col)],
            [Float(outLayer.weight.row), Float(outLayer.weight.col)],
            [Float(outLayer.bias.row), Float(outLayer.bias.col)]
        ]
        let data = try! JSONEncoder().encode([arrayDataRaw, rowColDataRaw])
        saveData.set(data, forKey: "kaiki_model")
        print("finished")
    }
    
    
}

class Kaiki_MiddleLayer {
    var weight: Matrix!
    var bias: Matrix!
    var input: Matrix!
    
    var gradW: Matrix!
    var gradB: Matrix!
    var gradInput: Matrix!
    
    var output: Matrix!
    
    public init(upperLayerCount: Int, thisLayerCount: Int){
        autoreleasepool{
            weight = autoreleasepool {
                Matrix(Calc.createRandomData(-0.5, 0.5, upperLayerCount * thisLayerCount), upperLayerCount, thisLayerCount)
            }
            bias = autoreleasepool {
                Matrix(Calc.createRandomData(-0.5, 0.5, thisLayerCount), 1, thisLayerCount)
            }
        }
        
    }
    
    public func forward(input: Matrix){
        self.input = input.copy()
        let u = input.product(weight).sum(bias)
        self.output = Calc.sigmoid(x: u)
    }
    
    public func back(gradOutput: Matrix){
        let ones = Matrix.createOnes(self.output.row, self.output.col)
        let temp = ones.subtract(self.output)
        let delta = gradOutput.elementwiseProduct(temp).elementwiseProduct(self.output)
        
        self.gradW = input.transpose().product(delta)
        self.gradB = delta.copy()
        self.gradInput = delta.product(self.weight.transpose())
    }
    
    public func update(lr: Float) {
        autoreleasepool {
            let mulW = autoreleasepool{
                self.gradW.multiplyScalar(lr)
            }
            let mulB = autoreleasepool {
                self.gradB.multiplyScalar(lr)
            }
//
            self.weight = autoreleasepool {
                self.weight.subtract(mulW).deepcopy()
            }
//
            self.bias = autoreleasepool {
                self.bias.subtract(mulB).deepcopy()
            }
        }
        
    }
}

class Kaiki_OutputLayer {
    
    var weight: Matrix
    var bias: Matrix
    var input: Matrix!
    
    var gradW: Matrix!
    var gradB: Matrix!
    var gradInput: Matrix!
    
    var output: Matrix!
    
    public init(upperLayerCount: Int, thisLayerCount: Int){
        
        weight = autoreleasepool {
            Matrix(Calc.createRandomData(-0.5, 0.5, upperLayerCount * thisLayerCount), upperLayerCount, thisLayerCount)
        }
        bias = autoreleasepool {
            Matrix(Calc.createRandomData(-0.5, 0.5, thisLayerCount), 1, thisLayerCount)
        }
    }
    
    public func forward(input: Matrix){
        self.input = input.copy()
        self.output = input.product(weight).sum(bias)
        
    }
    
    public func back(correct: Matrix){
        let delta = self.output.subtract(correct)
        self.gradW = input.transpose().product(delta)
        self.gradB = delta.copy()
        self.gradInput = delta.product(self.weight.transpose())
        
//        print(gradW.array)
    }
    
    public func update(lr: Float) {
        self.weight = autoreleasepool {
            self.weight.subtract(self.gradW.multiplyScalar(lr)).deepcopy()
        }
        self.bias = autoreleasepool {
            self.bias.subtract(self.gradB.multiplyScalar(lr)).deepcopy()
        }
    }
}
