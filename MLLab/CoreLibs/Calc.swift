//
//  Calc.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//

import Foundation

/// static class for calculation
class Calc {
    
    /// create float array
    public static func createFloatData(_ startFloat: Float, _ endFloat: Float, _ interval: Float) -> [Float] {
        var resultData: [Float] = []
        for i in stride(from: startFloat, to: endFloat, by: interval){
            resultData.append(i)
        }
        return resultData
    }
    
    /// create random float array
    public static func createRandomData(_ from: Float, _ to: Float, _ count: Int) -> [Float] {
        var resultData: [Float] = []
        for _ in 0..<count{
            resultData.append(Float.random(in: from...to))
        }
        return resultData
    }
    
    /// sigmoid of each element in matrix
    public static func sigmoid(x: Matrix) -> Matrix {
        
        let expFinishedMatrix = autoreleasepool {
            GPUCompute.Run_ExpOfMatrix(x.multiplyScalar(-1))
        }
        let ones = autoreleasepool {
            Matrix.createOnes(expFinishedMatrix.row, expFinishedMatrix.col)
        }
        let bunbo = autoreleasepool {
            expFinishedMatrix.sum(ones)
        }
        let result = autoreleasepool {
            GPUCompute.Run_ScalarDividedByMatrix(bunbo, scalar: 1)
        }
        return result
    }
    
    /// exp of each element in matrix
    public static func exp(x: Matrix) -> Matrix {
        return GPUCompute.Run_ExpOfMatrix(x)
    }
}
