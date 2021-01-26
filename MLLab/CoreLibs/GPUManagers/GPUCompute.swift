//
//  GPUCompute.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//

import Metal

class GPUCompute {
    
    public static func Run_ExpOfMatrix(_ matrix: Matrix) -> Matrix {
        let temp = autoreleasepool{
            GPUCore.RunCompute_Default("exp_of_matrix", inputDatas: [matrix.array], row: matrix.row, col: matrix.col)
        }
        return temp
    }

    public static func Run_ScalarDividedByMatrix(_ matrix: Matrix, scalar: Float) -> Matrix {
        let temp = autoreleasepool{
            GPUCore.RunCompute_Default("scalar_divided_by_matrix", inputDatas: [matrix.array, [scalar]], row: matrix.row, col: matrix.col)
        }
        return temp
    }
}
