//
//  Matrix.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//

import Metal
import Accelerate


/// Matrix Class
struct Matrix: CustomStringConvertible {
    
    /// matrix object
    private let laObject: la_object_t
    
    /// description of this matrix in string
    var description: String {
        return self.get2DArray().map {
            $0.description
        }.joined(separator: "\n")
    }
    
    public init(_ array: [Float], _ tate: Int, _ yoko: Int) {
        self.laObject = la_matrix_from_float_buffer(array, la_count_t(tate), la_count_t(yoko), la_count_t(yoko), la_hint_t(LA_NO_HINT), la_attribute_t(LA_DEFAULT_ATTRIBUTES))
    }
    
    public init(_ laObject: la_object_t) {
        self.laObject = laObject
    }
    
    /// row count of matrix
    public var row: Int {
        get {
            return Int(la_matrix_rows(self.laObject))
        }
    }
    
    /// col count of matrix
    public var col: Int {
        get {
            return Int(la_matrix_cols(self.laObject))
        }
    }
    
    /// one-dimensional array of matrix
    public var array: [Float] {
        get {
            var array: [Float] = [Float](repeating: 0, count: self.row * self.col)
            la_matrix_to_float_buffer(&array, la_count_t(self.col), self.laObject)
            return array
        }
    }
    
    subscript(row: Int, col: Int) -> Float {
        return self.array[row * self.col + col]
    }
    
    /// two-dimensional array of matrix
    public func get2DArray() -> [[Float]] {
        var result: [[Float]] = []
        for i in 0..<self.row {
            var temp: [Float] = []
            for j in 0..<self.col {
                temp.append(self[i, j])
            }
            result.append(temp)
        }
        return result
    }
    
    /// sum of matrix
    public func sum(_ other: Matrix) -> Matrix {
        let obj = la_sum(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    /// difference of matrix
    public func subtract(_ other: Matrix) -> Matrix {
        let obj = la_difference(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    /// multiplication of matrix and scalar
    public func multiplyScalar(_ scalar: Float) -> Matrix {
        let obj = la_scale_with_float(self.laObject, scalar)
        return Matrix(obj)
    }
    
    /// multiplication of each elements within two matrix
    public func elementwiseProduct(_ other: Matrix) -> Matrix {
        let obj = la_elementwise_product(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    /// matrix product of two matrix
    public func product(_ other: Matrix) -> Matrix {
        let obj = la_matrix_product(self.laObject, other.laObject)
        return Matrix(obj)
    }
    
    /// transposed matrix
    public func transpose() -> Matrix {
        let obj = la_transpose(self.laObject)
        return Matrix(obj)
    }
    
    /// sum all the elements in each row, and create col1 matrix
    public func sumOfAllElementsInRow() -> Matrix {
        let array = self.get2DArray()
        var result: [Float] = []
        for a in 0..<array.count{
            result.append(array[a].reduce(0, +))
        }
        return Matrix(result, result.count, 1)
    }
    
    /// sum of all elements in matrix
    public func sumOfAllElements() -> Float {
        return self.array.reduce(0, +)
    }
    
    /// exp of each element in matrix
    public func expAllElements() -> Matrix {
        return Calc.exp(x: self)
    }
    
    /// copy of matrix
    public func copy() -> Matrix {
        let mat = self
        return mat
    }
    
    /// deepcopy (well, not exactly...) of matrix
    public func deepcopy() -> Matrix {
        return Matrix(self.array, self.row, self.col)
    }
    
    
    /* ----- static functions ----- */
    
    /// sum of matrix
    public static func sum(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_sum(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    /// difference of matrix
    public static func subtract(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_difference(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    /// multiplication of scalar and matrix
    public static func multiplyScalar(_ m1: Matrix, _ s: Float) -> Matrix {
        let obj = la_scale_with_float(m1.laObject, s)
        return Matrix(obj)
    }
    
    /// multiplication of each elements within two matrix
    public static func elementwiseProduct(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_elementwise_product(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    /// matrix product of two matrix
    public static func product(_ m1: Matrix, _ m2: Matrix) -> Matrix {
        let obj = la_matrix_product(m1.laObject, m2.laObject)
        return Matrix(obj)
    }
    
    /// transpose matrix
    public static func transpose(_ m: Matrix) -> Matrix {
        let obj = la_transpose(m.laObject)
        return Matrix(obj)
    }
    
    /// create matrix of zeros
    public static func createZeros(_ tate: Int, _ yoko: Int) -> Matrix {
        return Matrix([Float](repeating: 0, count: tate * yoko), tate, yoko)
    }
    
    
    ///create matrix of ones
    public static func createOnes(_ tate: Int, _ yoko: Int) -> Matrix {
        return Matrix([Float](repeating: 1, count: tate * yoko), tate, yoko)
    }
}
