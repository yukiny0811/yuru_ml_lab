//
//  GPURender.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//

import Metal

class GPURender {
    
    private static let cos7L6: Float = -0.86602540378
    private static let cos11L6: Float = 0.86602540378
    private static let sin7L6: Float = -0.5
    private static let sin11L6: Float = -0.5
    
    
    /// draw triangle particle
    /// - Parameters:
    ///   - renderCommandEncoder: renderCommandEncoder
    ///   - positionX: position x in -1...1
    ///   - positionY: position y in -1...1
    ///   - size: usually works best under 0.1
    ///   - color: (r, g, b) in 0...1
    public static func Run_DrawTriangle(_ renderCommandEncoder: inout MTLRenderCommandEncoder, _ positionX: Float, _ positionY: Float, _ size: Float, _ color: (Float, Float, Float)) {
        
        let renderPositionX = positionX
        let renderPositionY = positionY
        
        let vertex1 = Vertex(position: SIMD3<Float>(renderPositionX, renderPositionY + size, 0), color: SIMD4<Float>(color.0, color.1, color.2, 1))
        let vertex2 = Vertex(position: SIMD3<Float>(renderPositionX + cos7L6 * size, renderPositionY + sin7L6 * size, 0), color: SIMD4<Float>(color.0, color.1, color.2, 1))
        let vertex3 = Vertex(position: SIMD3<Float>(renderPositionX + cos11L6 * size, renderPositionY + sin11L6 * size, 0), color: SIMD4<Float>(color.0, color.1, color.2, 1))
        
        let vertexDatas = [vertex1, vertex2, vertex3]
        GPUCore.RunRender_Default("test_vertex", "test_fragment", &renderCommandEncoder, vertexDatas)
    }
    
    
    /// draw triangle particle
    /// - Parameters:
    ///   - renderCommandEncoder: renderCommandEncoder
    ///   - positionX: position x in 0...1
    ///   - positionY: position y in 0...1
    ///   - size: usually works best under 0.1
    ///   - color: (r, g, b) in 0...1
    public static func Run_DrawTriangle_U(_ renderCommandEncoder: inout MTLRenderCommandEncoder, _ positionX: Float, _ positionY: Float, _ size: Float, color: (Float, Float, Float)) {
        
        let renderPositionX = positionX * 2 - 1
        let renderPositionY = positionY * 2 - 1
        
        let vertex1 = Vertex(position: SIMD3<Float>(renderPositionX, renderPositionY + size, 0), color: SIMD4<Float>(color.0, color.1, color.2, 1))
        let vertex2 = Vertex(position: SIMD3<Float>(renderPositionX + cos7L6 * size, renderPositionY + sin7L6 * size, 0), color: SIMD4<Float>(color.0, color.1, color.2, 1))
        let vertex3 = Vertex(position: SIMD3<Float>(renderPositionX + cos11L6 * size, renderPositionY + sin11L6 * size, 0), color: SIMD4<Float>(color.0, color.1, color.2, 1))
        
        let vertexDatas = [vertex1, vertex2, vertex3]
        GPUCore.RunRender_Default("test_vertex", "test_fragment", &renderCommandEncoder, vertexDatas)
    }
}
