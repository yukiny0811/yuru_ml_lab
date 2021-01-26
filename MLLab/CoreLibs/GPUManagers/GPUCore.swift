//
//  GPUCore.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//

import Metal
import simd

class GPUCore {
    
    public static var device: MTLDevice!
    public static var library: MTLLibrary!
    public static var commandQueue: MTLCommandQueue!
    
    public static var computePipelineStates: [String: MTLComputePipelineState] = [:]
    
    public static func start() {
        self.device = MTLCreateSystemDefaultDevice()!
        let frameworkBundle = Bundle(for: GPUCore.self)
        library = try! GPUCore.device.makeDefaultLibrary(bundle: frameworkBundle)
        commandQueue = self.device.makeCommandQueue()!
        
        addComputePipelineState(name: "exp_of_matrix")
        addComputePipelineState(name: "scalar_divided_by_matrix")
    }
    
    private static func addComputePipelineState(name: String) {
        computePipelineStates[name] = try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    }
    
    public static func RunCompute_Default(_ functionName: String, inputDatas: [[Float]], row: Int, col: Int) -> Matrix {
        
        return autoreleasepool {
            
            let outputData: [Float] = autoreleasepool {
                [Float](repeating: 0, count: row * col)
            }
            
            let commandBuffer: MTLCommandBuffer! = autoreleasepool {
                commandQueue.makeCommandBuffer()
            }
            let computeCommandEncoder: MTLComputeCommandEncoder! = autoreleasepool {
                commandBuffer.makeComputeCommandEncoder()
            }
            
            autoreleasepool {
                computeCommandEncoder.setComputePipelineState(computePipelineStates[functionName]!)
            }
            
            let inputBuffers: [MTLBuffer] = autoreleasepool {
                var inputBuffers: [MTLBuffer] = []
                for i in 0..<inputDatas.count {
                    let tempt: MTLBuffer = autoreleasepool {
                        GPUCore.device.makeBuffer(bytes: inputDatas[i], length: MemoryLayout<Float>.size * inputDatas[i].count, options: [])!
                    }
                    inputBuffers.append(tempt)
                }
                return inputBuffers
            }
            
            let outputBuffer = autoreleasepool {
                return GPUCore.device.makeBuffer(bytes: outputData, length: MemoryLayout<Float>.size * outputData.count, options: [])
            }
            
            autoreleasepool {
                for i in 0..<inputBuffers.count {
                    computeCommandEncoder.setBuffer(inputBuffers[i], offset: 0, index: i)
                }
                computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: inputBuffers.count)
            }
            
            let width = autoreleasepool {
                computePipelineStates[functionName]!.threadExecutionWidth
            }
            let threadGroupsPerGrid = autoreleasepool {
                MTLSize(width: (outputData.count + width - 1) / width, height: 1, depth: 1)
            }
            let threadsPerThreadGroup = autoreleasepool {
                MTLSize(width: width, height: 1, depth: 1)
            }
            autoreleasepool {
                computeCommandEncoder.dispatchThreadgroups(threadGroupsPerGrid, threadsPerThreadgroup: threadsPerThreadGroup)
            }
            
            autoreleasepool {
                computeCommandEncoder.endEncoding()
            }
            
            autoreleasepool {
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            }
            
            
            let resultData = autoreleasepool {
                Data(bytesNoCopy: outputBuffer!.contents(), count: MemoryLayout<Float>.size * outputData.count, deallocator: .none)
            }
            let newOutputData = autoreleasepool {
                resultData.withUnsafeBytes {
                    Array(
                        UnsafeBufferPointer(
                            start: $0.baseAddress!.assumingMemoryBound(to: Float.self),
                            count: $0.count / MemoryLayout<Float>.size
                        )
                    )
                }
            }
            
            return Matrix(newOutputData, row, col)
        }
    }
    
    public static func RunRender_Default(_ vertexFunctionName: String, _ fragmentFunctionName: String, _ renderCommandEncoder: inout MTLRenderCommandEncoder, _ vertexDatas: [Vertex]) {
        
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        let vertexFunction = GPUCore.library.makeFunction(name: vertexFunctionName)
        renderPipelineDescriptor.vertexFunction = vertexFunction
        
        let fragmentFunction = GPUCore.library.makeFunction(name: fragmentFunctionName)
        renderPipelineDescriptor.fragmentFunction = fragmentFunction
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[0].offset = 0
        
        vertexDescriptor.attributes[1].format = .float4
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.size
        
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
        
        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
        
        let renderPipelineState = try! GPUCore.device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        
        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
        
        let vertexBuffer = GPUCore.device.makeBuffer(bytes: vertexDatas, length: MemoryLayout<Vertex>.stride * vertexDatas.count, options: [])
        
        renderCommandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderCommandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: vertexDatas.count)
    }
}

struct Vertex {
    var position: SIMD3<Float>
    var color: SIMD4<Float>
}
