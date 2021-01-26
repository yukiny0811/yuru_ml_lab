//
//  ViewController.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//

import Cocoa
import MetalKit
import simd

class ViewController: NSViewController, MTKViewDelegate {
    
    @IBOutlet private weak var mtkView: MTKView!
    
    let saveData = UserDefaults.standard
    let fileManager = FileManager.default
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        
    }
    
    var nn: NeuralNetwork_Kaiki!
    var count = 0
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable, let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return 
        }
        
        let commandBuffer = GPUCore.commandQueue.makeCommandBuffer()
        var renderCommandEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        
        
        
        
//        test(renderCommandEncoder: &renderCommandEncoder!)
//        test2(renderCommandEncoder: &renderCommandEncoder!)
        
        
        let (input, result, correct) = nn.train()
        count += 1
        print(count)

        for i in 0..<input.count {
//            print(result[i])
            GPURender.Run_DrawTriangle(&renderCommandEncoder!, input[i], result[i], 0.015, (1, 0, 0.5))
            GPURender.Run_DrawTriangle(&renderCommandEncoder!, input[i], correct[i], 0.015, (0, 1, 0.5))
        }
        
        
        
        
        
        
        
        
        renderCommandEncoder?.endEncoding()
        
        commandBuffer?.present(drawable)
        commandBuffer?.commit()
        
        print("test")
    }
    
//    func test(renderCommandEncoder: inout MTLRenderCommandEncoder) {
//        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
//        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
//
//        let vertexFunction = GPUCore.library.makeFunction(name: "test_vertex")
//        renderPipelineDescriptor.vertexFunction = vertexFunction
//
//        let fragmentFunction = GPUCore.library.makeFunction(name: "test_fragment")
//        renderPipelineDescriptor.fragmentFunction = fragmentFunction
//
//        let vertexDescriptor = MTLVertexDescriptor()
//        vertexDescriptor.attributes[0].format = .float3
//        vertexDescriptor.attributes[0].bufferIndex = 0
//        vertexDescriptor.attributes[0].offset = 0
//
//        vertexDescriptor.attributes[1].format = .float4
//        vertexDescriptor.attributes[1].bufferIndex = 0
//        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.size
//
//        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
//
//        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
//
//        let renderPipelineState = try! GPUCore.device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
//
//        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
//        let vertexBuffer = GPUCore.device.makeBuffer(bytes: [
//            Vertex(position: SIMD3<Float>(0, 1, 0), color: SIMD4<Float>(1, 0, 0, 1)),
//            Vertex(position: SIMD3<Float>(-1, -1, 0), color: SIMD4<Float>(0, 1, 0, 1)),
//            Vertex(position: SIMD3<Float>(1, -1, 0), color: SIMD4<Float>(0, 0, 1, 1))
//        ], length: MemoryLayout<Vertex>.stride * 3, options: [])
//        renderCommandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
//        renderCommandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
//    }
//
//    func test2(renderCommandEncoder: inout MTLRenderCommandEncoder) {
//        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
//        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
//
//        let vertexFunction = GPUCore.library.makeFunction(name: "test_vertex")
//        renderPipelineDescriptor.vertexFunction = vertexFunction
//
//        let fragmentFunction = GPUCore.library.makeFunction(name: "test_fragment")
//        renderPipelineDescriptor.fragmentFunction = fragmentFunction
//
//        let vertexDescriptor = MTLVertexDescriptor()
//        vertexDescriptor.attributes[0].format = .float3
//        vertexDescriptor.attributes[0].bufferIndex = 0
//        vertexDescriptor.attributes[0].offset = 0
//
//        vertexDescriptor.attributes[1].format = .float4
//        vertexDescriptor.attributes[1].bufferIndex = 0
//        vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.size
//
//        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
//
//        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
//
//        let renderPipelineState = try! GPUCore.device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
//
//        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
//        let vertexBuffer = GPUCore.device.makeBuffer(bytes: [
//            Vertex(position: SIMD3<Float>(0, 0.3, 0), color: SIMD4<Float>(1, 0, 0, 1)),
//            Vertex(position: SIMD3<Float>(-1, -1, 0), color: SIMD4<Float>(0, 1, 0, 1)),
//            Vertex(position: SIMD3<Float>(1, -1, 0), color: SIMD4<Float>(0, 0, 1, 1))
//        ], length: MemoryLayout<Vertex>.stride * 3, options: [])
//        renderCommandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
//        renderCommandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
//    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.view.window?.setFrame(NSMakeRect(0, 0, 1000, 1000), display: false)
        self.view.frame = NSRect(x: 0, y: 0, width: 1000, height: 1000)
        
//        self.mtkView.frame = NSRect(x: 0, y: 0, width: 1000, height: 1000)
        
        GPUCore.start()
        
        mtkView.device = GPUCore.device
        mtkView.delegate = self
        
        nn = NeuralNetwork_Kaiki()
        
//        nn.staticTrain()
//        self.mtkView.drawableSize = CGSize(width: 1000, height: 1000)
        
        
//        NeuralNetwork_Kaiki()
        
//        let data = saveData.object(forKey: "kaiki_model") as! Data
//        let stored = try! JSONDecoder().decode([[[Float]]].self, from: data)
//        print(stored)
//
//        let mid = Kaiki_MiddleLayer(upperLayerCount: 1, thisLayerCount: 3)
//        let out = Kaiki_OutputLayer(upperLayerCount: 3, thisLayerCount: 1)
//
//        mid.weight = Matrix(stored[0][0], Int(stored[1][0][0]), Int(stored[1][0][1]))
//        mid.bias = Matrix(stored[0][1], Int(stored[1][1][0]), Int(stored[1][1][1]))
//        out.weight = Matrix(stored[0][2], Int(stored[1][2][0]), Int(stored[1][2][1]))
//        out.bias = Matrix(stored[0][3], Int(stored[1][3][0]), Int(stored[1][3][1]))
//
//        let input = Matrix([0.4], 1, 1)
//
//        mid.forward(input: input)
//        out.forward(input: mid.output)
//        print(out.output.array)
//
//        let path = fileManager.urls(for: .documentDirectory, in: .userDomainMask)
//        let url = path[0].appendingPathComponent("jsonFiles")
//        try! fileManager.createDirectory(at: url, withIntermediateDirectories: true, attributes: nil)
//        let jsonUrl = url.appendingPathComponent("kaiki_model.json")
//        do {
//            try data.write(to: jsonUrl)
//        } catch {
//            print("error")
//        }
//
//        print(jsonUrl)
    }
    
}
