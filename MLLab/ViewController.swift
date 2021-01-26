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
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { }
    
    var network: NeuralNetwork_Kaiki!
    var count = 0
    
    func draw(in view: MTKView) {
        
        guard let drawable = view.currentDrawable, let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return 
        }
        let commandBuffer = GPUCore.commandQueue.makeCommandBuffer()
        var renderCommandEncoder = commandBuffer?.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        
        count += 1
        print(count)
        network.drawUpdate(renderCommandEncoder: &renderCommandEncoder!)
        
        renderCommandEncoder?.endEncoding()
        commandBuffer?.present(drawable)
        commandBuffer?.commit()
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.view.window?.setFrame(NSMakeRect(0, 0, 1000, 1000), display: false)
        self.view.frame = NSRect(x: 0, y: 0, width: 1000, height: 1000)
        GPUCore.start()
        mtkView.device = GPUCore.device
        mtkView.delegate = self
        
        network = NeuralNetwork_Kaiki()
    }
    
}
