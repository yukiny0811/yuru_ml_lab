//
//  NeuralNetworkBase.swift
//  MLLab
//
//  Created by クワシマ・ユウキ on 2021/01/26.
//
import Metal

protocol NeuralNetworkBase {
    func drawUpdate(renderCommandEncoder: inout MTLRenderCommandEncoder)
}
