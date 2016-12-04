//
// Created by toby on 01.12.16.
//

#include "chianti/layers.h"
#include <string>
#include <iostream>

int main(int argc, const char** argv)
{
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 2, 2, 3 }, CNTK::DataType::Float);

    CNTK::FunctionPtr network;
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("same")
            .numFilters(64);
}