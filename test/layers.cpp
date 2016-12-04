#include <gtest/gtest.h>
#include "chianti/chianti.h"


TEST(Conv2DLayer, pad_shape_same)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("same")
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(5, outputShape[0]);
    ASSERT_EQ(5, outputShape[1]);
}

TEST(Conv2DLayer, pad_shape_full)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("full")
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(9, outputShape[0]);
    ASSERT_EQ(9, outputShape[1]);
}

TEST(Conv2DLayer, pad_shape_valid)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(Conv2DLayer, pad_shape_0)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({0, 0})
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(Conv2DLayer, pad_shape_1)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({5, 5})
            .pad({1, 1})
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(Conv2DLayer, pad_shape_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({5, 5})
            .pad({2, 2})
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(5, outputShape[0]);
    ASSERT_EQ(5, outputShape[1]);
}

