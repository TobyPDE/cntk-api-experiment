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
            .stride(1)
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
            .stride(1)
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
            .stride(1)
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
            .stride(1)
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
            .filterSize({3, 3})
            .pad({1, 1})
            .stride(1)
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(5, outputShape[0]);
    ASSERT_EQ(5, outputShape[1]);
}

TEST(Conv2DLayer, pad_shape_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({2, 2})
            .stride(1)
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(7, outputShape[0]);
    ASSERT_EQ(7, outputShape[1]);
}

TEST(Conv2DLayer, pad_shape_3)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({3, 3})
            .stride(1)
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(9, outputShape[0]);
    ASSERT_EQ(9, outputShape[1]);
}

TEST(Conv2DLayer, stride_shape_1)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("same")
            .stride(1)
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(5, outputShape[0]);
    ASSERT_EQ(5, outputShape[1]);
}

TEST(Conv2DLayer, stride_shape_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({1, 1})
            .stride(2)
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(Conv2DLayer, stride_shape_2_1)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({1, 1})
            .stride({2, 1})
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(5, outputShape[1]);
}

TEST(Conv2DLayer, stride_shape_1_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({1, 1})
            .stride({1, 2})
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(5, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(Conv2DLayer, stride_pool)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::CPUDevice();
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({2, 2})
            .pad({0, 0})
            .stride(2)
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(4, outputShape[0]);
    ASSERT_EQ(4, outputShape[1]);
}























TEST(Conv2DLayer, pad_shape_33)
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

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input.setConstant(1.0f);

    auto inputValue = Chianti::Util::tensorToValue(input, true);
    auto outputValue = Chianti::Util::tensorToValue(output, false);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    std::cout << output << "\n";
}
