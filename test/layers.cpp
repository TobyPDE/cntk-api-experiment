#include <gtest/gtest.h>
#include "chianti/chianti.h"


TEST(Conv2DLayer, pad_shape_same)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("same")
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("full")
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({0, 0})
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({1, 1})
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({2, 2})
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({3, 3})
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("same")
            .stride({1, 1})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 5, 5, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad({1, 1})
            .stride({2, 2})
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
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
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({2, 2})
            .pad({0, 0})
            .stride({2, 2})
            .numFilters(1);
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(4, outputShape[0]);
    ASSERT_EQ(4, outputShape[1]);
}

TEST(Conv2DLayer, W_initializer)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 3, 3, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
            .numFilters(1)
            .W(CNTK::ConstantInitializer(0));

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input.setConstant(1.0f);

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
    ASSERT_EQ(1, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(0, output(0, 0, 0, 0, 0));
}

TEST(Conv2DLayer, W_initializer_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 3, 3, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
            .numFilters(1)
            .W(CNTK::ConstantInitializer(1));

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input.setConstant(1.0f);

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
    ASSERT_EQ(1, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(9, output(0, 0, 0, 0, 0));
}

TEST(Conv2DLayer, W_eigen)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 3, 3, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Create a filter in Eigen
    Eigen::Tensor<float, 4> W(3, 3, 1, 1);
    W(0, 0, 0, 0) = 1.0f;
    W(0, 1, 0, 0) = 1.0f / 2.0f;
    W(0, 2, 0, 0) = 1.0f / 3.0f;
    W(1, 0, 0, 0) = 1.0f / 4.0f;
    W(1, 1, 0, 0) = 1.0f / 5.0f;
    W(1, 2, 0, 0) = 1.0f / 6.0f;
    W(2, 0, 0, 0) = 1.0f / 7.0f;
    W(2, 1, 0, 0) = 1.0f / 8.0f;
    W(2, 2, 0, 0) = 1.0f / 9.0f;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
            .numFilters(1)
            .W(W);

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input(0, 0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0, 0) = 2.0f;
    input(0, 2, 0, 0, 0) = 3.0f;
    input(1, 0, 0, 0, 0) = 4.0f;
    input(1, 1, 0, 0, 0) = 5.0f;
    input(1, 2, 0, 0, 0) = 6.0f;
    input(2, 0, 0, 0, 0) = 7.0f;
    input(2, 1, 0, 0, 0) = 8.0f;
    input(2, 2, 0, 0, 0) = 9.0f;

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
    ASSERT_EQ(1, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(9.0f, output(0, 0, 0, 0, 0));
}

TEST(Conv2DLayer, W_eigen_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 3, 3, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Create a filter in Eigen
    Eigen::Tensor<float, 4> W(3, 3, 1, 2);
    W(0, 0, 0, 0) = 1.0f;
    W(0, 1, 0, 0) = 1.0f / 2.0f;
    W(0, 2, 0, 0) = 1.0f / 3.0f;
    W(1, 0, 0, 0) = 1.0f / 4.0f;
    W(1, 1, 0, 0) = 1.0f / 5.0f;
    W(1, 2, 0, 0) = 1.0f / 6.0f;
    W(2, 0, 0, 0) = 1.0f / 7.0f;
    W(2, 1, 0, 0) = 1.0f / 8.0f;
    W(2, 2, 0, 0) = 1.0f / 9.0f;

    W(0, 0, 0, 1) = 2.0f;
    W(0, 1, 0, 1) = 2.0f / 2.0f;
    W(0, 2, 0, 1) = 2.0f / 3.0f;
    W(1, 0, 0, 1) = 2.0f / 4.0f;
    W(1, 1, 0, 1) = 2.0f / 5.0f;
    W(1, 2, 0, 1) = 2.0f / 6.0f;
    W(2, 0, 0, 1) = 2.0f / 7.0f;
    W(2, 1, 0, 1) = 2.0f / 8.0f;
    W(2, 2, 0, 1) = 2.0f / 9.0f;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
            .numFilters(2)
            .W(W);

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input(0, 0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0, 0) = 2.0f;
    input(0, 2, 0, 0, 0) = 3.0f;
    input(1, 0, 0, 0, 0) = 4.0f;
    input(1, 1, 0, 0, 0) = 5.0f;
    input(1, 2, 0, 0, 0) = 6.0f;
    input(2, 0, 0, 0, 0) = 7.0f;
    input(2, 1, 0, 0, 0) = 8.0f;
    input(2, 2, 0, 0, 0) = 9.0f;

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
    ASSERT_EQ(2, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(9.0f, output(0, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(18.0f, output(0, 0, 1, 0, 0));
}

TEST(Conv2DLayer, W_eigen_bias)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 3, 3, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Create a filter in Eigen
    Eigen::Tensor<float, 4> W(3, 3, 1, 2);
    W(0, 0, 0, 0) = 1.0f;
    W(0, 1, 0, 0) = 1.0f / 2.0f;
    W(0, 2, 0, 0) = 1.0f / 3.0f;
    W(1, 0, 0, 0) = 1.0f / 4.0f;
    W(1, 1, 0, 0) = 1.0f / 5.0f;
    W(1, 2, 0, 0) = 1.0f / 6.0f;
    W(2, 0, 0, 0) = 1.0f / 7.0f;
    W(2, 1, 0, 0) = 1.0f / 8.0f;
    W(2, 2, 0, 0) = 1.0f / 9.0f;

    W(0, 0, 0, 1) = 2.0f;
    W(0, 1, 0, 1) = 2.0f / 2.0f;
    W(0, 2, 0, 1) = 2.0f / 3.0f;
    W(1, 0, 0, 1) = 2.0f / 4.0f;
    W(1, 1, 0, 1) = 2.0f / 5.0f;
    W(1, 2, 0, 1) = 2.0f / 6.0f;
    W(2, 0, 0, 1) = 2.0f / 7.0f;
    W(2, 1, 0, 1) = 2.0f / 8.0f;
    W(2, 2, 0, 1) = 2.0f / 9.0f;

    // Create the bias
    Eigen::Tensor<float, 3> b(1, 1, 2);
    b(0, 0, 0) = 0.0f;
    b(0, 0, 1) = -9.0f;


    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
            .numFilters(2)
            .W(W)
            .b(b);

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input(0, 0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0, 0) = 2.0f;
    input(0, 2, 0, 0, 0) = 3.0f;
    input(1, 0, 0, 0, 0) = 4.0f;
    input(1, 1, 0, 0, 0) = 5.0f;
    input(1, 2, 0, 0, 0) = 6.0f;
    input(2, 0, 0, 0, 0) = 7.0f;
    input(2, 1, 0, 0, 0) = 8.0f;
    input(2, 2, 0, 0, 0) = 9.0f;

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
    ASSERT_EQ(2, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(9.0f, output(0, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(9.0f, output(0, 0, 1, 0, 0));
}

TEST(Conv2DLayer, W_eigen_bias_relu)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 3, 3, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Create a filter in Eigen
    Eigen::Tensor<float, 4> W(3, 3, 1, 2);
    W(0, 0, 0, 0) = 1.0f;
    W(0, 1, 0, 0) = 1.0f / 2.0f;
    W(0, 2, 0, 0) = 1.0f / 3.0f;
    W(1, 0, 0, 0) = 1.0f / 4.0f;
    W(1, 1, 0, 0) = 1.0f / 5.0f;
    W(1, 2, 0, 0) = 1.0f / 6.0f;
    W(2, 0, 0, 0) = 1.0f / 7.0f;
    W(2, 1, 0, 0) = 1.0f / 8.0f;
    W(2, 2, 0, 0) = 1.0f / 9.0f;

    W(0, 0, 0, 1) = 2.0f;
    W(0, 1, 0, 1) = 2.0f / 2.0f;
    W(0, 2, 0, 1) = 2.0f / 3.0f;
    W(1, 0, 0, 1) = 2.0f / 4.0f;
    W(1, 1, 0, 1) = 2.0f / 5.0f;
    W(1, 2, 0, 1) = 2.0f / 6.0f;
    W(2, 0, 0, 1) = 2.0f / 7.0f;
    W(2, 1, 0, 1) = 2.0f / 8.0f;
    W(2, 2, 0, 1) = 2.0f / 9.0f;

    // Create the bias
    Eigen::Tensor<float, 3> b(1, 1, 2);
    b(0, 0, 0) = 0.0f;
    b(0, 0, 1) = -19.0f;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
            .numFilters(2)
            .W(W)
            .b(b)
            .nonLinearity(Chianti::Nonlinearities::rectify);

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input(0, 0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0, 0) = 2.0f;
    input(0, 2, 0, 0, 0) = 3.0f;
    input(1, 0, 0, 0, 0) = 4.0f;
    input(1, 1, 0, 0, 0) = 5.0f;
    input(1, 2, 0, 0, 0) = 6.0f;
    input(2, 0, 0, 0, 0) = 7.0f;
    input(2, 1, 0, 0, 0) = 8.0f;
    input(2, 2, 0, 0, 0) = 9.0f;

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
    ASSERT_EQ(2, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(9.0f, output(0, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(0.0f, output(0, 0, 1, 0, 0));
}

TEST(Conv2DLayer, W_eigen_no_bias)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 3, 3, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Create a filter in Eigen
    Eigen::Tensor<float, 4> W(3, 3, 1, 2);
    W(0, 0, 0, 0) = 1.0f;
    W(0, 1, 0, 0) = 1.0f / 2.0f;
    W(0, 2, 0, 0) = 1.0f / 3.0f;
    W(1, 0, 0, 0) = 1.0f / 4.0f;
    W(1, 1, 0, 0) = 1.0f / 5.0f;
    W(1, 2, 0, 0) = 1.0f / 6.0f;
    W(2, 0, 0, 0) = 1.0f / 7.0f;
    W(2, 1, 0, 0) = 1.0f / 8.0f;
    W(2, 2, 0, 0) = 1.0f / 9.0f;

    W(0, 0, 0, 1) = 2.0f;
    W(0, 1, 0, 1) = 2.0f / 2.0f;
    W(0, 2, 0, 1) = 2.0f / 3.0f;
    W(1, 0, 0, 1) = 2.0f / 4.0f;
    W(1, 1, 0, 1) = 2.0f / 5.0f;
    W(1, 2, 0, 1) = 2.0f / 6.0f;
    W(2, 0, 0, 1) = 2.0f / 7.0f;
    W(2, 1, 0, 1) = 2.0f / 8.0f;
    W(2, 2, 0, 1) = 2.0f / 9.0f;

    // Act
    network = Chianti::Layers::Conv2DLayer(X, device)
            .filterSize({3, 3})
            .pad("valid")
            .stride({1, 1})
            .numFilters(2)
            .W(W)
            .b(false)
            .nonLinearity(Chianti::Nonlinearities::rectify);

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input(0, 0, 0, 0, 0) = 1.0f;
    input(0, 1, 0, 0, 0) = 2.0f;
    input(0, 2, 0, 0, 0) = 3.0f;
    input(1, 0, 0, 0, 0) = 4.0f;
    input(1, 1, 0, 0, 0) = 5.0f;
    input(1, 2, 0, 0, 0) = 6.0f;
    input(2, 0, 0, 0, 0) = 7.0f;
    input(2, 1, 0, 0, 0) = 8.0f;
    input(2, 2, 0, 0, 0) = 9.0f;

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
    ASSERT_EQ(2, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(9.0f, output(0, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(18.0f, output(0, 0, 1, 0, 0));
}

TEST(MaxPool2DLayer, pad_0)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({2, 2})
            .pad({0, 0})
            .stride({2, 2});
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(4, outputShape[0]);
    ASSERT_EQ(4, outputShape[1]);
}

TEST(MaxPool2DLayer, pad_0_size_3)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({3, 3})
            .pad({0, 0})
            .stride({2, 2});
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(MaxPool2DLayer, pad_false_size_3)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({3, 3})
            .pad(false)
            .stride({2, 2});
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(MaxPool2DLayer, pad_true_size_3)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({3, 3})
            .pad(true)
            .stride({2, 2});
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(4, outputShape[0]);
    ASSERT_EQ(4, outputShape[1]);
}

TEST(MaxPool2DLayer, pad_auto_size_3)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({3, 3})
            .pad("auto")
            .stride({2, 2});
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(4, outputShape[0]);
    ASSERT_EQ(4, outputShape[1]);
}

TEST(MaxPool2DLayer, pad_none_size_3)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({3, 3})
            .pad("none")
            .stride({2, 2});
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(3, outputShape[0]);
    ASSERT_EQ(3, outputShape[1]);
}

TEST(MaxPool2DLayer, pool_size_8)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 8, 8, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({8, 8})
            .pad("none")
            .stride({2, 2});
    auto outputVar = network->Output();
    auto outputShape = outputVar.Shape();

    // Assert
    ASSERT_EQ(1, outputShape[0]);
    ASSERT_EQ(1, outputShape[1]);
}

TEST(MaxPool2DLayer, value)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 4, 4, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::MaxPool2DLayer(X, device)
            .poolSize({2, 2})
            .stride({2, 2});

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            input(i, j, 0, 0, 0) = 4 * i + j;
        }
    }

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(2, outputShape[0]);
    ASSERT_EQ(2, outputShape[1]);
    ASSERT_EQ(1, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(5.0f, output(0, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(7.0f, output(0, 1, 0, 0, 0));
    ASSERT_FLOAT_EQ(13.0f, output(1, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(15.0f, output(1, 1, 0, 0, 0));
}

TEST(AveragePool2DLayer, value)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 4, 4, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::AveragePool2DLayer(X, device)
            .poolSize({2, 2})
            .stride({2, 2});

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            input(i, j, 0, 0, 0) = 4 * i + j;
        }
    }

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(2, outputShape[0]);
    ASSERT_EQ(2, outputShape[1]);
    ASSERT_EQ(1, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);
    ASSERT_FLOAT_EQ(10.0f / 4.0f, output(0, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(18.0f / 4.0f, output(0, 1, 0, 0, 0));
    ASSERT_FLOAT_EQ(42.0f / 4.0f, output(1, 0, 0, 0, 0));
    ASSERT_FLOAT_EQ(50.0f / 4.0f, output(1, 1, 0, 0, 0));
}

TEST(Upscale2DLayer, scaleFactor_2_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 4, 4, 2 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Upscale2DLayer(X, device)
            .scaleFactor({2, 2});

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                input(i, j, k, 0, 0) = (4 * i + j) * (k + 1);
            }
        }
    }

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(8, outputShape[0]);
    ASSERT_EQ(8, outputShape[1]);
    ASSERT_EQ(2, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                ASSERT_FLOAT_EQ(input(i / 2, j / 2, k, 0, 0), output(i, j, k, 0, 0));
            }
        }
    }
}

TEST(Upscale2DLayer, scaleFactor_1_2)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 4, 4, 2 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Upscale2DLayer(X, device)
            .scaleFactor({1, 2});

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                input(i, j, k, 0, 0) = (4 * i + j) * (k + 1);
            }
        }
    }

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(4, outputShape[0]);
    ASSERT_EQ(8, outputShape[1]);
    ASSERT_EQ(2, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                ASSERT_FLOAT_EQ(input(i, j / 2, k, 0, 0), output(i, j, k, 0, 0));
            }
        }
    }
}

TEST(Upscale2DLayer, scaleFactor_8_8)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 1, 1, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::Upscale2DLayer(X, device)
            .scaleFactor({8, 8});

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 1});
    auto outputShape = outputVar.Shape().AppendShape({1, 1});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input.setConstant(2.0f);

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    // Assert
    ASSERT_EQ(8, outputShape[0]);
    ASSERT_EQ(8, outputShape[1]);
    ASSERT_EQ(1, outputShape[2]);
    ASSERT_EQ(1, outputShape[3]);
    ASSERT_EQ(1, outputShape[4]);

    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            ASSERT_FLOAT_EQ(2.0f, output(i, j, 0, 0, 0));
        }
    }
}

TEST(BatchNormLayer, test)
{
    // Arrange
    auto device = CNTK::DeviceDescriptor::GPUDevice(0);
    auto X = CNTK::InputVariable({ 2, 2, 1 }, CNTK::DataType::Float);
    CNTK::FunctionPtr network;

    // Act
    network = Chianti::Layers::BatchNormLayer(X, device).useCuDNN(true);

    auto outputVar = network->Output();

    auto inputShape = X.Shape().AppendShape({1, 10});
    auto outputShape = outputVar.Shape().AppendShape({1, 10});

    Eigen::Tensor<float, 5> input(Chianti::Util::convertShape<5>(inputShape));
    Eigen::Tensor<float, 5> output(Chianti::Util::convertShape<5>(outputShape));

    input.setConstant(2.0f);

    auto inputValue = Chianti::Util::tensorToValue(input);
    auto outputValue = Chianti::Util::tensorToValue(output);

    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputs = {{outputVar, outputValue}};

    network->Forward({{X, inputValue}}, outputs, device);

    std::cout << input << "\n" << output << "\n";
}

