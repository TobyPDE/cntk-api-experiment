#pragma once

#include <string>
#include <array>

#include "CNTKLibrary.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace Chianti
{
    namespace Util
    {
        /**
         * Asserts a condition and throws an assertion exception is the condition is violated.
         *
         * @param condition The condition to assert
         * @param message The exception message that is used if the condition is not fulfilled
         */
        inline void assertMsg(bool condition, const std::string & message)
        {

        }

        /*!
         * Creates a view from a tensor.
         *
         * @param tensor The Eigen tensor
         * @return The CNTK value
         */
        template <int rank, typename T = float>
        inline CNTK::NDArrayViewPtr tensorToView(const Eigen::Tensor<T, rank> & tensor)
        {
            auto device = CNTK::DeviceDescriptor::CPUDevice();

            // Extract the shape from the tensor
            CNTK::NDShape shape;
            for (size_t n = 0; n < static_cast<size_t>(rank); n++)
            {
                shape = shape.AppendShape({static_cast<size_t>(tensor.dimension(n))});
            }

            return CNTK::MakeSharedObject<CNTK::NDArrayView>(shape, tensor.data(), tensor.size(), device);
        }

        /*!
         * Creates a view from a tensor.
         *
         * @param tensor The Eigen tensor
         * @return The CNTK value
         */
        template <int rank, typename T = float>
        inline CNTK::NDArrayViewPtr tensorToView(Eigen::Tensor<T, rank> & tensor)
        {
            auto device = CNTK::DeviceDescriptor::CPUDevice();

            // Extract the shape from the tensor
            CNTK::NDShape shape;
            for (size_t n = 0; n < static_cast<size_t>(rank); n++)
            {
                shape = shape.AppendShape({static_cast<size_t>(tensor.dimension(n))});
            }

            return CNTK::MakeSharedObject<CNTK::NDArrayView>(shape, tensor.data(), tensor.size(), device, false);
        }

        /*!
         * Creates a CNTK value from an Eigen tensor.
         *
         * @param tensor The Eigen tensor
         * @return The CNTK value
         */
        template <int rank, typename T = float>
        inline CNTK::ValuePtr tensorToValue(Eigen::Tensor<T, rank> & tensor)
        {
            return CNTK::MakeSharedObject<CNTK::Value>(tensorToView(tensor));
        }

        /*!
         * Creates a CNTK value from an Eigen tensor.
         *
         * @param tensor The Eigen tensor
         * @return The CNTK value
         */
        template <int rank, typename T = float>
        inline CNTK::ValuePtr tensorToValue(const Eigen::Tensor<T, rank> & tensor)
        {
            return CNTK::MakeSharedObject<CNTK::Value>(tensorToView(tensor));
        }

        /*!
         * Creates an Eigen Tensor from a CNTK shape.
         *
         * @param shape The CNTK shape
         * @return The eigen tensor
         */
        template<size_t rank>
        Eigen::array<long int, rank> convertShape(const CNTK::NDShape & shape)
        {
            // Verify that the shape has the correct number of dimensions
            if (shape.Rank() != rank)
            {
                // TODO: Throw exception
            }

            // Transfer the shape information to an array
            Eigen::array<long int, rank> newShape;
            for (size_t n = 0; n < rank; n++)
            {
                newShape[n] = static_cast<long int>(shape[n]);
            }
            return newShape;
        }
    }
}