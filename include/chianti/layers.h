#pragma once

#include "CNTKLibrary.h"
#include "values.h"
#include "nonlinearities.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <array>
#include <string>
#include <functional>

#define MAKE_SETTER(className, functionName, parameterName) \
className & functionName(const decltype(parameterName) & parameterName) {\
    this->parameterName = parameterName; \
    return *this; \
}

#define MAKE_GETTER(functionName, parameterName) \
decltype(parameterName) functionName() const {\
    return this->parameterName; \
}

namespace Chianti
{
    namespace Layers
    {
        /*!
         * Converts a Chianti parameter to a CNTK parameter.
         *
         * @tparam rank The rank of the tensor
         * @param v The parameter value
         * @param shape The shape of the final parameter
         * @return The CNTK parameter
         */
        template<int rank>
        inline CNTK::Variable resolveParameter(const Values::CompositeValue<Eigen::Tensor<float, rank>, CNTK::ParameterInitializer> & v, const CNTK::NDShape & shape, const CNTK::DeviceDescriptor & device)
        {
            if (Values::isActive<0>(v))
            {
                // 1. eigen tensor to parameter
                const Eigen::Tensor<float, rank> & t = Values::get<0>(v);

                // First: Create a view from the tensor
                auto view = Util::tensorToView<rank, float>(t);

                // Second: Create a parameter array on the device and copy the data
                auto params = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::DataType::Float, shape, device);
                params->CopyFrom(*view);

                // Create the parameter from an Eigen tensor
                return CNTK::Parameter(params);
            }
            else if (Values::isActive<1>(v))
            {
                // Parameter initializer to parameter
                return CNTK::Parameter(shape, CNTK::DataType::Float, Values::get<1>(v), device);
            }
            else
            {
                // This should never happen
                // TODO: Throw exception
            }
        }

        /*!
         * This is the base class for all layers.
         */
        class AbstractLayer
        {
        public:
            /*!
             * Converts the Chianti layer into a CNTK node.
             *
             * @return The CNTK node.
             */
            virtual CNTK::FunctionPtr build() const = 0;

            /*!
             * Implicitly converts the Chianti layer into a CNTK node.
             *
             * @return The CNTK node.
             */
            operator CNTK::FunctionPtr() const
            {
                return this->build();
            }

        protected:
            /*!
             * Initializes a new instance of the <AbstractLayer> class.
             *
             * @param device The device where the parameters of the layer shall be stored.
             */
            AbstractLayer(const CNTK::DeviceDescriptor & device) : device(device) {}

            /*!
             * Class destructor.
             */
            virtual ~AbstractLayer(){}

            /*!
             * The CNTK device descriptor.
             * This indicates where the parameters for the layer are stored.
             */
             CNTK::DeviceDescriptor device;
        };

        /*!
         * This is the base class for all layers only have a single input stream.
         */
        class AbstractSingleInputLayer : public AbstractLayer
        {
        protected:
            /*!
             * Initializes a new instance of the <AbstractSingleInputLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the layer parameters shall be stored.
             */
            AbstractSingleInputLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) : AbstractLayer(device), input(input) {}

            virtual ~AbstractSingleInputLayer() {}

            /*!
             * This it the CNTK variable that represents the layer input.
             */
            CNTK::Variable input;
        };

        /*!
         * This layer adds a 2D convolution operation followed by a bias-term (optional) and a non-linearity (optional).
         * Use this layer for processing multi-channel images.
         */
        class Conv2DLayer : public AbstractSingleInputLayer
        {
        public:
            /*!
             * The number of filter kernels.
             */
            uint64_t _numFilters;
            /*!
             * The size of the filters.
             */
            ::Chianti::Values::ArrayValue<uint64_t, 2> _filterSize;
            /*!
             * The amount of padding on each side
             */
            ::Chianti::Values::CompositeValue<::Chianti::Values::ArrayValue<uint64_t, 2>, std::string, uint64_t> _pad;
            /*!
             * The filter stride.
             */
            ::Chianti::Values::ArrayValue<uint64_t, 2> _stride;
            /*!
             * Filter kernel.
             */
            ::Chianti::Values::CompositeValue<Eigen::Tensor<float, 4>, CNTK::ParameterInitializer> _W;
            /*!
             * Bias parameter
             */
            ::Chianti::Values::CompositeValue<Eigen::Tensor<float, 1>, CNTK::ParameterInitializer, bool> _b;
            /*!
             * Non-linearity
             */
            ::Chianti::Values::CompositeValue<std::function<CNTK::Variable(const CNTK::Variable&)>, bool> _nonLinearity;

        public:
            /*!
             * Initializes a new instance of the <Conv2DLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit Conv2DLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) :
                    AbstractSingleInputLayer(input, device),
                    _numFilters(1),
                    _filterSize{3, 3},
                    _pad("same"),
                    _stride{1, 1},
                    _W(CNTK::HeNormalInitializer()),
                    _b(CNTK::ConstantInitializer(0)),
                    _nonLinearity(Chianti::Nonlinearities::rectify)
            {}

            // Define the getters and setters for the individual class members

            MAKE_GETTER(numFilters, _numFilters)
            MAKE_SETTER(Conv2DLayer, numFilters, _numFilters)

            MAKE_GETTER(filterSize, _filterSize)
            MAKE_SETTER(Conv2DLayer, filterSize, _filterSize)

            MAKE_GETTER(pad, _pad)
            MAKE_SETTER(Conv2DLayer, pad, _pad)

            MAKE_GETTER(stride, _stride)
            MAKE_SETTER(Conv2DLayer, stride, _stride)

            MAKE_GETTER(W, _W)
            MAKE_SETTER(Conv2DLayer, W, _W)

            MAKE_GETTER(b, _b)
            MAKE_SETTER(Conv2DLayer, b, _b)

            MAKE_GETTER(nonLinearity, _nonLinearity)
            MAKE_SETTER(Conv2DLayer, nonLinearity, _nonLinearity)

            /*!
             * Converts the Chianti layer into a CNTK node.
             *
             * @return The CNTK node.
             */
            CNTK::FunctionPtr build() const
            {
                // Determine the correct amount of padding
                CNTK::NDShape lowerPad = {0};
                CNTK::NDShape upperPad = {0};
                std::vector<bool> autoPadding = {true};

                if (Values::isActive<0>(this->_pad))
                {
                    const Values::ArrayValue<uint64_t, 2> padding = Values::get<0>(this->_pad);

                    // The padding has been manually specified
                    autoPadding = {false, false, false};
                    lowerPad = {padding[0], padding[1], 0};
                    upperPad = {padding[0], padding[1], 0};
                }
                else
                {
                    // Determine the special kind of padding to use
                    const std::string & padding = Values::get<1>(this->_pad);

                    if (padding == "full")
                    {
                        // Compute the convolution everywhere where the filter and the filters overlap at least one pixel
                        autoPadding = {false, false, false};
                        lowerPad = {this->_filterSize[0], this->_filterSize[1], 0};
                        upperPad = {this->_filterSize[0], this->_filterSize[1], 0};
                    }
                    else if (padding == "same")
                    {
                        // Pad such that the input map has the same size as the output map
                        // Actually, nothing to do here.
                        // This is the standard setting
                    }
                    else if (padding == "valid")
                    {
                        // No padding
                        // Only compute activations where the input and the filter fully overlap
                        autoPadding = {false, false, false};
                        lowerPad = {0, 0, 0};
                        upperPad = {0, 0, 0};
                    }
                    else
                    {
                        // TODO: Throw exception
                    }
                }

                size_t numInputChannels = this->input.Shape()[this->input.Shape().Rank() - 1];

                // Determine the shape of the convolution
                CNTK::NDShape filterShape = { this->_filterSize[0], this->_filterSize[1], numInputChannels, this->_numFilters };

                // Create the parameter
                auto convParams = resolveParameter<4>(this->_W, filterShape, this->device);

                return Convolution(convParams, this->input, { this->_stride[0], this->_stride[1], numInputChannels }, { true }, autoPadding, lowerPad, upperPad);
            }
        };
    }
}