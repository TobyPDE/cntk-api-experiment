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
            ::Chianti::Values::CompositeValue<::Chianti::Values::ArrayValue<uint64_t, 2>, std::string> _pad;
            /*!
             * The filter stride.
             */
            ::Chianti::Values::ArrayValue<uint64_t, 2> _stride;
            /*!
             * Filter kernel.
             */
            ::Chianti::Values::CompositeValue<Eigen::Tensor<float, 3>, CNTK::Variable, bool, CNTK::ParameterInitializer> _W;
            /*!
             * Bias parameter
             */
            ::Chianti::Values::CompositeValue<Eigen::Tensor<float, 1>, CNTK::Variable, bool, CNTK::ParameterInitializer> _b;
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
                return nullptr;
            }
        };

    }
}