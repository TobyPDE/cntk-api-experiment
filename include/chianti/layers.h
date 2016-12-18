#pragma once

#include "CNTKLibrary.h"
#include "values.h"
#include "nonlinearities.h"
#include "exception.h"

#include <unsupported/Eigen/CXX11/Tensor>
#include <array>
#include <string>
#include <functional>

#define MAKE_SETTER(functionName, parameterName) \
Self & functionName(const decltype(parameterName) & parameterName) {\
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
                Exception::terminate("Illegal composite value.", 0x1001);
            }
        }

        /*!
         * Converts a Chianti parameter to a CNTK parameter.
         *
         * @tparam rank The rank of the tensor
         * @param v The parameter value
         * @param shape The shape of the final parameter
         * @return The CNTK parameter
         */
        template<int rank>
        inline CNTK::Variable resolveParameter(const Values::CompositeValue<Eigen::Tensor<float, rank>, CNTK::ParameterInitializer, bool> & v, const CNTK::NDShape & shape, const CNTK::DeviceDescriptor & device)
        {
            if (Values::isActive<0>(v))
            {
                return resolveParameter(Values::CompositeValue<Eigen::Tensor<float, rank>, CNTK::ParameterInitializer>(Values::get<0>(v)), shape, device);
            }
            else if (Values::isActive<1>(v))
            {
                // Parameter initializer to parameter
                return resolveParameter(Values::CompositeValue<Eigen::Tensor<float, rank>, CNTK::ParameterInitializer>(Values::get<1>(v)), shape, device);
            }
            else
            {
                // No parameter is defined
                Exception::terminate("Illegal composite value.", 0x1002);
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
            typedef Conv2DLayer Self;

            /*!
             * The number of filter kernels.
             */
            uint64_t _numFilters;
            /*!
             * The size of the filters.
             */
            Values::ArrayValue<uint64_t, 2> _filterSize;
            /*!
             * The amount of padding on each side
             */
            Values::CompositeValue<Values::ArrayValue<uint64_t, 2>, std::string> _pad;
            /*!
             * The filter stride.
             */
            Values::ArrayValue<uint64_t, 2> _stride;
            /*!
             * Filter kernel.
             */
            Values::CompositeValue<Eigen::Tensor<float, 4>, CNTK::ParameterInitializer> _W;
            /*!
             * Bias parameter
             */
            Values::CompositeValue<Eigen::Tensor<float, 3>, CNTK::ParameterInitializer, bool> _b;
            /*!
             * Non-linearity
             */
            std::function<CNTK::FunctionPtr(CNTK::FunctionPtr)> _nonLinearity;

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
            MAKE_SETTER(numFilters, _numFilters)

            MAKE_GETTER(filterSize, _filterSize)
            MAKE_SETTER(filterSize, _filterSize)

            MAKE_GETTER(pad, _pad)
            MAKE_SETTER(pad, _pad)

            MAKE_GETTER(stride, _stride)
            MAKE_SETTER(stride, _stride)

            MAKE_GETTER(W, _W)
            MAKE_SETTER(W, _W)

            MAKE_GETTER(b, _b)
            MAKE_SETTER(b, _b)

            MAKE_GETTER(nonLinearity, _nonLinearity)
            MAKE_SETTER(nonLinearity, _nonLinearity)

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
                    const auto padding = Values::get<0>(this->_pad);

                    // The padding has been manually specified
                    autoPadding = {false, false, false};
                    lowerPad = {padding[0], padding[1], 0};
                    upperPad = {padding[0], padding[1], 0};
                }
                else
                {
                    // Determine the special kind of padding to use
                    const auto & padding = Values::get<1>(this->_pad);

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
                        throw Exception::IllegalArgumentException("Illegal string value for parameter 'pad'.");
                    }
                }

                size_t numInputChannels = this->input.Shape()[this->input.Shape().Rank() - 1];

                // Set up the convolution
                // ----------------------
                // Determine the shape of the convolution
                CNTK::NDShape filterShape = { this->_filterSize[0], this->_filterSize[1], numInputChannels, this->_numFilters };
                auto convParams = resolveParameter<4>(this->_W, filterShape, this->device);
                CNTK::FunctionPtr network = Convolution(
                        convParams,
                        this->input,
                        { this->_stride[0], this->_stride[1], numInputChannels },
                        { true },
                        autoPadding,
                        lowerPad,
                        upperPad);

                // Set up the bias term
                // --------------------
                if (!Values::isActive<2>(this->_b) || Values::get<2>(this->_b))
                {
                    // Add a bias term
                    CNTK::NDShape biasShape = { 1, 1, this->_numFilters };

                    // Create the parameter
                    if (Values::isActive<2>(this->_b))
                    {
                        // The user didn't define anything
                        // Create a 0 initialized parameter
                        auto biasParams = CNTK::Parameter(biasShape, CNTK::DataType::Float, CNTK::ConstantInitializer(0), this->device);
                        network = CNTK::Plus(network, biasParams);
                    }
                    else
                    {
                        // If the user specified an Eigen tensor, then the first two dimensions must have size 1
                        if (Values::isActive<0>(this->_b))
                        {
                            Exception::assertArgument(Values::get<0>(this->_b).dimensions()[0] == 1, "Bias must have shape (1, 1, numFilters).");
                            Exception::assertArgument(Values::get<0>(this->_b).dimensions()[1] == 1, "Bias must have shape (1, 1, numFilters).");
                        }

                        // The user specified the bias
                        auto biasParms = resolveParameter<3>(this->_b, biasShape, this->device);
                        network = CNTK::Plus(network, biasParms);
                    }
                }

                // Apply non-linearity
                network = this->_nonLinearity(network);

                return network;
            }
        };

        /**
         * Abstract pooling layer. Can do max pooling and average pooling.
         */
        class AbstractPool2DLayer : public AbstractSingleInputLayer
        {
        protected:
            typedef AbstractPool2DLayer Self;

            /*!
             * The size of the pooling region.
             */
            Values::ArrayValue<uint64_t, 2> _poolSize;
            /*!
             * The amount of padding on each side
             */
            Values::CompositeValue<Values::ArrayValue<uint64_t, 2>, std::string, bool> _pad;
            /*!
             * The pooling stride.
             */
            Values::ArrayValue<uint64_t, 2> _stride;

        private:
            /**
             * The pooling type.
             */
            const CNTK::PoolingType poolingType;

        protected:
            /*!
             * Initializes a new instance of the <Conv2DLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit AbstractPool2DLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device, const CNTK::PoolingType poolingType) :
            AbstractSingleInputLayer(input, device),
            _poolSize{2, 2},
            _pad("auto"),
            _stride{2, 2},
            poolingType(poolingType)
            {}

        public:

            // Define the getters and setters for the individual class members

            MAKE_GETTER(poolSize, _poolSize)
            MAKE_SETTER(poolSize, _poolSize)

            MAKE_GETTER(pad, _pad)
            MAKE_SETTER(pad, _pad)

            MAKE_GETTER(stride, _stride)
            MAKE_SETTER(stride, _stride)

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

                if (Values::isActive<0>(_pad))
                {
                    // The padding has been manually specified
                    const auto & padding = Values::get<0>(_pad);

                    autoPadding = {false, false, false};
                    lowerPad = {padding[0], padding[1], 0};
                    upperPad = {padding[0], padding[1], 0};
                }
                else if (Values::isActive<1>(_pad))
                {
                    // Padding is given as string
                    const auto & padding = Values::get<1>(_pad);

                    if (padding == "auto")
                    {
                        // Nothing to do here
                    }
                    else if (padding == "none")
                    {
                        // Use no padding
                        autoPadding = {false, false, false};
                        lowerPad = {0, 0, 0};
                        upperPad = {0, 0, 0};
                    }
                    else
                    {
                        // Unrecognized option
                        throw Exception::IllegalArgumentException("Invalid string value for pad.");
                    }
                }
                else if (Values::isActive<2>(_pad) && !Values::get<2>(_pad))
                {
                    // The user indicated that no padding shall be performed
                    autoPadding = {false, false, false};
                    lowerPad = {0, 0, 0};
                    upperPad = {0, 0, 0};
                }

                CNTK::FunctionPtr network = CNTK::Pooling(
                        this->input,
                        this->poolingType,
                        {_poolSize[0], _poolSize[1]},
                        {_stride[0], _stride[1]},
                        autoPadding,
                        lowerPad,
                        upperPad);

                return network;
            }
        };

        /**
         * Max pooling layer.
         */
        class MaxPool2DLayer : public AbstractPool2DLayer
        {
        public:
            /*!
             * Initializes a new instance of the <Conv2DLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit MaxPool2DLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) : AbstractPool2DLayer(input, device, CNTK::PoolingType::Max) {}
        };

        /**
         * Average pooling layer.
         */
        class AveragePool2DLayer : public AbstractPool2DLayer
        {
        public:
            /*!
             * Initializes a new instance of the <Conv2DLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit AveragePool2DLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) : AbstractPool2DLayer(input, device, CNTK::PoolingType::Average) {}
        };

        /**
         * This layer upscales a tensor with two spatial dimensions. By default, it upscales by repeating the values
         * along the spatial axes. However, it also support bilinear interpolation, which is more expensive.
         */
        class Upscale2DLayer : public AbstractSingleInputLayer {
        private:
            typedef Upscale2DLayer Self;
            /*!
             * The upscale factor
             */
            Values::ArrayValue<uint64_t, 2> _scaleFactor;

        public:
            /*!
             * Initializes a new instance of the <Conv2DLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit Upscale2DLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) :
            AbstractSingleInputLayer(input, device),
            _scaleFactor{2, 2}
            {}

            // Define the getters and setters for the individual class members

            MAKE_GETTER(scaleFactor, _scaleFactor)
            MAKE_SETTER(scaleFactor, _scaleFactor)

            /*!
             * Converts the Chianti layer into a CNTK node.
             *
             * @return The CNTK node.
             */
            CNTK::FunctionPtr build() const
            {
                // We implement the unpooling as the backwards pass of the convolution.
                // TODO: This can be implemented more efficiently. For example: https://github.com/Microsoft/CNTK/issues/711

                // Create the filter kernel for the unpooling operation
                size_t numInputChannels = this->input.Shape()[this->input.Shape().Rank() - 1];

                CNTK::NDShape filterShape = { this->_scaleFactor[0], this->_scaleFactor[1], numInputChannels, numInputChannels };

                Eigen::Tensor<float, 4> W(static_cast<int>(_scaleFactor[0]), static_cast<int>(_scaleFactor[1]), static_cast<int>(numInputChannels), static_cast<int>(numInputChannels));
                W.setConstant(0.0f);

                for (int i = 0; i < static_cast<int>(_scaleFactor[0]); i++)
                {
                    for (int j = 0; j < static_cast<int>(_scaleFactor[1]); j++)
                    {
                        for (int c = 0; c < static_cast<int>(numInputChannels); c++)
                        {
                            // A value of 1.0f means that each value is simply repeated
                            W(i, j, c, c) = 1.0f;
                        }
                    }
                }

                // Convert the Eigen tensor to a CNTK parameter
                auto view = Util::tensorToView<4, float>(W);
                auto params = CNTK::MakeSharedObject<CNTK::NDArrayView>(CNTK::DataType::Float, filterShape, device);
                params->CopyFrom(*view);

                // Create the convolution/deconvolution
                CNTK::FunctionPtr network = Convolution(
                        CNTK::Constant(params),
                        this->input,
                        { _scaleFactor[0], _scaleFactor[1], numInputChannels },
                        { true },
                        { false, false, false },
                        { 0, 0, 0 },
                        { 0, 0, 0 },
                        true);

                return network;
            }
        };

        /**
         * This is the base class for non-deterministic layers such as noise layers (DropOut) or normalization layers
         * (BatchNorm).
         */
        class AbstractNonDeterministicLayer : public AbstractSingleInputLayer {
        protected:
            typedef AbstractNonDeterministicLayer Self;
            /*!
             * Whether or not the output should be deterministic
             */
            bool _deterministic;

        public:
            /*!
             * Initializes a new instance of the <AbstractNonDeterministicLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit AbstractNonDeterministicLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) :
                    AbstractSingleInputLayer(input, device),
                    _deterministic(false)
                    {}

            // Define the getters and setters for the individual class members

            MAKE_GETTER(deterministic, _deterministic)
            MAKE_SETTER(deterministic, _deterministic)
        };

        /**
         * This layer implements DropOut.
         */
        class DropOutLayer : public AbstractNonDeterministicLayer {
        private:
            typedef DropOutLayer Self;
            /*!
             * The dropout rate
             */
            double _p;

        public:
            /*!
             * Initializes a new instance of the <AbstractNonDeterministicLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit DropOutLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) :
                    AbstractNonDeterministicLayer(input, device),
                    _p(0.25)
                    {}

            // Define the getters and setters for the individual class members

            MAKE_GETTER(p, _p)
            MAKE_SETTER(p, _p)

            /*!
             * Converts the Chianti layer into a CNTK node.
             *
             * @return The CNTK node.
             */
            CNTK::FunctionPtr build() const
            {
                CNTK::FunctionPtr network = this->input;

                // If the layer is deterministic or there is no drop-out specified, simply return the input
                if (!_deterministic && _p > 0.0)
                {
                    // Apply the drop-out
                    network = CNTK::Dropout(network, _p);
                }

                return network;
            }
        };

        /**
         * Batch normalization layer.
         * TODO: Make it possible to pass eigen tensors for the parameters of the layer.


    CNTK_API FunctionPtr BatchNormalization(const Variable& operand,
                                            const Variable& scale,
                                            const Variable& bias,
                                            const Variable& runningMean,
                                            const Variable& runningInvStd,
                                            bool spatial,
                                            double normalizationTimeConstant = 0,
                                            double blendTimeConstant = 0,
                                            double epsilon = 0.00001,
                                            bool useCuDNNEngine = false,
                                            const std::wstring& name = L"");

         */
        class BatchNormLayer : public AbstractNonDeterministicLayer {
        private:
            typedef BatchNormLayer Self;
            /**
             * Whether or not to use cuDNN
             */
            bool _useCuDNN;
            /**
             * Determines the smoothing of the running mean and the running std.
             */
            double _normalizationTimeConstant;
            /**
             * Regularization parameter.
             */
            double _epsilon;

        public:
            /*!
             * Initializes a new instance of the <AbstractNonDeterministicLayer> class.
             *
             * @param input The layer's input variables.
             * @param device The device where the parameters of the layer shall be stored.
             */
            explicit BatchNormLayer(CNTK::Variable input, const CNTK::DeviceDescriptor & device) :
            AbstractNonDeterministicLayer(input, device),
            _useCuDNN(false),
            // 5000.0 as recommended here: https://github.com/Microsoft/CNTK/wiki/BatchNormalization
            _normalizationTimeConstant(5000.0),
            _epsilon(1e-5)
            {}

            // Define the getters and setters for the individual class members

            MAKE_GETTER(useCuDNN, _useCuDNN)
            MAKE_SETTER(useCuDNN, _useCuDNN)

            MAKE_GETTER(normalizationTimeConstant, _normalizationTimeConstant)
            MAKE_SETTER(normalizationTimeConstant, _normalizationTimeConstant)

            MAKE_GETTER(epsilon, _epsilon)
            MAKE_SETTER(epsilon, _epsilon)

            /*!
             * Converts the Chianti layer into a CNTK node.
             *
             * @return The CNTK node.
             */
            CNTK::FunctionPtr build() const
            {
                CNTK::FunctionPtr network = this->input;

                CNTK::NDShape parameterShape;
                // Determine the size of the parameters
                // If the input tensor has more than one dimension, then this is considered to be a spatial batch-norm
                if (input.Shape().Rank() > 1)
                {
                    // The last dimension determines the number of channels
                    parameterShape = { input.Shape()[input.Shape().Rank() - 1] };
                }
                else
                {
                    parameterShape = { input.Shape()[0] };
                }

                // Create the parameters
                auto scale = CNTK::Parameter(parameterShape, 1.0f, device);
                auto bias = CNTK::Parameter(parameterShape, 0.0f, device);
                auto runningMean = CNTK::Parameter(parameterShape, 0.0f, device);
                auto runningInvStd = CNTK::Parameter(parameterShape, 1.0f, device);

                network = CNTK::BatchNormalization(
                        network,
                        scale,
                        bias,
                        runningMean,
                        runningInvStd,
                        input.Shape().Rank() > 1,
                        _normalizationTimeConstant,
                        0,
                        _epsilon,
                        _useCuDNN);

                return network;
            }
        };
    }
}