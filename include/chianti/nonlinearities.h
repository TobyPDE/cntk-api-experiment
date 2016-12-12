#pragma once

namespace Chianti
{
    namespace Nonlinearities
    {
        /*!
         * The ReLU non-linearity.
         */
        inline CNTK::FunctionPtr rectify(CNTK::FunctionPtr x)
        {
            return CNTK::ReLU(x);
        }

        /*!
         * No linearity / linear activation function.
         */
        inline CNTK::FunctionPtr linear(CNTK::FunctionPtr x)
        {
            return x;
        }
    }
}