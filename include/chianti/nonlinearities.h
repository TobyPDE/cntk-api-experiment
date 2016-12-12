#pragma once

namespace Chianti
{
    namespace Nonlinearities
    {
        /*!
         * The ReLU non-linearity.
         */
        inline CNTK::Variable rectify(const CNTK::Variable & x)
        {
            return CNTK::ReLU(x);
        }

        /*!
         * No linearity / linear activation function.
         */
        inline CNTK::Variable linear(const CNTK::Variable & x)
        {
            return x;
        }
    }
}