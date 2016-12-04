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
    }
}